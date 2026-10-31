"""생성 중 재접속 — 턴 이벤트를 Redis Stream에 미러링하고, 새 연결이 그걸 재생한다.

인메모리 큐(rag/turn/streaming.py)와 이중 쓰기다 — Redis가 죽어도 진행 중인 턴의 원 연결은 끊기지 않게.
읽기는 XREAD(비파괴)라 여러 탭이 각자 전체를 받는다.
예외 방침: 쓰기 실패는 삼키고 그 턴만 폴링으로 강등한다(재접속은 부차 기능). 읽기 실패는 전파한다.
"""

import asyncio
import json
import logging
from collections.abc import AsyncIterator

from config import settings
from database import AsyncSessionLocal
from rag.models import Message

logger = logging.getLogger(__name__)

# 스트림 종료 마커 — 인메모리 sentinel(queue.put(None))에 해당한다. 클라이언트로 내보내지 않는다.
EVENT_END = '__end__'

# kms: 접두와 tenant_id 포함은 필수 — tests/conftest.py의 purge_tenant가 kms:*{tenant}*로 정리한다.
_KEY_PREFIX = 'kms:stream:'


def stream_key(tenant_id: str, message_id: int) -> str:
    return f'{_KEY_PREFIX}{tenant_id}:{message_id}'


def _redis():
    """공용 Redis 클라이언트를 호출 시점에 읽는다 — tests/conftest.py가 shared_redis를 통째로 교체한다."""
    from rag import clients

    return clients.shared_redis


class StreamWriter:
    """턴 이벤트를 Redis Stream에 배치 기록한다. 실패는 전부 삼킨다.

    토큰마다 XADD 왕복을 걸면 생성 지연이 늘어 배치로 보낸다. 첫 배치만 창 없이 즉시 보낸다.
    add()는 토큰 루프에서 await 없이 부를 수 있게 동기다.
    """

    def __init__(self, tenant_id: str, message_id: int) -> None:
        self.key = stream_key(tenant_id, message_id)
        self._buf: list[tuple[str, dict]] = []
        self._wake = asyncio.Event()
        self._closing = False
        # 한 번 실패하면 이 스트림은 구멍 난 것으로 보고 포기한다 — 이어 쓰면 중간이 빠진
        # 답변이 정상 완료로 재생된다.
        self._degraded = False
        # 참조를 들고 있어야 실행 중 태스크가 GC되지 않는다.
        self._task = asyncio.create_task(self._loop())

    def add(self, event: str, data: dict) -> None:
        """버퍼에 쌓기만 한다 — 순수 인메모리라 실패하지 않는다."""
        self._buf.append((event, data))
        self._wake.set()

    async def _loop(self) -> None:
        first = True
        while True:
            if not self._closing:
                await self._wake.wait()
                self._wake.clear()
            # 첫 배치는 창 없이 즉시. 이후에는 창만큼 모아 왕복을 줄인다.
            # 종료 중이면 창을 기다리지 않는다 — finally가 이걸 await하고 있다.
            if not first and not self._closing:
                await asyncio.sleep(settings.stream_resume_flush_seconds)
            first = False
            await self._flush()
            # 버퍼가 빈 것까지 확인하고 나간다 — _flush의 await 사이에 들어온 done·종료 마커를 잃지 않게.
            if self._closing and not self._buf:
                return

    async def _flush(self) -> None:
        if not self._buf:
            return
        batch, self._buf = self._buf, []
        if self._degraded:
            return  # 이미 구멍 났다 — 더 써봐야 반쪽 스트림만 그럴듯해진다
        try:
            pipe = _redis().pipeline(transaction=False)
            for event, data in batch:
                pipe.xadd(self.key, {'event': event, 'data': json.dumps(data, ensure_ascii=False)})
            # 매 flush마다 연장 — 마지막 flush 뒤 TTL만큼 지나 자체 소멸한다.
            pipe.expire(self.key, settings.stream_resume_ttl_seconds)
            # 타임아웃 필수 — shared_redis에 소켓 타임아웃이 없어, 멈춘 Redis가 _run_generation의
            # finally(aclose 경유)를 붙잡으면 리미터 반납이 늦어진다.
            await asyncio.wait_for(pipe.execute(), timeout=settings.stream_resume_flush_timeout_seconds)
        except Exception:
            # TimeoutError도 여기로 온다.
            self._degraded = True
            logger.warning('재접속 스트림 기록 실패 — 이 턴은 폴링으로 강등 (key=%s)', self.key)

    async def aclose(self) -> None:
        """종료 마커까지 내보내고 배치 루프를 끝낸다.

        cancel이 아니라 await로 기다린다 — 진행 중인 flush를 끊으면 마지막 배치가 유실된다.
        """
        # 구멍 난 스트림엔 종료 마커를 붙이지 않는다 — 반쪽이 정상 완료로 보이지 않게.
        if not self._degraded:
            self._buf.append((EVENT_END, {}))
        self._closing = True
        self._wake.set()
        try:
            await self._task
        except Exception:
            logger.warning('재접속 스트림 종료 실패 (key=%s)', self.key)

        if self._degraded:
            # 지워서 재접속을 404 → 폴링으로 보낸다. 이것도 실패하면 마커 없는 스트림이 TTL만큼 남는다.
            try:
                await asyncio.wait_for(_redis().delete(self.key), timeout=settings.stream_resume_flush_timeout_seconds)
            except Exception:
                logger.warning('구멍 난 재접속 스트림 삭제 실패 — TTL로 소멸 (key=%s)', self.key)


async def reconnect_reader(tenant_id: str, message_id: int) -> AsyncIterator[str]:
    """쌓인 이벤트를 처음부터 재생하고, 끝나지 않았으면 실시간으로 이어 붙인다.

    호출부가 소유 검증을 끝냈다는 전제다(messages.id는 전 테넌트 공용 시퀀스).
    """
    from rag.turn.streaming import EVENT_PING, sse_event  # 순환 import 회피

    key = stream_key(tenant_id, message_id)
    last_id = '0'
    block_ms = int(settings.stream_resume_block_seconds * 1000)

    while True:
        entries = await _redis().xread({key: last_id}, count=100, block=block_ms)
        if not entries:
            # 종료 마커 없이 턴이 끝났을 수 있다(부분 flush 실패·TTL 만료) — DB로 판정한다.
            if await _turn_finished(tenant_id, message_id):
                return
            # ping은 스트림에 저장하지 않고 리더가 합성한다 — 유휴마다 XADD하면 무한 누적된다.
            yield sse_event(EVENT_PING, {})
            continue

        for _, batch in entries:
            for entry_id, fields in batch:
                last_id = entry_id
                event = fields[b'event'].decode()
                if event == EVENT_END:
                    return  # 마커는 클라이언트로 내보내지 않는다
                yield sse_event(event, json.loads(fields[b'data']))


async def _turn_finished(tenant_id: str, message_id: int) -> bool:
    """종료 마커가 없을 때의 폴백 판정 — status가 generating이 아니면 끝난 것으로 본다.

    한계: 스테일 스윕이 살아 있는 생성을 failed로 바꾸면 스트림을 조기에 닫는다. 조기 종료는
    FE가 이력 재조회로 복구할 수 있지만 매달림은 TTL 내내 남으므로 이쪽을 택했다.
    """
    try:
        async with AsyncSessionLocal() as session:
            msg = await session.get(Message, message_id)
            return msg is None or msg.tenant_id != tenant_id or msg.status != 'generating'
    except Exception:
        logger.exception('턴 종료 판정 실패 (message_id=%s)', message_id)
        return False  # 판정 불가면 살아 있는 스트림을 끊지 않는 쪽으로
