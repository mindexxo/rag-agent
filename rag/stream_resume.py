"""생성 중 재접속 (#75) — 턴 이벤트를 Redis Stream에 미러링하고, 새 연결이 그걸 재생한다.

왜 rag/streaming.py가 아닌 별 모듈인가: streaming.py는 **한 요청 안에서** 생산자(태스크)와
로컬 리더(그 요청의 SSE 제너레이터)가 주고받는 계약을 지킨다. 재접속은 **다른 HTTP 요청**이
남이 만들고 있는 산출물을 찾아와 읽는 cross-request 개념이고, 그 사이를 잇는 매개체(Redis)가
프로세스 경계까지 가로지른다 — rag/cancellation.py가 같은 근거로 분리돼 있다.

── 왜 이중 쓰기인가 (단일 경로 기각) ────────────────────────
모든 소비를 Redis로 통일하면 코드는 더 깔끔하다. 그런데 **진행 중인 턴의 가용성**에서 진다.

리미터가 fail-open이라(rag/limiter.py) Redis 장애 중에도 턴은 계속 들어온다. 단일 경로면
그 턴들은 시작하자마자 스트림이 없어 답변이 통째로 안 보이고, 이미 생성 중이던 턴도 답변
중간에 끊긴다(사용자 눈앞에서 문장이 잘린다). 이중 쓰기는 인메모리가 계속 흘러 둘 다 안 보인다.
E2E 실측(2026-08-23): delta 3에서 Redis를 죽여도 589개 전부 도착하고 done·인용까지 정상.
Redis가 worker20(네트워크 너머)이라 순단이 가설이 아니라는 점이 이 판단의 근거다.

스티키 세션도 기각했다 — #30이 이미 검토한 대로 "브라우저→인스턴스 고정이지 태스크→인스턴스
고정이 아니다". Redis Stream은 인스턴스 수와 무관해 그 의존이 아예 없다.

── 왜 XREAD인가 (consumer group 아님) ───────────────────────
XREAD는 **비파괴 읽기**다. 커서(last_id)를 연결마다 각자 들고 있어 여러 탭이 동시에 붙어도
각자 전체를 받는다. XREADGROUP(컨슈머 그룹)은 작업 큐용이라 멤버끼리 엔트리를 나눠 갖는다 —
그걸 쓰면 asyncio.Queue를 재사용할 수 없었던 이유(1:1 소비)가 그대로 재현된다.

── 예외 방침: 쓰기는 삼키고, 읽기는 전파 ────────────────────
기존 Redis 사용 두 곳과 다른 세 번째 방침이라 근거를 남긴다.
  limiter     fail-open — 상한 없이 통과. 상담 중단을 더 큰 손실로 봤다(limiter.py docstring)
  cancellation   재연결 — 신호가 유실돼도 스테일 스윕이 수습한다
  여기(쓰기)      삼킴   — 재접속은 **부차 기능**이다. 미러링 실패가 진행 중인 생성을 막으면
                          본말전도다. 실패한 턴은 재접속만 못 하고 폴링으로 강등된다(= 현행 동작).
                          **한 번 실패하면 그 스트림은 포기한다**(_degraded): 실패한 배치는
                          되돌리지 않으므로 순단 후 복구되면 그 구간만 빠진 채 뒤가 이어져,
                          재접속 화면이 중간 빠진 답변을 정상 완료로 보여준다. 조용한 손상은
                          폴링 강등보다 나쁘다 — 차라리 스트림을 없애 404로 보낸다.
  여기(읽기)      전파   — 재접속 엔드포인트가 Redis를 못 읽으면 그 요청은 존재 이유를 잃는다.
"""
import asyncio
import json
import logging
from collections.abc import AsyncIterator

from config import settings
from database import AsyncSessionLocal
from rag.models import Message

logger = logging.getLogger(__name__)

# 스트림 종료 마커 — 인메모리 sentinel(queue.put(None))이 wire를 건널 때 취하는 유일한 형태.
# FE 어휘가 아니다: 리더가 이걸 보면 스트림을 닫을 뿐 클라이언트로 내보내지 않는다.
# 클라이언트에게 최종 이벤트는 여전히 done이다(rag/streaming.py의 FE 이벤트 계약 그대로).
EVENT_END = '__end__'

# kms: 접두 + tenant_id 포함은 **필수**다 — tests/conftest.py의 purge_tenant가
# scan_iter(match=f'kms:*{tenant}*')로 정리하므로, 빠지면 테스트 잔여물이 남아
# 다음 테스트를 간헐적으로 깨뜨린다 (limiter의 kms:inflight:* 와 같은 관례).
_KEY_PREFIX = 'kms:stream:'


def stream_key(tenant_id: str, message_id: int) -> str:
    return f'{_KEY_PREFIX}{tenant_id}:{message_id}'


def _redis():
    """공용 Redis 클라이언트 — **호출 시점에** 모듈 속성을 읽는다 (limiter._redis()와 같은 규약).

    import 시점에 지역변수로 붙잡으면 안 된다: 테스트가 루프 위생 때문에 clients.shared_redis를
    통째로 새 인스턴스로 교체하는데(tests/conftest.py), 붙잡은 참조는 그 교체를 못 받는다.
    """
    from rag import clients
    return clients.shared_redis


class StreamWriter:
    """턴 이벤트를 Redis Stream에 배치 기록한다. 실패는 전부 삼킨다.

    **배치가 선택이 아닌 이유**: Redis가 네트워크 너머(worker20)라 토큰마다 await XADD를 걸면
    그 왕복이 LLM 토큰 루프에 그대로 얹혀 latency_ms를 오염시킨다. 이 저장소는 같은 이유로
    캐시 적재를 done 뒤로 옮긴 전례가 있다(실측 0.5~1초, rag/streaming.py maybe_cache 주석).

    첫 배치는 즉시 내보낸다 — 첫 토큰이 체감 지연의 대부분이라, 그것까지 창(window)에
    묶으면 재접속을 위해 정상 경로를 느리게 만드는 셈이 된다.
    add()는 동기 함수다 — 토큰 루프에서 await 없이 부를 수 있어야 한다.
    """

    def __init__(self, tenant_id: str, message_id: int) -> None:
        self.key = stream_key(tenant_id, message_id)
        self._buf: list[tuple[str, dict]] = []
        self._wake = asyncio.Event()
        self._closing = False
        # 한 번이라도 기록에 실패하면 이 스트림은 **구멍 난 것**이라 되살리지 않는다.
        # 실패한 배치는 되돌리지 않으므로(스왑 후 execute), 순단됐다 복구되면 그 구간만
        # 빠진 채 뒤가 이어져 "meta → delta(일부 유실) → done"이 된다. 그대로 두면 재접속
        # 화면이 **중간이 빠진 답변을 정상 완료로** 보여준다 — 조용한 손상이라 폴링 강등보다 나쁘다.
        self._degraded = False
        # 참조를 인스턴스가 들고 있어야 GC되지 않는다 (cancellation 레지스트리가 태스크 참조를
        # 겸하는 것과 같은 이유 — asyncio는 미참조 태스크를 실행 중에 수거할 수 있다).
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
            # **버퍼가 빈 것까지 확인하고 나간다.** _flush()는 execute()에서 await하므로,
            # 그 사이에 들어온 이벤트가 버퍼에 남는다 — 취소·실패 경로는 마지막 delta와
            # done·종료 마커 사이가 짧아 이 창에 걸리기 쉽다. _closing만 보고 나가면
            # done과 마커를 통째로 잃고, 재접속 클라이언트가 done 없이 끊기는 화면을 본다
            # (FE 계약상 비정상 종료 — rag/streaming.py 이벤트 계약).
            if self._closing and not self._buf:
                return

    async def _flush(self) -> None:
        if not self._buf:
            return
        batch, self._buf = self._buf, []
        if self._degraded:
            return          # 이미 구멍 났다 — 더 써봐야 반쪽 스트림만 그럴듯해진다
        try:
            pipe = _redis().pipeline(transaction=False)
            for event, data in batch:
                pipe.xadd(self.key, {'event': event,
                                     'data': json.dumps(data, ensure_ascii=False)})
            # 매 flush마다 갱신 — 생성 중엔 계속 연장되고, 마지막 flush 뒤로는 갱신이 멈춰
            # TTL만큼 뒤에 자체 소멸한다(리미터의 EXPIRE 방식. arq cron 스윕은 턴 단위 짧은
            # 수명엔 과하다).
            pipe.expire(self.key, settings.stream_resume_ttl_seconds)
            # **타임아웃이 필수다.** shared_redis에 소켓 타임아웃이 없어, Redis가 에러 없이
            # 멈추면 여기서 무한 대기한다. 이 함수는 _run_generation의 finally(aclose 경유)에서도
            # 불리므로, 그 대기가 곧 리미터 반납·스팬 종료 지연이 된다 — 슬롯이 물린 채 다른
            # 상담원이 429를 받는다. 예외만 삼키고 hang은 방치하면 원칙이 반쪽이다.
            await asyncio.wait_for(pipe.execute(),
                                   timeout=settings.stream_resume_flush_timeout_seconds)
        except Exception:
            # 삼킨다 — 사유는 모듈 docstring. 이 턴은 재접속만 못 하고 폴링으로 강등된다.
            # TimeoutError도 여기 합류한다(OSError 계열).
            self._degraded = True
            logger.warning('재접속 스트림 기록 실패 — 이 턴은 폴링으로 강등 (key=%s)', self.key)

    async def aclose(self) -> None:
        """종료 마커까지 내보내고 배치 루프를 끝낸다.

        task.cancel()이 아니라 await로 기다린다 — 진행 중인 flush를 끊으면 마지막 배치가
        유실돼 재접속 화면이 답변 끝부분을 잃는다.
        """
        # 구멍 난 스트림엔 종료 마커를 붙이지 않는다. 붙이면 반쪽짜리가 "정상 완료"로 보인다 —
        # 마커가 없으면 리더가 done 없이 닫히고, FE는 그걸 비정상 종료로 보고 이력을 재조회한다
        # (rag/streaming.py 이벤트 계약: "done 없이 끊기면 비정상 종료").
        if not self._degraded:
            self._buf.append((EVENT_END, {}))
        self._closing = True
        self._wake.set()
        try:
            await self._task
        except Exception:
            logger.warning('재접속 스트림 종료 실패 (key=%s)', self.key)

        if self._degraded:
            # 아예 지워 404 → 폴링으로 보내는 게 가장 깨끗하다. Redis가 아직 죽어 있으면
            # 이것도 실패하는데, 그때는 마커 없는 스트림이 TTL만큼 남고 리더가 done 없이 닫힌다.
            try:
                await asyncio.wait_for(_redis().delete(self.key),
                                       timeout=settings.stream_resume_flush_timeout_seconds)
            except Exception:
                logger.warning('구멍 난 재접속 스트림 삭제 실패 — TTL로 소멸 (key=%s)', self.key)


async def reconnect_reader(tenant_id: str, message_id: int) -> AsyncIterator[str]:
    """쌓인 이벤트를 처음부터 재생하고, 끝나지 않았으면 실시간으로 이어 붙인다.

    **소유 검증은 호출부(라우터)가 이미 끝냈다는 전제**다 — messages.id가 전 테넌트 공용
    시퀀스라 검증을 건너뛰면 id 추측만으로 남의 생성을 엿볼 수 있다. 여기서 다시 검사하지
    않는 이유는 소유 판정의 사본을 만들지 않기 위해서다(cancellation과 같은 책임 분리).

    항상 '0'(처음)부터 읽는다 — 새로고침은 클라이언트 상태가 날아간 상황이라 부분 재생이
    의미가 없다. SSE 표준의 Last-Event-ID 기반 증분 재개는 FE가 id를 보관하게 된 뒤에 얹는다.
    """
    from rag.streaming import EVENT_PING, sse_event   # 순환 import 방지 — 호출 시점 지연 로드

    key = stream_key(tenant_id, message_id)
    last_id = '0'
    block_ms = int(settings.stream_resume_block_seconds * 1000)

    while True:
        entries = await _redis().xread({key: last_id}, count=100, block=block_ms)
        if not entries:
            # 새 엔트리 없이 블록이 만료됐다. 턴이 이미 끝났는데 종료 마커가 없으면
            # (부분 flush 실패·TTL 만료) 여기서 안 끊으면 영원히 매달린다 — DB로 대신 판정한다.
            if await _turn_finished(tenant_id, message_id):
                return
            # ping은 스트림에 저장하지 않는다 — 유휴마다 XADD하면 무한 누적된다.
            # 리더가 자기 유휴 시점에 합성하는 것은 queue_reader와 같다.
            # 생성 경로(queue_reader)보다 촘촘하지만 무해하다 — 이 창은 종료 판정 주기를
            # 겸하고 있어 ping 주기와 따로 잡았다(config.stream_resume_block_seconds).
            yield sse_event(EVENT_PING, {})
            continue

        for _, batch in entries:
            for entry_id, fields in batch:
                last_id = entry_id
                event = fields[b'event'].decode()
                if event == EVENT_END:
                    return          # 마커는 클라이언트로 안 나간다 — queue_reader의 None과 동격
                yield sse_event(event, json.loads(fields[b'data']))


async def _turn_finished(tenant_id: str, message_id: int) -> bool:
    """종료 마커가 없을 때의 폴백 판정 — 이 턴이 이미 끝났다고 볼 것인가.

    Redis 쓰기가 **중간에** 실패하면(초반 delta는 성공, 이후 순단) 키는 존재하는데 종료 마커가
    영영 안 붙는다. 그 경우 리더가 TTL 내내 ping만 내보내며 매달려, 생성이 끝났는데도 안 끝난
    것처럼 보인다. 그래서 status를 대신 본다.

    **알려진 한계**: status != 'generating'이 "정말 끝났다"와 항상 같지는 않다. 스테일 스윕
    (GENERATION_STALE_SECONDS=500, rag/turn_state.py)이 **살아 있는 생성**을 failed로 바꿔놓을
    수 있고 — GPU 포화로 생성이 500초를 넘으면 실제로 그렇다 — 그때 토큰 간격이 block 주기보다
    벌어지면 이 폴백이 아직 도는 턴의 스트림을 조기에 닫는다. 취소 엔드포인트는 같은 경합을
    "태스크가 손에 있으면 DB가 뭐라 하든" 규칙으로 피하지만(routers/conversations.py), 재접속은
    **다른 인스턴스일 수 있어** 그 레지스트리를 못 본다.
    감수하는 쪽을 택한 이유: 조기 종료는 FE가 이력 재조회로 복구할 수 있는 반면, 매달림은
    TTL 내내 "끝나지 않는 답변"으로 남는다. 덜 나쁜 실패를 고른 것이지 안전한 판정이 아니다.
    """
    try:
        async with AsyncSessionLocal() as session:
            msg = await session.get(Message, message_id)
            return msg is None or msg.tenant_id != tenant_id or msg.status != 'generating'
    except Exception:
        logger.exception('턴 종료 판정 실패 (message_id=%s)', message_id)
        return False        # 판정 불가면 계속 기다린다 — 살아 있는 스트림을 끊지 않는 쪽으로
