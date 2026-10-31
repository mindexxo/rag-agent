"""생성 취소 — 진행 중 태스크 레지스트리와 인스턴스 간 취소 신호(Redis pub/sub) 전파.

pop-then-cancel: 레지스트리에서 꺼내면서 취소한다. 취소가 두 번 전달되면 _run_generation의
finally가 끊겨 리미터 반납·스팬 종료가 유실되므로, 두 번째 요청은 대상을 못 찾아야 한다.
같은 이유로 태스크 자신도 finally에 들어서는 즉시 unregister한다.
레지스트리는 실행 중 태스크의 GC 방지 참조도 겸한다 — streaming.py에 참조 집합을 따로 두지 말 것.

Redis 신호는 전달 보장이 없다 — 유실되면 스테일 스윕이 정리한다(202는 "접수"이지 "취소됨"이 아니다).
소유권 검증은 하지 않는다 — 호출부가 tenant·소유자·상태를 먼저 확인한 뒤에만 불러야 한다
(messages.id는 전 테넌트 공용 시퀀스라 id 추측으로 남의 생성을 죽일 수 있다).
Redis 채널은 인증이 없다 — Redis에 접근할 수 있는 주체는 같은 신뢰 구역으로 본다(취소로 가용성만 침해 가능, 데이터 노출은 없다).
"""

import asyncio
import logging

from rag import clients

logger = logging.getLogger(__name__)

# kms: prefix는 리미터 키(kms:inflight:*)와 같은 관례.
CANCEL_CHANNEL = 'kms:cancel'

# 구독 재연결 간격 — 지수 백오프(붙으면 초기화).
SUBSCRIBE_RETRY_MIN_SECONDS = 1
SUBSCRIBE_RETRY_MAX_SECONDS = 10  # 상한이 곧 '복구 후 원격 취소가 다시 먹기까지의 최대 지연'

# 진행 중 생성 태스크: assistant_message_id → task. 등록/해제는 rag/turn/streaming.py가 한다.
_registry: dict[int, asyncio.Task] = {}


def register(message_id: int, task: asyncio.Task) -> None:
    """취소 대상으로 등록. 강한 참조라 GC 방지 역할도 한다."""
    _registry[message_id] = task


def unregister(message_id: int) -> None:
    """레지스트리에서 제거(멱등)."""
    _registry.pop(message_id, None)


def cancel_local(message_id: int) -> bool:
    """이 프로세스가 그 태스크를 들고 있으면 취소한다. 취소했으면 True.

    pop과 cancel 사이에 await를 두지 말 것 — 그 원자성이 '두 번 눌러도 cancel은 한 번'의 근거다.
    """
    task = _registry.pop(message_id, None)
    if task is None:
        return False
    task.cancel()
    return True


async def request_cancel(message_id: int) -> None:
    """다른 인스턴스에 취소를 요청한다(로컬에 없을 때만 쓴다)."""
    # 공용 클라이언트를 호출 시점에 읽는다 — 모듈 속성으로 붙잡으면 테스트의 '교체'가 안 먹는다
    await clients.shared_redis.publish(CANCEL_CHANNEL, str(message_id))


async def subscribe_forever() -> None:
    """취소 채널 구독 루프 — 앱 수명 동안 하나만 돈다(main.py lifespan).

    끊기면 백오프로 무한히 다시 붙는다 — 포기하면 그 프로세스는 원격 취소를 조용히 전부 놓친다.
    listen이 정상 종료해도 대기를 거친다(없으면 busy-loop).
    """
    delay = SUBSCRIBE_RETRY_MIN_SECONDS
    while True:
        pubsub = None
        try:
            pubsub = clients.shared_redis.pubsub()  # 재연결 때마다 최신 클라이언트를 집는다
            await pubsub.subscribe(CANCEL_CHANNEL)
            delay = SUBSCRIBE_RETRY_MIN_SECONDS  # 붙었으니 백오프 초기화
            async for raw in pubsub.listen():
                if raw['type'] != 'message':
                    continue  # subscribe 확인 메시지 등
                _handle_signal(raw['data'])
        except asyncio.CancelledError:
            raise  # 앱 종료 — 재시도하지 않는다
        except Exception:
            # pubsub() 생성까지 try 안에 둔다 — 밖이면 거기서 터질 때 루프가 통째로 죽는다.
            logger.exception('취소 채널 구독이 끊겼다 — %s초 후 재연결', delay)
        finally:
            if pubsub is not None:
                try:
                    await pubsub.aclose()
                except Exception:
                    pass  # 정리 실패가 재연결을 막지 않게
        await asyncio.sleep(delay)
        delay = min(delay * 2, SUBSCRIBE_RETRY_MAX_SECONDS)


def _handle_signal(data) -> None:
    """채널 payload → 로컬 취소. 남의 신호(내가 안 들고 있는 id)는 조용히 무시한다."""
    try:
        message_id = int(data)
    except TypeError, ValueError:
        logger.warning('취소 채널 payload 파싱 실패: %r', data)
        return
    if cancel_local(message_id):
        logger.info('원격 취소 신호로 생성 중단 (assistant_message_id=%s)', message_id)
