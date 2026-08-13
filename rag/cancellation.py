"""생성 취소 (#30) — 진행 중 태스크 레지스트리 + 인스턴스 간 취소 신호 전파.

왜 rag/streaming.py가 아닌 별 모듈인가: streaming.py는 "연결이 끊겨도 생성은 완주한다"는
계약을 지키는 모듈이다(#26). 취소는 그 반대 방향 — 일부러 멈춘다 — 이고, 한 요청이 만든
태스크를 **다른 요청이 찾아와 죽이는** cross-request 개념이라 층위가 다르다. 같은 파일에
두면 _run_generation의 불변식을 읽을 때마다 "누가 밖에서 이 태스크를 건드릴 수 있나"를
함께 따져야 한다.

── 규약: pop-then-cancel ─────────────────────────────────
취소가 두 번 전달되면 _run_generation의 finally가 중간에 끊겨 리미터 반납·스팬 종료가
유실된다(실측 확인). 그래서 이 모듈은 레지스트리에서 **꺼내면서** 취소한다 — 두 번째
요청은 대상을 못 찾아 cancel()을 아예 호출하지 않는다. dict.pop과 task.cancel() 사이에
await가 없어야 그 원자성이 성립한다(단일 스레드 asyncio라 중간에 다른 요청이 끼지 못한다).

같은 이유로 태스크 자신도 정리(finally)에 진입하는 순간 unregister를 호출해야 한다.
그렇지 않으면 "답변은 done으로 커밋됐는데 finally에서 리미터를 반납하는 중"인 1ms짜리
창에 취소가 도착해 정리가 파괴된다(실측: 반납·스팬 모두 유실, 게다가 클라이언트는 이미
끝난 답변에 대해 204를 받는다). 자기 자신을 먼저 빼두면 그 요청은 404로 정확히 답한다.

이 레지스트리는 GC 방지 역할도 겸한다 — asyncio는 미참조 태스크를 실행 중에 GC할 수 있다.
그래서 streaming.py에 별도 참조 집합을 남기지 않는다(같은 목적의 자료구조 이중화 금지).

── 전파 ──────────────────────────────────────────────────
레지스트리는 프로세스 메모리에만 있다. 웹 인스턴스가 여럿이면 취소 요청이 태스크를 들고
있지 않은 인스턴스로 라우팅될 수 있어, 그때 Redis 채널로 넘긴다. 스티키 세션을 쓰지 않는
이유는 이슈 #30 참조(브라우저→인스턴스 고정이지 태스크→인스턴스 고정이 아니다).

Redis pub/sub은 전달 보장이 없다(fire-and-forget). 구독자가 없거나 순단이면 신호는 그냥
사라지고, 그 턴은 300초 스테일 스윕이 failed로 정리한다 — 즉 202는 "접수했다"이지
"취소됐다"가 아니다. 그래서 로컬 경로(204)는 Redis에 전혀 의존하지 않게 짰다.

신뢰 경계: **소유권 검증은 이 모듈이 하지 않는다.** cancel_local/request_cancel은 둘 다
message_id만 받으므로, 호출부(HTTP 엔드포인트)가 tenant·created_by를 먼저 검증한 뒤에만
불러야 한다 — 검증을 건너뛰면 남의 테넌트 생성을 id 추측만으로 죽일 수 있다(messages.id는
전 테넌트 공용 시퀀스). 여기서 tenant를 함께 들고 검사하지 않는 이유는 소유 규칙(_owned)의
사본을 메모리에 하나 더 만들면 동기화 부담이 세 곳으로 늘기 때문이다(routers/conversations.py
_owned ↔ rag/conversation.py ensure_conversation이 이미 이중화 상태).

반면 Redis 채널은 검증할 방법이 없다 — 사내망에서 접근 가능한 누구든 임의 message_id를
발행해 진행 중 생성을 죽일 수 있다. 리미터 ZSET이 이미 같은 수준으로 열려 있어(누구든
ZREM 가능) 동일 신뢰 구역으로 보고 수용한다. 가용성 침해만 가능하고 데이터 노출은 없다.
"""
import asyncio
import logging

from rag import clients

logger = logging.getLogger(__name__)

# 취소 신호 채널. kms: prefix는 리미터 키(kms:inflight:*)와 같은 관례.
CANCEL_CHANNEL = 'kms:cancel'

# 구독이 끊겼을 때 재시도 간격 — 짧게 잡는다. 이 루프가 죽으면 그 프로세스는 원격 취소를
# 영구히 놓치는데, 증상이 "취소가 간헐적으로 안 먹힌다"로만 나타나 진단이 어렵다.
SUBSCRIBE_RETRY_SECONDS = 3

# 진행 중 생성 태스크: assistant_message_id → task. 등록/해제는 rag/streaming.py가 한다.
_registry: dict[int, asyncio.Task] = {}


def register(message_id: int, task: asyncio.Task) -> None:
    """취소 대상으로 등록. 강한 참조를 들어 GC 방지 역할도 함께 한다."""
    _registry[message_id] = task


def unregister(message_id: int) -> None:
    """레지스트리에서 제거 (멱등). 태스크가 정리에 진입할 때, 그리고 완료 콜백에서 호출된다."""
    _registry.pop(message_id, None)


def cancel_local(message_id: int) -> bool:
    """이 프로세스가 그 태스크를 들고 있으면 취소한다. 취소했으면 True.

    pop과 cancel 사이에 await를 두지 말 것 — 그 원자성이 '따닥 두 번 눌러도 cancel은
    한 번'을 보장하는 유일한 근거다(모듈 docstring 참조).
    """
    task = _registry.pop(message_id, None)
    if task is None:
        return False
    task.cancel()
    return True


async def request_cancel(message_id: int) -> None:
    """다른 인스턴스에 취소를 요청한다 (로컬에 없을 때만 쓴다)."""
    # 공용 클라이언트를 호출 시점에 읽는다 — 모듈 속성으로 붙잡으면 테스트의 '교체'가 안 먹는다
    await clients.shared_redis.publish(CANCEL_CHANNEL, str(message_id))


async def subscribe_forever() -> None:
    """취소 채널 구독 루프 — 앱 수명 동안 하나만 돈다 (main.py lifespan).

    순단으로 구독이 끊기면 로그를 남기고 SUBSCRIBE_RETRY_SECONDS 후 다시 붙는다.
    재연결 없이 두면 한 번의 순단으로 그 프로세스가 재기동 전까지 원격 취소를 전부
    놓치는 조용한 장애가 된다.
    """
    while True:
        pubsub = clients.shared_redis.pubsub()   # 재연결 때마다 최신 클라이언트를 집는다
        try:
            await pubsub.subscribe(CANCEL_CHANNEL)
            async for raw in pubsub.listen():
                if raw['type'] != 'message':
                    continue          # subscribe 확인 메시지 등
                _handle_signal(raw['data'])
        except asyncio.CancelledError:
            raise                     # 앱 종료 — 재시도하지 않는다
        except Exception:
            logger.exception('취소 채널 구독이 끊겼다 — %s초 후 재연결', SUBSCRIBE_RETRY_SECONDS)
            await asyncio.sleep(SUBSCRIBE_RETRY_SECONDS)
        finally:
            try:
                await pubsub.aclose()
            except Exception:
                pass                  # 정리 실패가 재연결을 막지 않게


def _handle_signal(data) -> None:
    """채널 payload → 로컬 취소. 남의 신호(내가 안 들고 있는 id)는 조용히 무시한다."""
    try:
        message_id = int(data)
    except (TypeError, ValueError):
        logger.warning('취소 채널 payload 파싱 실패: %r', data)
        return
    if cancel_local(message_id):
        logger.info('원격 취소 신호로 생성 중단 (assistant_message_id=%s)', message_id)
