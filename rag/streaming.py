"""SSE 턴 스트리밍 — 이벤트 어휘 + 두 경로의 수명 관리 (#26).

`/kms/query`는 SSE만 지원한다(비스트리밍 JSON 경로는 #26에서 삭제). 경로는 둘이고,
갈리는 기준은 **답이 이미 확정돼 있느냐**다.

  즉시 경로  immediate_stream   입력차단 / 캐시히트 / 근거없음 — LLM 호출 없음
  생성 경로  spawn_generation + queue_reader   knowledge / other — LLM 호출

왜 생성 경로만 백그라운드 태스크인가: StreamingResponse 제너레이터 안에서 LLM을 호출하면
클라이언트가 연결을 끊는 순간 생성이 중단돼 ① 답변 유실 ② DB 미저장 ③ 캐시 미저장
④ generating 행 고착이 된다. 이미 GPU를 쓴 결과를 버리는 셈이라, **연결 수명과 생성 수명을
분리**해 태스크가 끝까지 완주하게 한다. 리미터 슬롯·OTel 루트 스팬의 소유권도 함께 넘겨받는다
(실제 GPU 점유 구간을 반영해야 하므로).

── FE 이벤트 계약 ──────────────────────────────────────────
봉투는 `event: <name>\\ndata: <json>\\n\\n`. 이름은 아래 EVENT_* 상수가 유일한 정의점.

  즉시 경로   meta → sources(확정) → token(1건) → done
  생성 경로   meta → sources(낙관) → token×N → [sources([]) ← 거절일 때만] → done
  오류        … → error → done          (done은 "스트림 종료" 신호이지 성공 신호가 아니다)

**sources는 여러 번 올 수 있고 마지막 것이 유효하다.** 생성 경로는 첫 토큰보다 먼저 인용을
보내야 FE가 각주 영역을 그릴 수 있는데, 그 시점엔 LLM이 스스로 거절할지 알 수 없다
(is_refusal은 답변 완성 후 판정). 그래서 낙관적으로 보내고 거절로 판명되면 빈 배열로 정정한다.
즉시 경로는 답이 이미 확정돼 있어 처음부터 정확한 값을 1회만 보낸다 — 두 경로의 타이밍 차이는
버그가 아니라 의도된 설계라서, 공통화는 이벤트 조립까지만 하고 제어 흐름은 각자 둔다.
"""
import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator

from config import settings
from database import AsyncSessionLocal
from rag import cancellation, limiter, otel
from rag.limiter import Lease
from rag.answer_check import is_refusal
from rag.service import PreparedRag, RagService

logger = logging.getLogger(__name__)

# SSE 이벤트 이름 — FE 계약의 단일 정의점 (문자열 리터럴 산재 방지)
EVENT_META = 'meta'
EVENT_SOURCES = 'sources'
EVENT_TOKEN = 'token'
EVENT_ERROR = 'error'
EVENT_DONE = 'done'

# 태스크 참조 유지(GC 방지)와 취소 대상 색인은 rag/cancellation.py의 레지스트리가 겸한다 —
# 같은 목적의 자료구조를 둘로 두지 않는다. 취소 규약(pop-then-cancel)은 그쪽 docstring 참조.


def sse_event(event: str, data) -> str:
    """SSE 한 이벤트를 봉투 형식 문자열로 만든다.
    event: 이름 / data: JSON payload / 빈 줄로 이벤트 종료.
    """
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _meta_event(prepared: PreparedRag) -> str:
    return sse_event(EVENT_META, {
        "conversation_id": prepared.conversation_id,
        "assistant_message_id": prepared.assistant_message_id,   # FE 재접속·상태조회용
        "cached": prepared.is_cache_hit,
        "cache_kind": prepared.cache_kind,
        "reason": "no_evidence" if prepared.no_evidence else "ok",
    })


def _sources_event(prepared: PreparedRag, answer: str | None = None) -> str:
    """인용 이벤트. answer를 주면(즉시 경로 — 답 확정 후) 거절 판정까지 반영해 정확한 값을 낸다.
    answer=None(생성 경로 프리앰블)은 아직 답이 없어 근거 유무만 반영한다 — 거절 정정은 뒤에서.
    """
    hide = prepared.no_evidence or (answer is not None and is_refusal(answer))
    return sse_event(EVENT_SOURCES, [] if hide else [s.model_dump() for s in prepared.sources])


def _elapsed_ms(t_request: float) -> int:
    """사용자 체감 응답시간 — prepare(인텐트·검색·condense)를 포함한 요청 시작 기준.
    t_request는 호출부가 time.monotonic()으로 찍은 값이어야 한다 (기준 시계 일치)."""
    return int((time.monotonic() - t_request) * 1000)


async def immediate_stream(service: RagService, prepared: PreparedRag, session,
                           t_request: float) -> AsyncIterator[str]:
    """즉시 경로(입력차단·캐시히트·근거없음): 답이 확정 → 인라인 저장·스트림. 태스크 불필요.

    persist-before-stream(#16): 답이 이미 확정된 경로이므로 저장·commit을 먼저 한다 —
    재생 중 disconnect여도 턴이 남고(이력 구멍 방지), get_semantic이 잡은 hit_count
    행 잠금도 스트림 전에 풀린다(동시 히트가 첫 클라이언트 수신 속도에 직렬화되던 문제).
    """
    answer = ''.join([chunk async for chunk in service.generate(prepared)])
    # 저장 실패가 이미 확정된 답변의 '전달'까지 막으면 안 된다 (fail-open — 리뷰 반영).
    # StreamingResponse는 첫 yield 전에 200 헤더가 이미 나가므로, 여기서 예외가 새면
    # 사용자는 빈 스트림을 받는다. 실패는 로그만 남기고 전달은 계속한다.
    try:
        await service.save(prepared, answer, latency_ms=_elapsed_ms(t_request))
        await session.commit()
    except Exception:
        logger.exception("즉시 경로 저장 실패 — 답변 전달은 계속 (conversation=%s)",
                         prepared.conversation_id)

    yield _meta_event(prepared)
    yield _sources_event(prepared, answer)
    # 확정된 답변은 한 번에 보낸다 — 쪼개도 await 없이 연달아 나가 같은 청크로 도착하고,
    # FE가 자체 타자기(버퍼 길이 기준)로 그리므로 이벤트 개수는 렌더에 영향이 없다 (#26).
    yield sse_event(EVENT_TOKEN, {"text": answer})
    yield sse_event(EVENT_DONE, {})


def spawn_generation(prepared: PreparedRag, queue: asyncio.Queue, lease: Lease,
                     t_request: float, root_span) -> asyncio.Task:
    """생성 태스크를 띄우고 참조를 보관한다. 호출자는 lease.handed_off를 True로 이미 세워둘 것.

    이 태스크가 리미터 반납과 OTel 루트 스팬 종료를 이어받는다 — 요청이 먼저 반환되므로
    요청 쪽에서 정리하면 생성 도중에 슬롯이 풀리고 턴 duration이 잘린다.
    """
    task = asyncio.create_task(_run_generation(prepared, queue, lease, t_request, root_span))
    message_id = prepared.assistant_message_id
    cancellation.register(message_id, task)
    # 완료 콜백은 백업이다 — 태스크는 정리에 진입할 때 스스로 unregister한다(_run_generation).
    # 그래도 남겨두는 이유: 등록 직후 예외로 죽는 등 finally를 못 타는 경로가 있으면 누수된다.
    task.add_done_callback(lambda _t: cancellation.unregister(message_id))
    return task


async def _finalize_out_of_band(lease: Lease, prepared: PreparedRag, answer: str,
                                status: str, latency_ms: int | None) -> None:
    """비정상 종료(취소·실패)의 상태 기록 — 새 세션으로, 실패는 삼킨다.

    본 흐름의 세션은 이미 롤백/닫힘 상태일 수 있어 자기 세션을 새로 연다.
    기록이 실패하면 그 턴은 generating으로 남고 스테일 스윕(rag.conversation.GENERATION_STALE_SECONDS, cron 5분 주기)이 failed로 정리한다 — 의도(취소 vs
    실패)는 잃지만 고착은 남지 않는다. 상태 기록 실패가 정리(finally)를 막아선 안 되므로 삼킨다.
    """
    try:
        async with AsyncSessionLocal() as session:
            await RagService(tenant_id=lease.tenant_id, session=session).finalize(
                prepared, answer, status=status, latency_ms=latency_ms)
            await session.commit()
    except Exception:
        logger.exception("%s 상태 기록 실패 (tenant=%s, conversation=%s)",
                         status, lease.tenant_id, prepared.conversation_id)


async def _run_generation(prepared: PreparedRag, queue: asyncio.Queue, lease: Lease,
                          t_request: float, root_span) -> None:
    """백그라운드: 생성 → 큐로 토큰 전달 → 완료 시 DB finalize. 연결과 무관하게 끝까지.
    자기 세션의 RagService 사용 (요청 세션은 응답 종료 시 닫히므로).
    """
    parts = []
    try:
        async with AsyncSessionLocal() as session:
            svc = RagService(tenant_id=lease.tenant_id, session=session)
            with otel.span('generate', 'LLM') as sp:   # 배경 태스크 — 컨텍스트 복사로 kms.query의 자식 (#7)
                async for chunk in svc.generate(prepared):
                    parts.append(chunk)
                    await queue.put((EVENT_TOKEN, {"text": chunk}))
                answer = "".join(parts)
                otel.set_attrs(sp, {otel.LLM_MODEL: settings.vllm_model, otel.OUTPUT_VALUE: otel.clip(answer)})

            # LLM이 스스로 거절(규칙 3)한 답변이면 이미 내보낸 인용을 정정
            if is_refusal(answer):
                await queue.put((EVENT_SOURCES, []))
            await svc.finalize(prepared, answer, status="done", latency_ms=_elapsed_ms(t_request))
            await session.commit()
    except asyncio.CancelledError:
        # 명시적 취소(#30). CancelledError는 BaseException 계열이라 아래 except Exception이
        # 잡지 못한다 — 이 절이 없으면 finalize를 못 타고 generating으로 남아 스테일 스윕이
        # failed로 만든다(사용자는 취소했는데 화면엔 실패).
        # 취소 예외는 한 번만 전달되므로 여기서 잡은 뒤의 await(세션·commit)는 정상 완료된다 — 실측 확인.
        # 낙관적으로 내보낸 인용을 정정한다 — 거절 판정(위)과 같은 이유·같은 방식.
        # finalize가 status != done이면 sources를 []로 저장하므로, 정정하지 않으면 취소 직후
        # 화면엔 인용이 붙어 있는데 새로고침하면 사라지는 불일치가 남는다(실기동 확인).
        await queue.put((EVENT_SOURCES, []))
        await _finalize_out_of_band(lease, prepared, "".join(parts), "cancelled",
                                    _elapsed_ms(t_request))
        raise                                    # 태스크가 '취소됨'으로 끝나도록 재전파
    except Exception:
        logger.exception("답변 생성 실패 (tenant=%s, conversation=%s)", lease.tenant_id, prepared.conversation_id)
        await _finalize_out_of_band(lease, prepared, "", "failed", None)
        # 예외 원문은 서버 로그로만 — str(exc)에 내부 경로·설정이 섞일 수 있어 클라이언트에 노출 금지
        await queue.put((EVENT_ERROR, {"code": "generation_failed",
                                       "message": "답변 생성 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요."}))
    finally:
        # 취소 대상에서 먼저 빠진다 (#30) — 반드시 아래 await들보다 앞, 그리고 동기 문장으로.
        # 답변이 done으로 커밋된 뒤 리미터를 반납하는 찰나에 취소가 도착하면 이 finally가
        # 중간에 끊겨 반납·스팬 종료가 유실된다(실측). 자기를 먼저 빼두면 그 요청은 대상을
        # 못 찾아 cancel()을 호출하지 않고, 상태(done)를 근거로 404를 받는다.
        cancellation.unregister(prepared.assistant_message_id)
        await queue.put(None)                    # 리더 종료 sentinel — finalize·commit '뒤'에 넣는다
        await limiter.release(lease)              # 리미터는 태스크 수명 끝에 (실제 GPU 점유 구간)
        root_span.end()                           # 핸드오프된 턴 루트 스팬 — duration=턴 전체 (#7)


async def queue_reader(prepared: PreparedRag, queue: asyncio.Queue) -> AsyncIterator[str]:
    """생성 경로 SSE: 큐 이벤트를 흘린다 (읽기 전용). 연결이 끊겨도 태스크엔 영향 없음.

    그 무영향이 성립하는 근거가 **큐가 unbounded**라는 것이다. 소비가 멈춰도 생산자의
    `await queue.put(...)`이 블록되지 않아 태스크가 완주하고 finalize·리미터 반납이 보장된다.
    여기에 maxsize를 걸면 읽지 않는 큐가 차는 순간 태스크가 put에서 영구 대기하고 finally가
    실행되지 않는다 — generating 고착(스테일 스윕, cron 5분 주기)·리미터 슬롯(120초 prune)은 회수 장치가
    있지만 **태스크 세션의 DB 커넥션은 회수 장치가 없어 풀이 영구히 잠식된다.**
    쌓이는 양은 max_tokens 상한이 걸린 텍스트라 생성 1건당 수 KB에 불과하다.
    """
    yield _meta_event(prepared)
    yield _sources_event(prepared)
    while True:
        item = await queue.get()
        if item is None:
            break
        event, data = item
        yield sse_event(event, data)
    yield sse_event(EVENT_DONE, {})
