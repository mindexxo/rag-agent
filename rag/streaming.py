"""SSE 턴 스트리밍 — 이벤트 어휘와 두 경로(즉시·생성)의 수명 관리.

  즉시 경로  immediate_stream   입력차단 / 캐시히트 / 근거없음 — LLM 호출 없음
  생성 경로  spawn_generation + queue_reader   knowledge / other — LLM 호출

생성 경로는 백그라운드 태스크다 — 연결이 끊겨도 생성·저장·캐시 적재가 완주하도록 연결 수명과
생성 수명을 분리한다. 리미터 슬롯과 OTel 루트 스팬도 태스크가 넘겨받는다.

── FE 이벤트 계약 ─────────────────────────
봉투는 `event: <name>\\ndata: <json>\\n\\n`. 이름은 아래 EVENT_* 상수가 정의점.

  공통       meta → delta×N → done
  오류       … → error → done          (done은 "스트림 종료" 신호이지 성공 신호가 아니다)
  ping       유휴 시 수시 — 클라이언트는 무시 (프록시 유휴 종료 대비, 생성 경로만)

- done이 최종 상태를 싣는다: {finish_reason, latency_ms, citations}. finish_reason 어휘는
  messages.status와 같고, citations는 실제 인용된 출처만이다.
- done은 생성 태스크가 값과 함께 큐에 넣고 queue_reader는 전달만 한다 — 리더가 빈 done을
  합성하지 않는다. done 없이 스트림이 끊기면 비정상 종료다.
"""

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass

from config import settings
from database import AsyncSessionLocal
from rag import cancellation, limiter, metrics, otel, stream_resume
from rag.citation_tail import TailSplitter, resolve_citations
from rag.limiter import Lease
from rag.models import TurnStatus
from rag.prepared import PreparedRag
from rag.service import RagService
from schemas.kms import SourceCitation

logger = logging.getLogger(__name__)

# SSE 이벤트 이름 — FE 계약의 정의점
EVENT_META = 'meta'
EVENT_DELTA = 'delta'
EVENT_PING = 'ping'
EVENT_ERROR = 'error'
EVENT_DONE = 'done'

# /kms/query와 재접속 엔드포인트가 공유한다 — 갈리면 한쪽만 프록시 버퍼링에 걸린다.
SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


def sse_event(event: str, data) -> str:
    """SSE 한 이벤트를 봉투 형식 문자열로 만든다."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _meta_payload(prepared: PreparedRag) -> dict:
    """meta 페이로드 — 로컬 리더와 재접속 스트림 기록이 공유한다."""
    return {
        "conversation_id": prepared.conversation_id,
        "assistant_message_id": prepared.assistant_message_id,
        "cached": prepared.is_cache_hit,
        "cache_kind": prepared.cache_kind,
        "reason": "no_evidence" if prepared.no_evidence else "ok",
    }


def _meta_event(prepared: PreparedRag) -> str:
    return sse_event(EVENT_META, _meta_payload(prepared))


@dataclass
class TurnResult:
    """턴 하나의 최종 결과 — done 페이로드·루트 스팬·DB 저장이 모두 이 한 객체를 소비해 어긋나지 않게 한다.

    citations는 실제 인용된 출처만 담는다(후보 전체가 아니라).
    """

    answer: str
    citations: list[SourceCitation]
    finish_reason: TurnStatus  # 어휘 = turn_state.TERMINAL
    latency_ms: int | None


def _done_payload(result: TurnResult) -> dict:
    return {
        "finish_reason": result.finish_reason,
        "latency_ms": result.latency_ms,
        "citations": [c.model_dump() for c in result.citations],
    }


def _elapsed_ms(t_request: float) -> int:
    """요청 시작 기준 체감 응답시간. t_request는 time.monotonic()으로 찍은 값이어야 한다."""
    return int((time.monotonic() - t_request) * 1000)


def _record_turn_result(root_span, result: TurnResult) -> None:
    """턴 결과를 루트 kms.query 스팬에 기록한다. is_recording 가드를 먼저 둬 no-op이면 clip 비용도 없게 한다."""
    if not root_span.is_recording():
        return
    otel.set_attrs(
        root_span,
        {
            otel.OUTPUT_VALUE: otel.clip(result.answer),
            'kms.status': result.finish_reason,
            'kms.latency_ms': result.latency_ms,
            'kms.cited_docs': [c.filename for c in result.citations],
        },
    )


async def immediate_stream(
    service: RagService, prepared: PreparedRag, session, t_request: float, root_span
) -> AsyncIterator[str]:
    """즉시 경로(입력차단·캐시히트·근거없음): 답이 확정돼 있어 태스크 없이 저장 후 스트림한다.

    저장·commit을 스트림보다 먼저 한다 — disconnect여도 턴이 남고, get_semantic의 hit_count 행
    잠금도 스트림 전에 풀린다. 루트 스팬 종료를 이어받는다(핸들러 반환 뒤에 돌기 때문). 첫 바이트
    전에 끊겨 iterate가 안 되면 저장·스팬이 함께 유실된다.
    """
    try:
        answer = ''.join([chunk async for chunk in service.generate(prepared)])
        latency_ms = _elapsed_ms(t_request)  # 저장·스팬·done이 같은 값을 쓰도록 한 번만 계산
        # 즉시 경로의 인용 = prepared.sources 그대로 — 캐시 히트는 인용만 남긴 복원값, 차단·근거없음은 빈 목록.
        result = TurnResult(
            answer=answer,
            citations=list(prepared.sources),
            finish_reason=prepared.terminal_status,
            latency_ms=latency_ms,
        )
        # 저장 실패가 확정된 답변의 전달까지 막으면 안 된다 — 로그만 남긴다. 자리표시는 이미
        # 커밋돼 있어 generating으로 남고 스테일 스윕이 회수한다.
        try:
            await service.finalize(
                prepared, answer, result.citations, status=prepared.terminal_status, latency_ms=latency_ms
            )
            await session.commit()
        except Exception:
            logger.exception("즉시 경로 저장 실패 — 답변 전달은 계속 (conversation=%s)", prepared.conversation_id)
        _record_turn_result(root_span, result)

        yield _meta_event(prepared)
        # 확정된 답변은 한 번에 보낸다 — FE가 자체 타자기 효과로 그린다.
        yield sse_event(EVENT_DELTA, {"text": answer})
        yield sse_event(EVENT_DONE, _done_payload(result))
    finally:
        root_span.end()  # 핸드오프된 턴 루트 스팬


def spawn_generation(
    prepared: PreparedRag, queue: asyncio.Queue, lease: Lease, t_request: float, root_span
) -> asyncio.Task:
    """생성 태스크를 띄우고 참조를 보관한다. 호출자는 lease.handed_off를 True로 이미 세워둘 것.

    이 태스크가 리미터 반납과 OTel 루트 스팬 종료를 이어받는다 — 요청이 먼저 반환되므로
    요청 쪽에서 정리하면 생성 도중에 슬롯이 풀리고 턴 duration이 잘린다.
    """
    task = asyncio.create_task(_run_generation(prepared, queue, lease, t_request, root_span))
    message_id = prepared.assistant_message_id
    cancellation.register(message_id, task)
    # 백업 — 태스크는 finally에서 스스로 unregister하지만, finally를 못 타는 경로의 누수를 막는다.
    task.add_done_callback(lambda _t: cancellation.unregister(message_id))
    return task


async def _finalize_out_of_band(
    lease: Lease, prepared: PreparedRag, answer: str, status: TurnStatus, latency_ms: int | None
) -> None:
    """비정상 종료(취소·실패)의 상태 기록 — 새 세션으로, 실패는 삼킨다.

    본 흐름의 세션은 이미 롤백/닫힘 상태일 수 있다. 기록이 실패해도 스테일 스윕이 failed로
    정리하므로, 정리(finally)를 막지 않게 삼킨다.
    """
    try:
        async with AsyncSessionLocal() as session:
            await RagService(tenant_id=lease.tenant_id, session=session).finalize(
                prepared, answer, citations=[], status=status, latency_ms=latency_ms
            )
            await session.commit()
    except Exception:
        logger.exception(
            "%s 상태 기록 실패 (tenant=%s, conversation=%s)", status, lease.tenant_id, prepared.conversation_id
        )


async def _run_generation(
    prepared: PreparedRag, queue: asyncio.Queue, lease: Lease, t_request: float, root_span
) -> None:
    """백그라운드: 생성 → 큐로 토큰 전달 → 완료 시 DB finalize. 연결과 무관하게 끝까지 돈다.

    요청 세션은 응답 종료 시 닫히므로 자기 세션을 쓴다.
    """
    parts = []
    # 출처 꼬리는 knowledge만 만든다. 꼬리는 delta로 나가지 않고 parts에는 본문만 쌓인다.
    splitter = TailSplitter() if prepared.route == "knowledge" else None

    # 재접속용 미러링 — 큐와 같은 이벤트를 Redis Stream에도 남긴다. 요청이 아니라 태스크가 소유한다.
    writer = stream_resume.StreamWriter(lease.tenant_id, prepared.assistant_message_id)
    # meta는 로컬 리더가 직접 합성하므로 큐를 거치지 않고 스트림에만 심는다.
    writer.add(EVENT_META, _meta_payload(prepared))

    async def emit(event: str, data: dict) -> None:
        """생성 경로 이벤트 방출의 단일 지점 — 큐와 재접속 스트림에 함께 써서 한쪽 누락을 막는다."""
        await queue.put((event, data))
        writer.add(event, data)

    try:
        async with AsyncSessionLocal() as session:
            svc = RagService(tenant_id=lease.tenant_id, session=session)
            with otel.span('generate', 'LLM') as sp:  # 컨텍스트 복사로 kms.query의 자식
                first_token_at = None  # 체감 TTFT — 첫 청크 도착 시각
                async for chunk in svc.generate(prepared):
                    if first_token_at is None:
                        first_token_at = time.monotonic()
                        # 요청 도착부터 — prepare 포함 체감값이라 서버측 TTFT와 다르다.
                        metrics.TTFT_SECONDS.labels(route=prepared.route).observe(first_token_at - t_request)
                    text = splitter.feed(chunk) if splitter else chunk
                    if text:
                        parts.append(text)
                        await emit(EVENT_DELTA, {"text": text})
                if splitter and (tail_flush := splitter.finish()):
                    parts.append(tail_flush)  # 마커 접두인 줄 알았던 본문 끝자락
                    await emit(EVENT_DELTA, {"text": tail_flush})
                answer = "".join(parts)
                # LLM 종료 사유(stop|length|other) — 집계용 카운터와 드릴다운용 스팬 속성을 같이 남긴다.
                if svc.last_finish_reason is not None:
                    metrics.FINISH_REASON_TOTAL.labels(route=prepared.route, reason=svc.last_finish_reason).inc()
                if sp.is_recording():  # 가드가 먼저 — no-op이면 clip 비용도 없게
                    otel.set_attrs(sp, {otel.LLM_MODEL: settings.llm_model, otel.OUTPUT_VALUE: otel.clip(answer)})
                    if svc.last_finish_reason is not None:
                        otel.set_attrs(sp, {'kms.llm_finish_reason': svc.last_finish_reason})
                    if svc.last_usage is not None:  # 상용 공급자만 — 캐시 적중 토큰이 0이면 캐싱 전제가 깨진 것
                        otel.set_attrs(
                            sp,
                            {
                                'kms.llm_prompt_tokens': svc.last_usage.prompt_tokens,
                                'kms.llm_completion_tokens': svc.last_usage.completion_tokens,
                                'kms.llm_cached_tokens': svc.last_usage.cached_tokens,
                                'kms.llm_cache_write_tokens': svc.last_usage.cache_write_tokens,
                            },
                        )
                    if splitter and splitter.tail_raw is not None:
                        # 꼬리 원문 — 인용 오류가 모델 번호 탓인지 매핑 버그인지 가르는 유일한 증거.
                        otel.set_attrs(sp, {'kms.citation_tail': otel.clip(splitter.tail_raw)})
                    if splitter and splitter.truncated:
                        # 꼬리가 max_tokens 등으로 잘린 "성공처럼 보이는 실패"
                        otel.set_attrs(sp, {'kms.tail_truncated': True})

            # citations가 곧 '근거 있음'의 정의다 — 답변 문구 판정으로 덮어쓰지 않는다.
            citations = [] if splitter is None else resolve_citations(splitter.tail_raw, *prepared.citation_candidates)
            latency_ms = _elapsed_ms(t_request)  # 저장·스팬·done이 같은 값을 쓰도록 한 번만 계산
            await svc.finalize(prepared, answer, citations, status=TurnStatus.DONE, latency_ms=latency_ms)
            await session.commit()
            result = TurnResult(
                answer=answer, citations=citations, finish_reason=TurnStatus.DONE, latency_ms=latency_ms
            )
            _record_turn_result(root_span, result)
            # done은 finalize·commit 뒤에 — 값이 전부 확정된 다음 내보낸다.
            await emit(EVENT_DONE, _done_payload(result))

            # 캐시 적재는 done 뒤에 따로 — 각주 표시를 늦추지 않고, 적재 실패가 done 턴을 오염시키지 않게.
            # 먼저 취소 대상에서 빠진다 — 아래 await 중 늦은 취소가 done 턴을 cancelled로 덮어쓰지 않게.
            cancellation.unregister(prepared.assistant_message_id)
            try:
                await svc.maybe_cache(prepared, answer, citations)
                await session.commit()
            except Exception:
                logger.exception('캐시 저장 실패 — 턴은 이미 done (conversation=%s)', prepared.conversation_id)
    except asyncio.CancelledError:
        # 명시적 취소. CancelledError는 except Exception에 안 잡힌다. 취소는 한 번만 전달되므로
        # 여기서 잡은 뒤의 await는 정상 완료된다.
        answer, latency_ms = "".join(parts), _elapsed_ms(t_request)
        await _finalize_out_of_band(lease, prepared, answer, TurnStatus.CANCELLED, latency_ms)
        result = TurnResult(answer=answer, citations=[], finish_reason=TurnStatus.CANCELLED, latency_ms=latency_ms)
        _record_turn_result(root_span, result)
        await emit(EVENT_DONE, _done_payload(result))
        raise  # 태스크가 '취소됨'으로 끝나도록 재전파
    except Exception:
        logger.exception("답변 생성 실패 (tenant=%s, conversation=%s)", lease.tenant_id, prepared.conversation_id)
        # 부분 텍스트와 latency를 취소와 같게 저장한다 — 화면에 흐른 답변이 새로고침 후에도 남게.
        answer, latency_ms = "".join(parts), _elapsed_ms(t_request)
        await _finalize_out_of_band(lease, prepared, answer, TurnStatus.FAILED, latency_ms)
        result = TurnResult(answer=answer, citations=[], finish_reason=TurnStatus.FAILED, latency_ms=latency_ms)
        _record_turn_result(root_span, result)
        # 예외 원문은 서버 로그로만 — 내부 경로·설정이 섞일 수 있어 클라이언트에 노출 금지
        await emit(
            EVENT_ERROR,
            {"code": "generation_failed", "message": "답변 생성 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요."},
        )
        await emit(EVENT_DONE, _done_payload(result))  # 계약: error 뒤에도 done으로 닫는다
    finally:
        # 취소 대상에서 먼저, 동기로 빠진다 — 아래 await 중 도착한 취소가 이 finally를 끊어
        # 리미터 반납·스팬 종료를 유실시키지 않게.
        cancellation.unregister(prepared.assistant_message_id)
        await queue.put(None)  # 리더 종료 sentinel — finalize·commit 뒤에 넣는다
        # 스트림 쪽 종료 마커 — 위 sentinel과 같은 사실이다. 한쪽만 남기면 재접속 리더가 종료를 모른다.
        await writer.aclose()
        await limiter.release(lease)  # 리미터는 태스크 수명 끝에 (실제 생성 점유 구간)
        root_span.end()  # 핸드오프된 턴 루트 스팬 — duration=턴 전체


async def queue_reader(prepared: PreparedRag, queue: asyncio.Queue) -> AsyncIterator[str]:
    """생성 경로 SSE: 큐 이벤트를 흘린다(읽기 전용). 연결이 끊겨도 태스크엔 영향 없다.

    큐는 unbounded여야 한다 — maxsize를 걸면 읽지 않는 큐가 찬 순간 태스크가 put에서 영구
    대기해 finally가 안 돌고, 태스크 세션의 DB 커넥션이 회수되지 않는다.
    """
    yield _meta_event(prepared)
    while True:
        try:
            item = await asyncio.wait_for(queue.get(), timeout=settings.sse_ping_interval_seconds)
        except asyncio.TimeoutError:
            # 프록시가 유휴 연결을 끊지 않게 ping. wait_for가 get을 취소해도 아이템은 큐에 남는다.
            yield sse_event(EVENT_PING, {})
            continue
        if item is None:
            break
        event, data = item
        yield sse_event(event, data)
