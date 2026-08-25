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

── FE 이벤트 계약 (#56에서 표준화) ─────────────────────────
봉투는 `event: <name>\\ndata: <json>\\n\\n`. 이름은 아래 EVENT_* 상수가 유일한 정의점.

  공통       meta → delta×N → done
  오류       … → error → done          (done은 "스트림 종료" 신호이지 성공 신호가 아니다)
  ping       유휴 시 수시 — 클라이언트는 무시 (프록시 유휴 종료 대비, 생성 경로만)

원칙: 서버가 계산한 것을 서버가 보낸다 — 클라이언트가 본문을 파싱해 상태를 유추하지 않는다.
- delta는 순수 텍스트 조각. done이 최종 상태를 싣는다:
  {finish_reason, latency_ms, citations} — finish_reason 어휘는 messages.status와 동일
  (done/cancelled/failed/blocked — 매핑 두 벌 금지), citations는 실제 인용된 출처 객체만.
- 구 계약의 sources 이벤트(후보 낙관 전송 + 거절/취소 시 [] 정정)는 삭제 — "후보"와 "확정"을
  한 이벤트가 겸하다 사고 났던 자리(FE 07-19). 확정 인용은 done 한 번에만 실린다.
- done은 생성 태스크가 값과 함께 큐에 넣고 queue_reader는 순수 전달자다 — 이전처럼 리더가
  빈 done을 합성하지 않는다. done 없이 스트림이 끊기면 비정상 종료다(FE는 이미 그렇게 처리).
"""
import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass


from config import settings
from database import AsyncSessionLocal
from rag import cancellation, limiter, otel, stream_resume
from rag.limiter import Lease
from rag.citation_tail import TailSplitter, resolve_citations
from rag.models import TurnStatus
from rag.service import PreparedRag, RagService
from schemas.kms import SourceCitation

logger = logging.getLogger(__name__)

# SSE 이벤트 이름 — FE 계약의 단일 정의점 (문자열 리터럴 산재 방지)
EVENT_META = 'meta'
EVENT_DELTA = 'delta'
EVENT_PING = 'ping'
EVENT_ERROR = 'error'
EVENT_DONE = 'done'

# SSE 응답 헤더 — /kms/query와 재접속(#75) 두 엔드포인트가 공유한다.
# 값이 갈리면 한쪽만 프록시 버퍼링에 걸려 '어떤 스트림은 안 흐른다'가 된다.
SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}

# 태스크 참조 유지(GC 방지)와 취소 대상 색인은 rag/cancellation.py의 레지스트리가 겸한다 —
# 같은 목적의 자료구조를 둘로 두지 않는다. 취소 규약(pop-then-cancel)은 그쪽 docstring 참조.


def sse_event(event: str, data) -> str:
    """SSE 한 이벤트를 봉투 형식 문자열로 만든다.
    event: 이름 / data: JSON payload / 빈 줄로 이벤트 종료.
    """
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _meta_payload(prepared: PreparedRag) -> dict:
    """meta 페이로드 — 로컬 리더(_meta_event)와 재접속 스트림 기록이 공유한다 (#75).
    두 경로가 각자 조립하면 재접속 화면의 meta만 조용히 달라진다."""
    return {
        "conversation_id": prepared.conversation_id,
        "assistant_message_id": prepared.assistant_message_id,   # FE 재접속·상태조회용
        "cached": prepared.is_cache_hit,
        "cache_kind": prepared.cache_kind,
        "reason": "no_evidence" if prepared.no_evidence else "ok",
    }


def _meta_event(prepared: PreparedRag) -> str:
    return sse_event(EVENT_META, _meta_payload(prepared))


@dataclass
class TurnResult:
    """턴 하나의 최종 결과 — done 페이로드·루트 스팬·DB 저장이 전부 이 한 객체를 소비한다 (#56).

    같은 사실(답변·인용·상태·소요)을 세 소비처가 각자 재계산하면 반드시 어긋난다 —
    4개 종료 분기(즉시/완료/취소/실패)가 여기서 한 번 조립하고, 이후는 읽기만 한다.
    citations는 실제 인용된 출처 객체만 담는다(후보 전체가 아니라).
    """
    answer: str
    citations: list[SourceCitation]
    finish_reason: TurnStatus   # 어휘 = turn_state.TERMINAL (전체 − generating). 와이어 값은 #56 그대로
    latency_ms: int | None          # failed는 None (DB 규칙과 동일)


def _done_payload(result: TurnResult) -> dict:
    return {
        "finish_reason": result.finish_reason,
        "latency_ms": result.latency_ms,
        "citations": [c.model_dump() for c in result.citations],
    }


def _elapsed_ms(t_request: float) -> int:
    """사용자 체감 응답시간 — prepare(인텐트·검색·condense)를 포함한 요청 시작 기준.
    t_request는 호출부가 time.monotonic()으로 찍은 값이어야 한다 (기준 시계 일치)."""
    return int((time.monotonic() - t_request) * 1000)


def _record_turn_result(root_span, result: TurnResult) -> None:
    """턴 결과를 루트 kms.query 스팬에 기록 — 값은 전부 DB 저장 규칙과 동일해야 한다 (#54).

    트레이스·DB·done 이벤트가 서로 다른 얘기를 하면 교차검증이 무너진다(#34 NFC가 그렇게
    잡혔다) — 셋 다 같은 TurnResult를 소비한다 (#56).
    is_recording 가드가 먼저다 — no-op이면 clip(문자열 절단) 비용까지 없어야 한다(otel.py 규약).
    """
    if not root_span.is_recording():
        return
    otel.set_attrs(root_span, {
        otel.OUTPUT_VALUE: otel.clip(result.answer),   # 답변은 절단 — 스팬 텍스트 정책(#7)
        'kms.status': result.finish_reason,
        'kms.latency_ms': result.latency_ms,           # failed는 None → 속성 자체가 안 실림 (DB NULL과 대응)
        'kms.cited_docs': [c.filename for c in result.citations],
    })


async def immediate_stream(service: RagService, prepared: PreparedRag, session,
                           t_request: float, root_span) -> AsyncIterator[str]:
    """즉시 경로(입력차단·캐시히트·근거없음): 답이 확정 → 인라인 저장·스트림. 태스크 불필요.

    persist-before-stream(#16): 답이 이미 확정된 경로이므로 저장·commit을 먼저 한다 —
    재생 중 disconnect여도 턴이 남고(이력 구멍 방지), get_semantic이 잡은 hit_count
    행 잠금도 스트림 전에 풀린다(동시 히트가 첫 클라이언트 수신 속도에 직렬화되던 문제).

    루트 스팬 핸드오프(#54): 이 제너레이터는 핸들러가 반환한 '뒤' Starlette가 돌리므로,
    핸들러 finally에서 스팬을 닫으면 답변·latency가 확정되기 전에 스팬이 죽는다. 그래서
    생성 경로(_run_generation)처럼 스팬 종료를 이어받는다 — finally라 disconnect
    (GeneratorExit)에도 닫힌다. 단 클라이언트가 첫 바이트 전에 끊겨 아예 iterate가 안 되면
    스팬(과 저장 — 기존 리스크)이 유실된다: 알려진 트레이드오프, 저장과 운명을 같이한다.
    """
    try:
        answer = ''.join([chunk async for chunk in service.generate(prepared)])
        latency_ms = _elapsed_ms(t_request)   # 저장·스팬·done이 같은 값을 쓰도록 한 번만 계산 (#54)
        # 즉시 경로의 인용 = prepared.sources 그대로 (#56): 캐시 히트는 저장 시점에 이미
        # 인용만 남긴 값이 복원된 것이고, 차단·근거없음은 sources가 애초에 빈 목록이다.
        # #61에서 답변 텍스트 판정(옛 is_refusal)을 걷어냈다 — 이 경로엔 애초에 불필요했다.
        # 세 경우 모두 sources가 라우팅·저장 단계에서 이미 확정돼 있어, 답변 문구를 다시
        # 들여다볼 근거가 없었다(차단·근거없음은 rag/service.py에서 [], 히트는 복원값).
        result = TurnResult(answer=answer, citations=list(prepared.sources),
                            finish_reason=prepared.terminal_status, latency_ms=latency_ms)
        # 저장 실패가 이미 확정된 답변의 '전달'까지 막으면 안 된다 (fail-open — 리뷰 반영).
        # StreamingResponse는 첫 yield 전에 200 헤더가 이미 나가므로, 여기서 예외가 새면
        # 사용자는 빈 스트림을 받는다. 실패는 로그만 남기고 전달은 계속한다.
        # 여기서 실패해도 질문 자체는 남는다 — 자리표시가 턴 시작에 이미 커밋됐다(#72).
        # 그 경우 자리표시는 generating으로 남고 스테일 스윕이 failed로 회수한다.
        try:
            await service.finalize(prepared, answer, result.citations,
                                   status=prepared.terminal_status, latency_ms=latency_ms)
            await session.commit()
        except Exception:
            logger.exception("즉시 경로 저장 실패 — 답변 전달은 계속 (conversation=%s)",
                             prepared.conversation_id)
        _record_turn_result(root_span, result)

        yield _meta_event(prepared)
        # 확정된 답변은 한 번에 보낸다 — 쪼개도 await 없이 연달아 나가 같은 청크로 도착하고,
        # FE가 자체 타자기(버퍼 길이 기준)로 그리므로 이벤트 개수는 렌더에 영향이 없다 (#26).
        yield sse_event(EVENT_DELTA, {"text": answer})
        yield sse_event(EVENT_DONE, _done_payload(result))
    finally:
        root_span.end()                       # 핸드오프된 턴 루트 스팬 (#54)


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
                                status: TurnStatus, latency_ms: int | None) -> None:
    """비정상 종료(취소·실패)의 상태 기록 — 새 세션으로, 실패는 삼킨다.

    본 흐름의 세션은 이미 롤백/닫힘 상태일 수 있어 자기 세션을 새로 연다.
    기록이 실패하면 그 턴은 generating으로 남고 스테일 스윕(rag.conversation.GENERATION_STALE_SECONDS, cron 1분 주기)이 failed로 정리한다 — 의도(취소 vs
    실패)는 잃지만 고착은 남지 않는다. 상태 기록 실패가 정리(finally)를 막아선 안 되므로 삼킨다.
    """
    try:
        async with AsyncSessionLocal() as session:
            await RagService(tenant_id=lease.tenant_id, session=session).finalize(
                prepared, answer, citations=[], status=status, latency_ms=latency_ms)
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
    # 출처 꼬리 분리(#56) — knowledge만 꼬리를 만든다(OTHER·즉시 경로는 인용 없음).
    # 꼬리는 delta로 나가지 않고, parts(=화면=저장)에는 splitter가 방출한 본문만 쌓인다.
    splitter = TailSplitter() if prepared.route == "knowledge" else None

    # 재접속용 미러링 (#75) — 인메모리 큐와 **같은 이벤트**를 Redis Stream에도 남긴다.
    # writer의 생성·정리 주체는 이 태스크다 (lease·root_span·취소 레지스트리와 같은 수명 원칙:
    # 요청이 아니라 태스크가 소유한다 — 연결이 끊겨도 생성은 완주하므로).
    writer = stream_resume.StreamWriter(lease.tenant_id, prepared.assistant_message_id)
    # meta는 로컬 리더가 prepared로 직접 합성하므로(큐를 안 거친다) emit 대상이 아니다 —
    # 재접속 리더에겐 prepared가 없으니 스트림에만 심는다. 비대칭이지만 근거가 있다.
    writer.add(EVENT_META, _meta_payload(prepared))

    async def emit(event: str, data: dict) -> None:
        """생성 경로 이벤트 방출의 **단일 지점** (#75).

        큐 put과 스트림 기록을 한 함수가 함께 한다 — 방출 자리가 6곳(delta 2·done 3·error 1)이라
        호출부마다 두 줄을 쓰면 언젠가 한쪽을 빠뜨린다. 그러면 재접속 화면만 조용히 달라진다.
        이 저장소가 반복해서 경계하는 '우연히 같은' 상태를 구조적으로 막는다.
        """
        await queue.put((event, data))
        writer.add(event, data)

    try:
        async with AsyncSessionLocal() as session:
            svc = RagService(tenant_id=lease.tenant_id, session=session)
            with otel.span('generate', 'LLM') as sp:   # 배경 태스크 — 컨텍스트 복사로 kms.query의 자식 (#7)
                async for chunk in svc.generate(prepared):
                    text = splitter.feed(chunk) if splitter else chunk
                    if text:
                        parts.append(text)
                        await emit(EVENT_DELTA, {"text": text})
                if splitter and (tail_flush := splitter.finish()):
                    parts.append(tail_flush)             # 마커 접두인 줄 알았던 본문 끝자락
                    await emit(EVENT_DELTA, {"text": tail_flush})
                answer = "".join(parts)
                if sp.is_recording():   # 가드가 먼저 — no-op이면 clip 비용도 없어야 한다 (otel.py 규약, #54 시정)
                    otel.set_attrs(sp, {otel.LLM_MODEL: settings.vllm_model, otel.OUTPUT_VALUE: otel.clip(answer)})
                    if splitter and splitter.tail_raw is not None:
                        # 꼬리 원문(번호 목록) — OUTPUT_VALUE는 걷어낸 본문이라 여기 없으면 어디에도
                        # 안 남는다. 고객이 "인용 문서가 잘못됐다"고 할 때 모델 판단(번호 자체가
                        # 틀림) vs 매핑 버그(번호는 맞는데 citations가 다름)를 가르는 유일한 증거.
                        otel.set_attrs(sp, {'kms.citation_tail': otel.clip(splitter.tail_raw)})
                    if splitter and splitter.truncated:
                        # "성공처럼 보이는 실패"를 관측 가능하게 — 꼬리가 max_tokens 등으로 잘림
                        otel.set_attrs(sp, {'kms.tail_truncated': True})

            # citations가 곧 '근거 있음'의 정의가 됐으므로(#61) 문구 판정으로 이 값을
            # 덮어쓰지 않는다. 옛 `refusal or` 절은 "본문은 거절인데 꼬리에 번호가 남는"
            # 모델 모순을 []로 덮는 안전장치였는데, 비대칭이었다 — 반대 방향(확신에 찬
            # 답변인데 꼬리가 빈 경우)은 원래 못 잡았다. 그 절을 지우면 후자가 처음으로
            # 드러난다. 범위 검증은 resolve_citations가 제약 유무와 무관하게 수행한다.
            citations = [] if splitter is None else resolve_citations(
                splitter.tail_raw, *prepared.citation_candidates)   # 제약과 같은 파생점 (#65)
            latency_ms = _elapsed_ms(t_request)   # 저장·스팬·done이 같은 값을 쓰도록 한 번만 계산 (#54)
            await svc.finalize(prepared, answer, citations, status=TurnStatus.DONE, latency_ms=latency_ms)
            await session.commit()
            result = TurnResult(answer=answer, citations=citations,
                                finish_reason=TurnStatus.DONE, latency_ms=latency_ms)
            _record_turn_result(root_span, result)
            # done은 finalize·commit '뒤'에 — 값이 전부 확정된 다음 최종 상태로 내보낸다 (#56)
            await emit(EVENT_DONE, _done_payload(result))

            # 캐시 적재는 done '뒤' (#56 재배치, 사용자 결정 8/18) — 저장이 크리티컬 패스에
            # 있으면 각주 표시가 그만큼 늦고(실측 0.5~1초), 저장 실패가 완결된 턴을 failed로
            # 오염시켰다. 실패는 로그만 — 턴은 이미 done. (경위는 maybe_cache docstring)
            # 취소 대상에서 먼저 빠진다(동기·멱등, finally의 unregister와 같은 근거) —
            # 아래 await 중 늦은 취소가 done 턴을 cancelled로 덮어쓰는 레이스 차단.
            cancellation.unregister(prepared.assistant_message_id)
            try:
                await svc.maybe_cache(prepared, answer, citations)
                await session.commit()
            except Exception:
                logger.exception('캐시 저장 실패 — 턴은 이미 done (conversation=%s)',
                                 prepared.conversation_id)
    except asyncio.CancelledError:
        # 명시적 취소(#30). CancelledError는 BaseException 계열이라 아래 except Exception이
        # 잡지 못한다 — 이 절이 없으면 finalize를 못 타고 generating으로 남아 스테일 스윕이
        # failed로 만든다(사용자는 취소했는데 화면엔 실패).
        # 취소 예외는 한 번만 전달되므로 여기서 잡은 뒤의 await(세션·commit)는 정상 완료된다 — 실측 확인.
        answer, latency_ms = "".join(parts), _elapsed_ms(t_request)
        await _finalize_out_of_band(lease, prepared, answer, "cancelled", latency_ms)
        # 저장 규칙(finalize: status≠done → refusal=False·citations=[])과 동일 값 — 부분 텍스트는 저장값 그대로.
        # 구 계약의 "인용 [] 정정 이벤트"는 불필요해졌다 — 인용은 done에서만 확정되므로 (#56)
        result = TurnResult(answer=answer, citations=[], finish_reason=TurnStatus.CANCELLED,
                            latency_ms=latency_ms)
        _record_turn_result(root_span, result)
        await emit(EVENT_DONE, _done_payload(result))
        raise                                    # 태스크가 '취소됨'으로 끝나도록 재전파
    except Exception:
        logger.exception("답변 생성 실패 (tenant=%s, conversation=%s)", lease.tenant_id, prepared.conversation_id)
        # 부분 텍스트를 **취소와 같게** 저장한다. 예전엔 ""로 지웠는데, 그러면 화면엔 흐르던
        # 답변이 남아 있는데 DB는 비어 있어 새로고침하면 본문이 사라진다(실기동 실측 — 상담원이
        # 방금 읽은 내용을 자기도 다시 확인 못 한다). 위 취소 분기와 바로 붙어 있으면서 한쪽만
        # 지우고 있었고, "멈췄든 오류가 났든 답을 못 받았다는 점은 같다"는 이 저장소의 판단
        # (rag/conversation.py UNANSWERED)과도 어긋났다.
        # latency_ms도 함께 남긴다 — '얼마 만에 실패했는지'는 장애 분석의 기본 값인데 None이었다.
        # 이력 정책은 건드리지 않는다: failed를 _ISOLATED에 넣으면 질문까지 사라져 #59가
        # 고친 버그가 재발한다.
        answer, latency_ms = "".join(parts), _elapsed_ms(t_request)
        await _finalize_out_of_band(lease, prepared, answer, "failed", latency_ms)
        result = TurnResult(answer=answer, citations=[], finish_reason=TurnStatus.FAILED,
                            latency_ms=latency_ms)
        _record_turn_result(root_span, result)
        # 예외 원문은 서버 로그로만 — str(exc)에 내부 경로·설정이 섞일 수 있어 클라이언트에 노출 금지
        await emit(EVENT_ERROR, {"code": "generation_failed",
                                 "message": "답변 생성 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요."})
        await emit(EVENT_DONE, _done_payload(result))   # 계약: error 뒤에도 done으로 닫는다
    finally:
        # 취소 대상에서 먼저 빠진다 (#30) — 반드시 아래 await들보다 앞, 그리고 동기 문장으로.
        # 답변이 done으로 커밋된 뒤 리미터를 반납하는 찰나에 취소가 도착하면 이 finally가
        # 중간에 끊겨 반납·스팬 종료가 유실된다(실측). 자기를 먼저 빼두면 그 요청은 대상을
        # 못 찾아 cancel()을 호출하지 않고, 상태(done)를 근거로 404를 받는다.
        cancellation.unregister(prepared.assistant_message_id)
        await queue.put(None)                    # 리더 종료 sentinel — finalize·commit '뒤'에 넣는다
        # 스트림 쪽 종료 마커 — 위 sentinel과 **같은 사실**의 다른 인코딩이다 (#75).
        # 두 줄이 나란히 있어야 하는 이유: Redis Stream은 nil 엔트리를 표현할 수 없어
        # sentinel을 그대로 미러링할 수 없다. 한쪽만 남기면 재접속 리더가 종료를 못 알아본다.
        await writer.aclose()
        await limiter.release(lease)              # 리미터는 태스크 수명 끝에 (실제 GPU 점유 구간)
        root_span.end()                           # 핸드오프된 턴 루트 스팬 — duration=턴 전체 (#7)


async def queue_reader(prepared: PreparedRag, queue: asyncio.Queue) -> AsyncIterator[str]:
    """생성 경로 SSE: 큐 이벤트를 흘린다 (읽기 전용). 연결이 끊겨도 태스크엔 영향 없음.

    그 무영향이 성립하는 근거가 **큐가 unbounded**라는 것이다. 소비가 멈춰도 생산자의
    `await queue.put(...)`이 블록되지 않아 태스크가 완주하고 finalize·리미터 반납이 보장된다.
    여기에 maxsize를 걸면 읽지 않는 큐가 차는 순간 태스크가 put에서 영구 대기하고 finally가
    실행되지 않는다 — generating 고착(스테일 스윕, cron 1분 주기)·리미터 슬롯(120초 prune)은 회수 장치가
    있지만 **태스크 세션의 DB 커넥션은 회수 장치가 없어 풀이 영구히 잠식된다.**
    쌓이는 양은 max_tokens 상한이 걸린 텍스트라 생성 1건당 수 KB에 불과하다.
    """
    yield _meta_event(prepared)
    while True:
        try:
            item = await asyncio.wait_for(queue.get(), timeout=settings.sse_ping_interval_seconds)
        except asyncio.TimeoutError:
            # 유휴 — 프록시가 idle 연결을 끊지 않게 ping (#56). 클라이언트는 무시하는 계약.
            # wait_for가 내부 get을 취소해도 아이템은 큐에 남아 다음 get이 집는다(유실 없음 —
            # asyncio.Queue의 취소 안전성, test_queue_reader가 회귀로 고정).
            yield sse_event(EVENT_PING, {})
            continue
        if item is None:
            break
        event, data = item
        yield sse_event(event, data)
