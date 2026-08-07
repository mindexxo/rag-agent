"""KMS 쿼리 라우터.

  POST /kms/query?stream=false
  X-Tenant-Id 헤더로 테넌트 식별.
  D 단계에서는 비스트리밍만 구현.
"""
import asyncio
import json
import logging
import time

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.responses import StreamingResponse

from config import settings
from database import get_session, AsyncSessionLocal
from rag.limiter import query_limiter
from rag.models import TenantQuota
from rag.prompts import BLOCKED_OUTPUT_ANSWER, NO_EVIDENCE_ANSWER, is_refusal
from rag.service import RagService
from schemas.kms import KmsQueryRequest, KmsQueryResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix='/kms')

def get_tenant_id(x_tenant_id: str = Header(...)) -> str:
    """X-TENANT-ID 헤더 필수 없을 시 422 응답"""
    return x_tenant_id

def get_user_id(x_user_id: str | None = Header(None)) -> str | None:
    """X-User-Id (선택). 있으면 사용자별 동시 제한도 적용, 없으면 테넌트 제한만.
    (기본값·상한은 config/tenant_quotas. GPU/모델 확정 시 env로 튜닝.)"""
    return x_user_id

async def concurrency_guard(
        request: Request,
        tenant_id: str = Depends(get_tenant_id),
        user_id: str | None = Depends(get_user_id),
        session: AsyncSession = Depends(get_session),
):
    """테넌트별 동시 in-flight 상한 (F6). 초과 시 429.

    생성 경로는 limiter 소유권을 백그라운드 태스크로 넘기므로(request.state.limiter_handed_off),
    그 경우 여기 finally에서 release하지 않는다 — 연결이 끊겨도 태스크가 도는 동안 slot을 잡고
    있어야 함(REVIEW P1-11). 즉시/비스트리밍 경로는 여기서 release.
    """
    quota = (await session.execute(
        select(TenantQuota).where(TenantQuota.tenant_id == tenant_id)
    )).scalars().first()
    tenant_limit = quota.concurrency_limit if quota else settings.concurrency_limit_default
    user_limit = quota.user_concurrency if quota else settings.user_concurrency_default

    token = await query_limiter.try_acquire(tenant_id, tenant_limit, user_id, user_limit)
    if token is None:
        raise HTTPException(status_code=429, detail='동시 요청이 많아 처리할 수 없습니다. 잠시 후 다시 시도해 주세요.')
    request.state.limiter_token = token       # 생성경로가 태스크로 release를 넘길 때 사용
    request.state.limiter_user_id = user_id
    try:
        yield
    finally:
        if not getattr(request.state, "limiter_handed_off", False):
            await query_limiter.release(tenant_id, token, user_id)

def sse_event(event: str, data) -> str:
    """SSE 한 이벤트를 봉투 형식 문자열로 만든다.
    event: 이름 / data: JSON payload / 빈 줄로 이벤트 종료.
    """
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

_SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}

# 백그라운드 생성 태스크 참조 유지 (asyncio가 미참조 태스크를 GC하지 않게)
_running_tasks: set[asyncio.Task] = set()


def _sse_meta(prepared, no_evidence: bool) -> str:
    return sse_event("meta", {
        "conversation_id": prepared.conversation_id,
        "assistant_message_id": prepared.assistant_message_id,   # FE 재접속·상태조회용
        "cached": prepared.is_cache_hit,
        "cache_kind": prepared.cache_kind,
        "reason": "no_evidence" if no_evidence else "ok",
    })

@router.post('/query', dependencies=[Depends(concurrency_guard)])
async def query(
        request: KmsQueryRequest, # FastAPI가 요청 바디 JSON을 자동으로 파싱해서 KmsQueryRequest 객체로 만들어줍니다. 타입 힌트만 써주면 됨
        http_request: Request,
        stream: bool = True,
        tenant_id: str = Depends(get_tenant_id), # FastAPI가 get_tenant_id() 먼저 호출 → 결과를 tenant_id에 넣어줌
        user_id: str | None = Depends(get_user_id),
        session: AsyncSession = Depends(get_session),
):
    t_request = time.monotonic()   # 응답시간 기준점 — prepare(인텐트·검색·condense) 포함 (사용자 체감)
    service = RagService(tenant_id=tenant_id, session=session, user_id=user_id)

    try:
        prepared = await service.prepare(request.query, request.conversation_id, request.attachments,
                                         domain_hint=request.domain_hint)
    except ValueError:
        # 존재하지 않거나 다른 테넌트 소유의 conversation_id — 500 아닌 404 (REVIEW findings ③)
        raise HTTPException(status_code=404, detail='대화를 찾을 수 없습니다.')
    no_evidence = prepared.no_evidence   # 판정 정의는 PreparedRag 한 곳 (리팩터 — 5곳 중복 제거)

    # ── 비스트리밍 경로 (기존 JSON) ──
    if not stream:
        parts = []
        async for token in service.generate(prepared):
            parts.append(token)
        answer = ''.join(parts)

        # 가드레일
        verdict = await service.guard_output(prepared, answer)
        if not verdict.safe:
            return KmsQueryResponse(
                answer=BLOCKED_OUTPUT_ANSWER,
                sources=[],
                conversation_id=prepared.conversation_id,
                reason='blocked_output',
                cached=False,
                cache_kind=None,
            )
        await service.save(prepared, answer, latency_ms=int((time.monotonic() - t_request) * 1000))
        await session.commit()

        return KmsQueryResponse(
            answer=answer,
            # 검색 근거 없음 + LLM 스스로 거절(규칙 3)한 경우 모두 인용 제외 — 거절 답변에 인용이 붙는 모순 방지
            sources=[] if (no_evidence or is_refusal(answer)) else prepared.sources,
            conversation_id=prepared.conversation_id,
            reason='no_evidence' if no_evidence else 'ok',
            cached=prepared.is_cache_hit,
            cache_kind=prepared.cache_kind,
        )

    # ── 스트리밍: 즉시 경로 (비LLM: cache-hit/blocked/no_evidence) ──
    if not prepared.needs_generation:
        return StreamingResponse(
            _immediate_stream(service, prepared, no_evidence, session, t_request),
            media_type="text/event-stream", headers=_SSE_HEADERS,
        )

    # ── 스트리밍: 생성 경로 (LLM → 백그라운드 태스크 + 큐 리더) ──
    await service.begin_turn(prepared)          # placeholder(generating) commit → assistant_message_id
    queue: asyncio.Queue = asyncio.Queue()
    limiter_token = getattr(http_request.state, "limiter_token", None)
    limiter_user_id = getattr(http_request.state, "limiter_user_id", None)
    task = asyncio.create_task(_run_generation(tenant_id, prepared, queue, limiter_token, limiter_user_id, t_request))
    _running_tasks.add(task)
    task.add_done_callback(_running_tasks.discard)
    http_request.state.limiter_handed_off = True   # limiter release는 태스크가 담당
    return StreamingResponse(
        _queue_reader(prepared, no_evidence, queue),
        media_type="text/event-stream", headers=_SSE_HEADERS,
    )


async def _run_generation(tenant_id: str, prepared, queue: asyncio.Queue,
                          limiter_token: str | None, limiter_user_id: str | None,
                          t_request: float) -> None:
    """백그라운드: 생성→큐로 토큰 전달→완료 시 DB finalize. 연결과 무관하게 끝까지.
    자기 세션의 RagService 사용 (요청 세션은 응답 종료 시 닫히므로).
    limiter는 이 태스크 수명 끝(finally)에서 토큰으로 release.
    """
    parts = []
    try:
        async with AsyncSessionLocal() as session:
            svc = RagService(tenant_id=tenant_id, session=session)
            async for chunk in svc.generate(prepared):
                parts.append(chunk)
                await queue.put(("token", {"text": chunk}))
            answer = "".join(parts)

            verdict = await svc.guard_output(prepared, answer)   # 현재 off → safe (no-op)
            if not verdict.safe:
                await svc.finalize(prepared, BLOCKED_OUTPUT_ANSWER, status="blocked", latency_ms=int((time.monotonic() - t_request) * 1000))
                await session.commit()
                await queue.put(("blocked_output", {"message": BLOCKED_OUTPUT_ANSWER}))
            else:
                # LLM이 스스로 거절(규칙 3)한 답변이면 이미 내보낸 인용을 정정
                if is_refusal(answer):
                    await queue.put(("sources", []))
                await svc.finalize(prepared, answer, status="done", latency_ms=int((time.monotonic() - t_request) * 1000))
                await session.commit()
    except Exception:
        logger.exception("답변 생성 실패 (tenant=%s, conversation=%s)", tenant_id, prepared.conversation_id)
        try:
            async with AsyncSessionLocal() as s2:
                await RagService(tenant_id=tenant_id, session=s2).finalize(prepared, "", status="failed")
                await s2.commit()
        except Exception:
            logger.exception("failed 상태 기록도 실패 (tenant=%s)", tenant_id)
        # 예외 원문은 서버 로그로만 — str(exc)에 내부 경로·설정이 섞일 수 있어 클라이언트에 노출 금지
        await queue.put(("error", {"code": "generation_failed",
                                   "message": "답변 생성 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요."}))
    finally:
        await queue.put(None)                        # 리더 종료 sentinel
        await query_limiter.release(tenant_id, limiter_token, limiter_user_id)   # limiter는 태스크 수명 끝에


async def _queue_reader(prepared, no_evidence: bool, queue: asyncio.Queue):
    """큐 이벤트를 SSE로 흘림 (읽기 전용). 연결 끊겨도 태스크엔 영향 없음."""
    yield _sse_meta(prepared, no_evidence)
    yield sse_event("sources", [] if no_evidence else [s.model_dump() for s in prepared.sources])
    while True:
        item = await queue.get()
        if item is None:
            break
        event_type, data = item
        yield sse_event(event_type, data)
    yield sse_event("done", {})


async def _immediate_stream(service, prepared, no_evidence: bool, session, t_request: float):
    """즉시 경로(cache-hit/blocked/no_evidence): 답이 확정 → 인라인 저장·스트림.
    placeholder·태스크 불필요."""
    yield _sse_meta(prepared, no_evidence)
    yield sse_event("sources", [] if no_evidence else [s.model_dump() for s in prepared.sources])
    parts = []
    async for chunk in service.generate(prepared):
        parts.append(chunk)
        yield sse_event("token", {"text": chunk})
    answer = "".join(parts)
    if is_refusal(answer):
        yield sse_event("sources", [])
    await service.save(prepared, answer, latency_ms=int((time.monotonic() - t_request) * 1000))
    await session.commit()
    yield sse_event("done", {})
