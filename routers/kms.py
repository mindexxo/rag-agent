"""KMS 쿼리 라우터.

  POST /kms/query — SSE 전용 (text/event-stream). X-Tenant-Id 헤더로 테넌트 식별.
  비스트리밍 JSON 경로(?stream=false)는 #26에서 삭제 — FE는 SSE만 쓰고, 두 경로가
  갈라져 자란 탓에 출력 차단 턴이 저장되지 않는 등 불일치가 쌓였다.

이 파일은 HTTP 경계만 담당한다 — 헤더 파싱, 동시 상한(429), 404 매핑, StreamingResponse
조립, 그리고 요청 스코프 자원(리미터 슬롯·OTel 루트 스팬)의 소유권 부기.
이벤트 어휘와 두 경로의 수명 관리는 rag/streaming.py (FE 이벤트 계약도 그 모듈 docstring에).
"""
import asyncio
import time

from fastapi import APIRouter, Depends, Header, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.responses import StreamingResponse

from config import settings
from database import get_session
from rag import limiter, otel, streaming
from rag.limiter import Lease
from rag.llm_schemas import LlmJudgmentFailed
from rag.models import TenantQuota
from rag.service import RagService
from schemas.kms import KmsQueryRequest

router = APIRouter(prefix='/kms')

_SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


def get_tenant_id(x_tenant_id: str = Header(...)) -> str:
    """X-TENANT-ID 헤더 필수 없을 시 422 응답"""
    return x_tenant_id


def get_user_id(x_user_id: str | None = Header(None)) -> str | None:
    """X-User-Id (선택). 있으면 사용자별 동시 제한도 적용, 없으면 테넌트 제한만.
    (기본값·상한은 config/tenant_quotas. GPU/모델 확정 시 env로 튜닝.)"""
    return x_user_id


async def concurrency_guard(
        tenant_id: str = Depends(get_tenant_id),
        user_id: str | None = Depends(get_user_id),
        session: AsyncSession = Depends(get_session),
):
    """테넌트별 동시 in-flight 상한 (F6). 초과 시 429. 획득한 Lease를 핸들러에 넘긴다.

    생성 경로는 반납 책임을 백그라운드 태스크로 넘기므로(lease.handed_off) 그 경우 여기
    finally에서 반납하지 않는다 — 연결이 끊겨도 태스크가 도는 동안 슬롯을 잡고 있어야 한다
    (REVIEW P1-11). 즉시 경로는 여기서 반납.
    """
    quota = (await session.execute(
        select(TenantQuota).where(TenantQuota.tenant_id == tenant_id)
    )).scalars().first()
    tenant_limit = quota.concurrency_limit if quota else settings.concurrency_limit_default
    user_limit = quota.user_concurrency if quota else settings.user_concurrency_default

    lease = await limiter.try_acquire(tenant_id, tenant_limit, user_id, user_limit)
    if lease is None:
        raise HTTPException(status_code=429, detail='동시 요청이 많아 처리할 수 없습니다. 잠시 후 다시 시도해 주세요.')
    try:
        yield lease
    finally:
        if not lease.handed_off:
            await limiter.release(lease)


@router.post('/query')
async def query(
        request: KmsQueryRequest, # FastAPI가 요청 바디 JSON을 자동으로 파싱해서 KmsQueryRequest 객체로 만들어줍니다. 타입 힌트만 써주면 됨
        tenant_id: str = Depends(get_tenant_id), # FastAPI가 get_tenant_id() 먼저 호출 → 결과를 tenant_id에 넣어줌
        user_id: str | None = Depends(get_user_id),
        session: AsyncSession = Depends(get_session),
        lease: Lease = Depends(concurrency_guard),
):
    t_request = time.monotonic()   # 응답시간 기준점 — prepare(인텐트·검색·condense) 포함 (사용자 체감)
    service = RagService(tenant_id=tenant_id, session=session, user_id=user_id)

    # 턴 루트 스팬(#7) — 데코레이터가 아닌 수동 관리: 두 스트림(즉시·생성) 모두 핸들러가
    # 반환한 뒤에야 답이 확정되므로, span.end()를 스트림 쪽에 핸드오프해 루트 duration이
    # 턴 전체를 반영하게 한다(#54). 핸들러 finally는 스트림이 시작되지 못한 경우
    # (prepare 단계 예외)만 직접 종료한다.
    root_span, otel_token = otel.start_turn()
    root_handed_off = False
    try:
        try:
            prepared = await service.prepare(request.query, request.conversation_id, request.attachments,
                                             domain_hint=request.domain_hint)
        except LlmJudgmentFailed as exc:
            # LLM이 분류·재작성 판단을 못 냈다 (#72) — 폴백으로 추측하지 않고 실패로 끝낸다.
            # 500(서버 버그)이 아니라 503: 재시도하면 될 수 있는 일시적 실패다.
            # 이 시점엔 사용자의 질문이 이미 저장돼 있고(턴 시작 자리표시), prepare()의
            # 안전망이 그 자리표시를 failed로 닫아둔 상태다.
            #
            # **스팬은 ERROR로 남긴다.** 404(존재하지 않는 대화)와 달리 이건 클라이언트 잘못이
            # 아니라 시스템 건강 신호다 — 폴백을 걷어내면서 새로 생긴 실패 계열이라, 여기서
            # 마킹하지 않으면 에러율 대시보드에 아예 안 잡힌다. 아래 `except HTTPException`이
            # 마킹 없이 통과시키므로 이 자리에서 직접 남겨야 한다.
            otel.mark_error(root_span, exc)
            raise HTTPException(status_code=503, detail='일시적으로 처리할 수 없습니다. 잠시 후 다시 시도해 주세요.')
        except ValueError:
            # 존재하지 않거나 다른 테넌트 소유의 conversation_id — 500 아닌 404 (REVIEW findings ③)
            # 정상 클라이언트 오류 — 스팬을 ERROR로 남기지 않는다 (에러율 오염 방지, 리뷰 반영)
            raise HTTPException(status_code=404, detail='대화를 찾을 수 없습니다.')
        otel.set_attrs(root_span, {          # 루트 스팬 속성 — prepare 결과 반영 (#7)
            otel.INPUT_VALUE: request.query,
            'kms.tenant_id': tenant_id,
            'kms.conversation_id': prepared.conversation_id,
            'kms.route': prepared.route,
            'kms.cache_kind': prepared.cache_kind,
            'kms.no_evidence': prepared.no_evidence,   # 판정 정의는 PreparedRag 한 곳
        })

        # ── 즉시 경로 (비LLM: 입력차단/캐시히트/근거없음) ──
        if not prepared.needs_generation:
            response = StreamingResponse(
                streaming.immediate_stream(service, prepared, session, t_request, root_span),
                media_type="text/event-stream", headers=_SSE_HEADERS,
            )
            # 스팬 종료는 immediate_stream의 finally가 담당(#54) — 플래그는 응답 객체 생성 '뒤'에
            # 세워, 여기서 실패하면 핸들러 finally가 종료한다(생성 경로의 spawn 순서와 같은 원칙).
            # lease는 손대지 않는다 — concurrency_guard(yield 의존성)가 스트림 종료까지 물고
            # 있다가 반납하므로, handed_off를 세우면 아무도 반납하지 않는다(슬롯 영구 유출).
            root_handed_off = True
            return response

        # ── 생성 경로 (LLM → 백그라운드 태스크 + 큐 리더) ──
        # 자리표시는 prepare()가 턴 시작에 이미 커밋했다(#72) — prepared.assistant_message_id가
        # 채워져 있으므로 여기서 따로 열 것이 없다. 태스크가 다른 세션에서 그 id로 UPDATE한다.
        queue: asyncio.Queue = asyncio.Queue()      # unbounded 필수 — 근거는 streaming.queue_reader docstring
        streaming.spawn_generation(prepared, queue, lease, t_request, root_span)
        # 소유권 이전 표시는 spawn '뒤'에 — spawn이 실패하면 플래그가 서지 않아 요청 쪽이
        # 정리한다(슬롯 유출·스팬 미종료 방지). create_task와 이 두 줄 사이엔 await가 없어
        # 태스크가 먼저 돌아 이중 반납할 여지도 없다.
        lease.handed_off = True
        root_handed_off = True
        return StreamingResponse(
            streaming.queue_reader(prepared, queue),
            media_type="text/event-stream", headers=_SSE_HEADERS,
        )
    except HTTPException:
        raise                                       # 정상 클라이언트 오류(404 등) — ERROR 마킹 없이
    except Exception as exc:
        otel.mark_error(root_span, exc)             # 예상외 오류만 스팬 ERROR (에러율 신뢰 유지)
        raise
    finally:
        otel.detach_turn(otel_token)
        if not root_handed_off:
            root_span.end()
