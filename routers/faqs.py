"""FAQ API — 목록·생성·수정·삭제. 빈 값 422와 응답 조립만 하고 업무 로직은 rag/faq.py가 한다."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_session
from rag import faq as faq_svc
from rag.models import Faq
from routers.deps import get_tenant_id, get_user_id
from routers.errors import faq_http_errors
from schemas.kms import (
    DOC_LIST_DEFAULT_LIMIT,
    DOC_LIST_MAX_LIMIT,
    FaqCreateRequest,
    FaqListResponse,
    FaqResponse,
    FaqUpdateRequest,
)

router = APIRouter(prefix='/kms')


def _to_response(f: Faq, ref_count: int | None = None) -> FaqResponse:
    return FaqResponse(
        id=f.id,
        question=f.question,
        variants=f.variants or [],
        answer=f.answer,
        is_active=f.is_active,
        category_id=f.category_id,
        created_at=f.created_at,
        updated_at=f.updated_at,
        created_by=f.created_by,
        updated_by=f.updated_by,
        ref_count=ref_count,
    )


@router.get('/faqs', response_model=FaqListResponse)
async def list_faqs(
    limit: int = DOC_LIST_DEFAULT_LIMIT,
    offset: int = 0,
    q: str | None = None,  # 질문·유사 질문·답변 부분 일치
    category_id: int | None = None,  # 0 = 미분류만, 그 외 = 그 카테고리
    is_active: bool | None = None,
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_session),
) -> FaqListResponse:
    """FAQ 목록 — 페이징·검색·필터 (#278, FE 요청 6). 최신 등록이 위로 온다(문서 목록과 같은 정렬 기준).

    ⚠ #278에서 응답이 배열에서 `{items, total, has_more}`로 바뀌었다(FE 요청대로, 호출자는 FE 하나).
    파라미터 위생(limit 범위 밖은 상한으로, 음수 offset은 0, 빈 q는 미전송)은 문서 목록과 같다.
    """
    if not 1 <= limit <= DOC_LIST_MAX_LIMIT:
        limit = DOC_LIST_MAX_LIMIT
    if offset < 0:
        offset = 0
    q = (q or '').strip() or None  # 빈 문자열을 필터로 쓰면 '%%'가 되어 전건 매칭이다

    page = await faq_svc.list_faqs(session, tenant_id, faq_svc.FaqFilter(q, category_id, is_active), limit, offset)
    return FaqListResponse(
        items=[_to_response(f, page.ref_counts.get(f.id, 0)) for f in page.faqs],
        total=page.total,
        has_more=offset + len(page.faqs) < page.total,  # len — limit을 쓰면 마지막 부분 페이지에서 어긋난다
    )


@router.post('/faqs', response_model=FaqResponse)
async def create_faq(
    request: FaqCreateRequest,
    tenant_id: str = Depends(get_tenant_id),
    user_id: str | None = Depends(get_user_id),
    session: AsyncSession = Depends(get_session),
) -> FaqResponse:
    # 공백 제거·빈 유사 질문 제거·길이 상한은 스키마가 했다(schemas/kms.py FAQ_*_MAX) — 여기선 빈 값만 본다.
    if not request.question or not request.answer:
        raise HTTPException(status_code=422, detail='질문/답변이 비어 있습니다.')
    with faq_http_errors():
        faq = await faq_svc.create_faq(session, tenant_id, user_id, request)
    return _to_response(faq, 0)


@router.patch('/faqs/{faq_id}', response_model=FaqResponse)
async def update_faq(
    faq_id: int,
    request: FaqUpdateRequest,
    tenant_id: str = Depends(get_tenant_id),
    user_id: str | None = Depends(get_user_id),
    session: AsyncSession = Depends(get_session),
) -> FaqResponse:
    with faq_http_errors():
        faq = await faq_svc.get_faq(session, tenant_id, faq_id)

    # 빈 문자열 가드 (create는 스키마에서 막지만 PATCH는 optional이라 여기서 본다).
    # 빈 question/answer가 저장·재임베딩되면 검색에 쓰레기 항목이 남는다. strip·길이 상한은 스키마가 했다.
    if request.question is not None and not request.question:
        raise HTTPException(status_code=422, detail='질문은 비울 수 없습니다.')
    if request.answer is not None and not request.answer:
        raise HTTPException(status_code=422, detail='답변은 비울 수 없습니다.')
    with faq_http_errors():
        faq = await faq_svc.update_faq(session, faq, user_id, request)
    counts = await faq_svc.ref_counts(session, tenant_id, [faq.id])
    return _to_response(faq, counts.get(faq.id, 0))


@router.delete('/faqs/{faq_id}', status_code=204)
async def delete_faq(
    faq_id: int,
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_session),
) -> None:
    """항목 삭제 — 청크는 FK cascade로 함께 삭제, 관련 캐시 무효화."""
    with faq_http_errors():
        faq = await faq_svc.get_faq(session, tenant_id, faq_id)
    await faq_svc.delete_faq(session, faq)
