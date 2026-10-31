"""FAQ 카테고리 API — 목록·생성·이름 변경·삭제. 빈 이름 422와 응답 조립만 하고 업무 로직은 rag/faq.py가 한다."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_session
from rag import faq as faq_svc
from routers.deps import get_tenant_id
from routers.errors import faq_http_errors
from schemas.kms import FaqCategoryCreateRequest, FaqCategoryInfo, FaqCategoryUpdateRequest

router = APIRouter(prefix='/kms')


@router.get('/faq-categories', response_model=list[FaqCategoryInfo])
async def list_categories(
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_session),
) -> list[FaqCategoryInfo]:
    """이름순. faq_count는 소속 FAQ 수(활성·비활성 모두) — 화면이 삭제 가능 여부를 미리 안다."""
    rows = await faq_svc.list_categories(session, tenant_id)
    return [FaqCategoryInfo(id=c.id, name=c.name, faq_count=n) for c, n in rows]


@router.post('/faq-categories', response_model=FaqCategoryInfo)
async def create_category(
    request: FaqCategoryCreateRequest,
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_session),
) -> FaqCategoryInfo:
    if not request.name:  # strip·길이 상한은 스키마가 했다(FAQ_CATEGORY_NAME_MAX)
        raise HTTPException(status_code=422, detail='카테고리 이름이 비어 있습니다.')
    with faq_http_errors():
        category = await faq_svc.create_category(session, tenant_id, request.name)
    return FaqCategoryInfo(id=category.id, name=category.name, faq_count=0)


@router.patch('/faq-categories/{category_id}', response_model=FaqCategoryInfo)
async def update_category(
    category_id: int,
    request: FaqCategoryUpdateRequest,
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_session),
) -> FaqCategoryInfo:
    with faq_http_errors():
        category = await faq_svc.get_category(session, tenant_id, category_id)
    if not request.name:
        raise HTTPException(status_code=422, detail='카테고리 이름이 비어 있습니다.')
    name = request.name
    with faq_http_errors():
        count = await faq_svc.rename_category(session, category, name)
    return FaqCategoryInfo(id=category_id, name=name, faq_count=count)


@router.delete('/faq-categories/{category_id}', status_code=204)
async def delete_category(
    category_id: int,
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_session),
) -> None:
    """소속 FAQ가 있으면 409 — 폴더 삭제와 같은 규칙(FE 요청 그대로). FAQ를 옮기거나 지운 뒤 삭제한다."""
    with faq_http_errors():
        category = await faq_svc.get_category(session, tenant_id, category_id)
        await faq_svc.delete_category(session, category)
