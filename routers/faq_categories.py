"""FAQ 카테고리 라우터 (#278, FE 요청 4) — 폴더(routers/folders.py)와 같은 1단 관리 분류.

폴더와 다른 점 하나: **검색에 영향이 없다.** 참조 토글·설명이 없으니 캐시 무효화도 색인 fan-out도 없다 —
이름을 바꿔도 FAQ 청크는 그대로다. 규칙은 폴더 그대로: 빈 이름 422, 같은 이름 409(UNIQUE (tenant_id, name)),
소속 FAQ가 있으면 삭제 409. FAQ의 카테고리 소속 변경은 FAQ 라우터의 PATCH가 담당한다(문서↔폴더와 같은 분담).
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_session
from rag.models import Faq, FaqCategory
from routers.kms import get_tenant_id
from schemas.kms import FaqCategoryCreateRequest, FaqCategoryInfo, FaqCategoryUpdateRequest

router = APIRouter(prefix='/kms')


async def _get_category(session: AsyncSession, tenant_id: str, category_id: int) -> FaqCategory:
    category = (await session.execute(
        select(FaqCategory)
        .where(FaqCategory.tenant_id == tenant_id)   # 격리 — WHERE 절 명시
        .where(FaqCategory.id == category_id)
    )).scalars().first()
    if category is None:
        raise HTTPException(status_code=404, detail='category not found')
    return category


async def _faq_count(session: AsyncSession, tenant_id: str, category_id: int) -> int:
    return (await session.execute(
        select(func.count()).select_from(Faq)
        .where(Faq.tenant_id == tenant_id)
        .where(Faq.category_id == category_id)
    )).scalar_one()


@router.get('/faq-categories', response_model=list[FaqCategoryInfo])
async def list_categories(
        tenant_id: str = Depends(get_tenant_id),
        session: AsyncSession = Depends(get_session),
):
    """이름순. faq_count는 소속 FAQ 수(활성·비활성 모두) — 화면이 삭제 가능 여부를 미리 안다."""
    rows = (await session.execute(
        select(FaqCategory, func.count(Faq.id))
        .outerjoin(Faq, (Faq.category_id == FaqCategory.id) & (Faq.tenant_id == FaqCategory.tenant_id))
        .where(FaqCategory.tenant_id == tenant_id)
        .group_by(FaqCategory.id)
        .order_by(FaqCategory.name)
    )).all()
    return [FaqCategoryInfo(id=c.id, name=c.name, faq_count=n) for c, n in rows]


@router.post('/faq-categories', response_model=FaqCategoryInfo)
async def create_category(
        request: FaqCategoryCreateRequest,
        tenant_id: str = Depends(get_tenant_id),
        session: AsyncSession = Depends(get_session),
):
    if not request.name:   # strip·길이 상한은 스키마가 했다(FAQ_CATEGORY_NAME_MAX)
        raise HTTPException(status_code=422, detail='카테고리 이름이 비어 있습니다.')
    category = FaqCategory(tenant_id=tenant_id, name=request.name)
    session.add(category)
    try:
        await session.commit()
    except IntegrityError:
        await session.rollback()
        raise HTTPException(status_code=409, detail=f"'{request.name}' 카테고리가 이미 있습니다.")
    return FaqCategoryInfo(id=category.id, name=category.name, faq_count=0)


@router.patch('/faq-categories/{category_id}', response_model=FaqCategoryInfo)
async def update_category(
        category_id: int,
        request: FaqCategoryUpdateRequest,
        tenant_id: str = Depends(get_tenant_id),
        session: AsyncSession = Depends(get_session),
):
    category = await _get_category(session, tenant_id, category_id)
    if not request.name:
        raise HTTPException(status_code=422, detail='카테고리 이름이 비어 있습니다.')
    name = request.name
    count = await _faq_count(session, tenant_id, category.id)   # 대입 전에 센다 — 대입 뒤 SELECT는 autoflush로 UNIQUE에서 500이 난다(#276 교훈)
    category.name = name
    try:
        await session.commit()
    except IntegrityError:
        await session.rollback()   # rollback이 category를 만료시키므로 지역 변수만 쓴다
        raise HTTPException(status_code=409, detail=f"'{name}' 카테고리가 이미 있습니다.")
    return FaqCategoryInfo(id=category_id, name=name, faq_count=count)


@router.delete('/faq-categories/{category_id}', status_code=204)
async def delete_category(
        category_id: int,
        tenant_id: str = Depends(get_tenant_id),
        session: AsyncSession = Depends(get_session),
):
    """소속 FAQ가 있으면 409 — 폴더 삭제와 같은 규칙(FE 요청 그대로). FAQ를 옮기거나 지운 뒤 삭제한다."""
    category = await _get_category(session, tenant_id, category_id)
    count = await _faq_count(session, tenant_id, category.id)
    if count:
        raise HTTPException(
            status_code=409,
            detail=f'카테고리에 FAQ {count}개가 있어 삭제할 수 없습니다. FAQ를 옮기거나 삭제한 뒤 다시 시도해 주세요.',
        )
    await session.delete(category)
    await session.commit()
