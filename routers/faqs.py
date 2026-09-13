"""FAQ 라우터 (F3 — 전용 저장, 검색은 chunks 통합).

항목 단위 CRUD. 저장·수정 시 항목 1개만 재임베딩해 chunks에 반영한다 (항목 단위 인덱싱).
검색 우선 관문·원문 반환 없음 — FAQ 청크는 일반 문서 청크와 같은 풀에서 경쟁 (B안 철학).

캐시 무효화 키: FAQ는 문서 id와 겹치지 않도록 음수 네임스페이스(-faq_id)를 쓴다.
semantic 캐시의 문서 집합 비교·무효화가 코드 수정 없이 그대로 동작한다 (service._source_doc_ids 참조).
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_session
from rag import cache, outbox
from rag.models import Faq
from routers.kms import get_tenant_id
from schemas.kms import FaqResponse, FaqCreateRequest, FaqUpdateRequest

router = APIRouter(prefix='/kms')


def _to_response(f: Faq) -> FaqResponse:
    return FaqResponse(id=f.id, question=f.question, variants=f.variants or [], answer=f.answer, is_active=f.is_active)


async def _get_faq(session: AsyncSession, tenant_id: str, faq_id: int) -> Faq:
    faq = (await session.execute(
        select(Faq)
        .where(Faq.tenant_id == tenant_id)   # 격리 — WHERE 절 명시
        .where(Faq.id == faq_id)
    )).scalars().first()
    if faq is None:
        raise HTTPException(status_code=404, detail='FAQ not found')
    return faq


@router.get('/faqs', response_model=list[FaqResponse])
async def list_faqs(
        tenant_id: str = Depends(get_tenant_id),
        session: AsyncSession = Depends(get_session),
):
    faqs = (await session.execute(
        select(Faq).where(Faq.tenant_id == tenant_id).order_by(Faq.id)
    )).scalars().all()
    return [_to_response(f) for f in faqs]


@router.post('/faqs', response_model=FaqResponse)
async def create_faq(
        request: FaqCreateRequest,
        tenant_id: str = Depends(get_tenant_id),
        session: AsyncSession = Depends(get_session),
):
    if not request.question.strip() or not request.answer.strip():
        raise HTTPException(status_code=422, detail='질문/답변이 비어 있습니다.')
    faq = Faq(
        tenant_id=tenant_id,
        question=request.question.strip(),
        variants=[v.strip() for v in request.variants if v.strip()],
        answer=request.answer.strip(),
    )
    session.add(faq)
    await session.flush()   # id 확보 (대기열 payload에 필요)

    # 임베딩·색인은 워커가 한다(rag/outbox.py INDEX_FAQ) — 라우터는 행만 같은 트랜잭션에 남긴다.
    # 그래서 생성 직후 최대 1분(재시도 포함 1~5분)은 검색에 안 잡힌다 — 제품 가이드 그대로.
    outbox.enqueue(session, tenant_id, outbox.INDEX_FAQ, faq_id=faq.id)
    await session.commit()
    return _to_response(faq)


@router.patch('/faqs/{faq_id}', response_model=FaqResponse)
async def update_faq(
        faq_id: int,
        request: FaqUpdateRequest,
        tenant_id: str = Depends(get_tenant_id),
        session: AsyncSession = Depends(get_session),
):
    faq = await _get_faq(session, tenant_id, faq_id)

    # 빈 문자열 가드 (create는 스키마에서 막지만 PATCH는 optional이라 뚫려 있었음 — P1-15).
    # 빈 question/answer가 저장·재임베딩되면 검색에 쓰레기 항목이 남는다.
    if request.question is not None and not request.question.strip():
        raise HTTPException(status_code=422, detail='질문은 비울 수 없습니다.')
    if request.answer is not None and not request.answer.strip():
        raise HTTPException(status_code=422, detail='답변은 비울 수 없습니다.')

    content_changed = False
    if request.question is not None and request.question.strip() != faq.question:
        faq.question = request.question.strip()
        content_changed = True
    if request.variants is not None:
        clean = [v.strip() for v in request.variants if v.strip()]
        if clean != (faq.variants or []):
            faq.variants = clean
            content_changed = True
    if request.answer is not None and request.answer.strip() != faq.answer:
        faq.answer = request.answer.strip()
        content_changed = True

    turned_off = False
    if request.is_active is not None:
        turned_off = faq.is_active and not request.is_active
        faq.is_active = request.is_active

    if content_changed or turned_off:
        # 이 항목을 근거로 만든 캐시는 즉시 무효화한다 — 색인 반영(아래 outbox)은 다음 회차라
        # 캐시까지 늦추면 낡은 답이 두 경로로 나간다.
        await cache.invalidate_source(session, tenant_id, -faq.id)   # 음수 = FAQ 네임스페이스

    # 색인 반영을 같은 트랜잭션에 적재 (#139) — 내용 변경은 재임베딩·재색인(INDEX_FAQ),
    # 활성 토글은 메타 부분 갱신(META_FAQS)이면 된다(재색인은 GPU를 태운다).
    if content_changed:
        outbox.enqueue(session, tenant_id, outbox.INDEX_FAQ, faq_id=faq.id)
    elif request.is_active is not None:
        outbox.enqueue(session, tenant_id, outbox.META_FAQS, faq_ids=[faq.id])
    await session.commit()
    return _to_response(faq)


@router.delete('/faqs/{faq_id}', status_code=204)
async def delete_faq(
        faq_id: int,
        tenant_id: str = Depends(get_tenant_id),
        session: AsyncSession = Depends(get_session),
):
    """항목 삭제 — 청크는 FK cascade로 함께 삭제, 관련 캐시 무효화."""
    faq = await _get_faq(session, tenant_id, faq_id)
    await cache.invalidate_source(session, tenant_id, -faq.id)
    await session.delete(faq)
    outbox.enqueue(session, tenant_id, outbox.DROP_FAQS, faq_ids=[faq_id])   # 같은 트랜잭션 (#139)
    await session.commit()
