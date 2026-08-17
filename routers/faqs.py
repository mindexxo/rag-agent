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
from rag import cache
from rag.embeddings import embed_texts
from rag.faq_indexing import build_faq_chunk_text, reindex_faq
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


async def _embed_chunk_text(question: str, variants: list[str], answer: str):
    """임베딩 — async(AsyncClient) 전환으로 스레드풀 불필요, 직접 await."""
    text = build_faq_chunk_text(question, variants, answer)
    return (await embed_texts([text]))[0]


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
    await session.flush()   # id 확보 (청크 FK에 필요)

    embedding = await _embed_chunk_text(faq.question, faq.variants, faq.answer)
    await reindex_faq(session, faq, embedding)
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

    if content_changed:
        # 내용이 바뀌면 청크 재임베딩 + 이 항목을 근거로 만든 캐시 무효화
        embedding = await _embed_chunk_text(faq.question, faq.variants or [], faq.answer)
        await reindex_faq(session, faq, embedding)
    if content_changed or turned_off:
        await cache.invalidate_source(session, tenant_id, -faq.id)   # 음수 = FAQ 네임스페이스

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
    await session.commit()
