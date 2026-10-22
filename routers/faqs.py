"""FAQ 라우터 (F3 — 전용 저장, 검색은 chunks 통합).

항목 단위 CRUD. 저장·수정 시 항목 1개만 재임베딩해 chunks에 반영한다 (항목 단위 인덱싱).
검색 우선 관문·원문 반환 없음 — FAQ 청크는 일반 문서 청크와 같은 풀에서 경쟁 (B안 철학).

캐시 무효화 키: FAQ는 문서 id와 겹치지 않도록 음수 네임스페이스(-faq_id)를 쓴다.
semantic 캐시의 문서 집합 비교·무효화가 코드 수정 없이 그대로 동작한다 (service._source_doc_ids 참조).
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import Text, cast, func, or_, select
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_session
from rag import cache, outbox
from rag.models import Faq
from routers.kms import get_tenant_id
from schemas.kms import FaqResponse, FaqCreateRequest, FaqUpdateRequest

router = APIRouter(prefix='/kms')


def _to_response(f: Faq) -> FaqResponse:
    return FaqResponse(id=f.id, question=f.question, variants=f.variants or [], answer=f.answer, is_active=f.is_active)


async def _reject_duplicate(session: AsyncSession, tenant_id: str, phrases: list[str],
                            exclude_id: int | None = None) -> None:
    """대표 질문·유사 질문이 같은 테넌트의 다른 FAQ(대표 질문 또는 유사 질문)와 **완전 일치**하면 409 (#276).

    완전 일치만 본다(공백 정리는 스키마가 했다) — 띄어쓰기·대소문자를 무시하면 정당한 변형 질문까지 막는다.
    같은 질문이 두 항목에 있으면 둘 다 후보에 올라 리랭커 점수로 순서가 정해지는데, 내용이 같으면
    점수도 거의 같아 어느 쪽이 먼저일지 비결정이다 — 등록 시점에 막는 이유.
    `?|`(jsonb_exists_any): variants 배열에 후보 문구 중 하나라도 있으면 참.
    """
    if not phrases:
        return
    stmt = (select(Faq.id, Faq.question, Faq.variants)
            .where(Faq.tenant_id == tenant_id)   # 격리 — WHERE 절 명시
            .where(or_(Faq.question.in_(phrases),
                       func.jsonb_exists_any(Faq.variants, cast(phrases, ARRAY(Text))))))
    if exclude_id is not None:
        stmt = stmt.where(Faq.id != exclude_id)
    for other_id, question, variants in (await session.execute(stmt)).all():
        hit = next(p for p in phrases if p == question or p in (variants or []))
        raise HTTPException(status_code=409, detail=f"같은 질문의 FAQ(#{other_id})가 이미 있습니다: '{hit}'")


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
    # 공백 제거·빈 유사 질문 제거·길이 상한은 스키마가 했다(schemas/kms.py FAQ_*_MAX) — 여기선 빈 값만 본다.
    if not request.question or not request.answer:
        raise HTTPException(status_code=422, detail='질문/답변이 비어 있습니다.')
    await _reject_duplicate(session, tenant_id, [request.question, *request.variants])
    faq = Faq(
        tenant_id=tenant_id,
        question=request.question,
        variants=list(request.variants),
        answer=request.answer,
    )
    session.add(faq)
    try:
        await session.flush()   # id 확보 (대기열 payload에 필요)
    except IntegrityError:
        # UNIQUE (tenant_id, question) — 위 검사와 커밋 사이에 같은 질문이 들어온 경쟁. 폴더 라우터의 IntegrityError→409 변환과 같다
        await session.rollback()
        raise HTTPException(status_code=409, detail=f"같은 질문의 FAQ가 이미 있습니다: '{request.question}'")

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
    # 빈 question/answer가 저장·재임베딩되면 검색에 쓰레기 항목이 남는다. strip·길이 상한은 스키마가 했다.
    if request.question is not None and not request.question:
        raise HTTPException(status_code=422, detail='질문은 비울 수 없습니다.')
    if request.answer is not None and not request.answer:
        raise HTTPException(status_code=422, detail='답변은 비울 수 없습니다.')

    # 바뀐 뒤의 모습을 먼저 계산해 중복을 검사하고, 통과하면 그때 대입한다 — 대입을 먼저 하면 검사 SELECT 직전
    # autoflush가 UPDATE를 내보내 DB UNIQUE에서 터진다(409가 아니라 500, 리뷰 지적 1).
    question = request.question if request.question is not None else faq.question
    variants = list(request.variants) if request.variants is not None else (faq.variants or [])
    answer = request.answer if request.answer is not None else faq.answer
    content_changed = (question, variants, answer) != (faq.question, faq.variants or [], faq.answer)
    if content_changed:
        await _reject_duplicate(session, tenant_id, [question, *variants], exclude_id=faq.id)   # 자기 자신은 제외
        faq.question, faq.variants, faq.answer = question, variants, answer
        try:
            await session.flush()   # UNIQUE (tenant_id, question) 경쟁을 여기서 받는다 — 뒤의 캐시 무효화 질의가 autoflush로 먼저 터지지 않게
        except IntegrityError:
            await session.rollback()   # 폴더 라우터의 IntegrityError→409 변환과 같다. rollback이 faq를 만료시키므로 지역 변수를 쓴다
            raise HTTPException(status_code=409, detail=f"같은 질문의 FAQ가 이미 있습니다: '{question}'")

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
