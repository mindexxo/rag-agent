"""FAQ 라우터 (F3 — 전용 저장, 검색은 chunks 통합).

항목 단위 CRUD. 저장·수정 시 항목 1개만 재임베딩해 chunks에 반영한다 (항목 단위 인덱싱).
검색 우선 관문·원문 반환 없음 — FAQ 청크는 일반 문서 청크와 같은 풀에서 경쟁 (B안 철학).

캐시 무효화 키: FAQ는 문서 id와 겹치지 않도록 음수 네임스페이스(-faq_id)를 쓴다.
semantic 캐시의 문서 집합 비교·무효화가 코드 수정 없이 그대로 동작한다 (service._source_doc_ids 참조).

관리 필드(#278, FE 요청 4~6): 카테고리(routers/faq_categories.py)·등록/변경자·인용 횟수·목록 검색/필터/페이징.
카테고리와 등록/변경자는 **관리용**이라 바꿔도 재임베딩·캐시 무효화가 없다 — 검색·생성 입력이 아니다.
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import Text, cast, exists, func, or_, select
from sqlalchemy import text as sql_text
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_session
from rag import cache, outbox
from rag.models import Faq, FaqCategory
from routers.kms import get_tenant_id, get_user_id
from schemas.kms import (DOC_LIST_DEFAULT_LIMIT, DOC_LIST_MAX_LIMIT, FaqCreateRequest, FaqListResponse,
                         FaqResponse, FaqUpdateRequest)
from text_norm import LIKE_ESCAPE_CHAR, like_pattern

router = APIRouter(prefix='/kms')


def _to_response(f: Faq, ref_count: int | None = None) -> FaqResponse:
    return FaqResponse(id=f.id, question=f.question, variants=f.variants or [], answer=f.answer, is_active=f.is_active,
                       category_id=f.category_id, created_at=f.created_at, updated_at=f.updated_at,
                       created_by=f.created_by, updated_by=f.updated_by, ref_count=ref_count)


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


async def _check_category(session: AsyncSession, tenant_id: str, category_id: int | None) -> None:
    """내 테넌트의 카테고리가 아니면 404 — 문서 업로드·PATCH가 남의 테넌트 폴더를 막는 것과 같은 규칙. None(미분류)은 통과."""
    if category_id is None:
        return
    found = (await session.execute(
        select(FaqCategory.id)
        .where(FaqCategory.tenant_id == tenant_id)   # 격리 — WHERE 절 명시
        .where(FaqCategory.id == category_id)
    )).scalar_one_or_none()
    if found is None:
        raise HTTPException(status_code=404, detail='category not found')


async def _ref_counts(session: AsyncSession, tenant_id: str, faq_ids: list[int]) -> dict[int, int]:
    """FAQ id별 인용 횟수 (#278) — 저장 시 확정된 실인용 목록(messages.cited_faq_ids)의 집계.

    문서의 ref_count(routers/documents.py _ref_counts, 파일명 키)와 같은 정의(실인용·목록 페이지로만 좁힘)다.
    #278 전 메시지는 FAQ 인용이 cited_docs의 'FAQ' 한 원소로만 남아 어느 항목인지 알 수 없다 — 안 센다(복원 불가).
    """
    if not faq_ids:
        return {}
    rows = (await session.execute(sql_text("""
        SELECT fid::bigint AS faq_id, count(*) AS cnt
        FROM messages, jsonb_array_elements_text(messages.cited_faq_ids) AS fid
        WHERE messages.tenant_id = :tenant_id
          AND messages.role = 'assistant'
          AND jsonb_typeof(messages.cited_faq_ids) = 'array'
          AND fid::bigint = ANY(:faq_ids)
        GROUP BY 1
    """), {"tenant_id": tenant_id, "faq_ids": faq_ids})).all()
    return {r.faq_id: r.cnt for r in rows}


def _list_where(tenant_id: str, q: str | None, category_id: int | None, is_active: bool | None):
    """목록 조회 조건 (#278). count와 페이지 쿼리가 같은 것을 써야 has_more가 어긋나지 않는다 — 문서 목록과 같은 구조.

    q: 질문·유사 질문·답변 중 하나라도 부분 일치(대소문자 무시). 유사 질문은 JSONB 배열을 **원소별로** 푼다
       (jsonb_array_elements_text) — 배열을 통째로 텍스트 캐스팅하면 `["a","b"]`의 따옴표·쉼표까지 검색 대상이 되어
       `"` 한 글자에 유사 질문 있는 FAQ가 전부 걸린다(리뷰 지적 2, 사용자 결정 b). 와일드카드 이스케이프는 문서 목록과
       같은 like_pattern. 호출부가 빈 문자열을 None으로 바꿔 넘긴다.
    category_id: 0 = 미분류만(문서 목록의 folder_id=0과 같은 약속 — 쿼리 파라미터엔 null이 없다), 그 외 = 그 카테고리.
    """
    where = (Faq.tenant_id == tenant_id)
    if q is not None:
        pattern = like_pattern(q)
        variant = func.jsonb_array_elements_text(Faq.variants).table_valued('value').alias('variant')
        variant_hit = exists(select(1).select_from(variant)
                             .where(variant.c.value.ilike(pattern, escape=LIKE_ESCAPE_CHAR))
                             .correlate(Faq))   # 바깥 faqs 행의 variants를 푼다 — correlate가 없으면 faqs를 한 번 더 조인한다
        where &= or_(Faq.question.ilike(pattern, escape=LIKE_ESCAPE_CHAR),
                     Faq.answer.ilike(pattern, escape=LIKE_ESCAPE_CHAR),
                     variant_hit)
    if category_id is not None:
        where &= Faq.category_id.is_(None) if category_id == 0 else Faq.category_id == category_id
    if is_active is not None:
        where &= Faq.is_active == is_active
    return where


@router.get('/faqs', response_model=FaqListResponse)
async def list_faqs(
        limit: int = DOC_LIST_DEFAULT_LIMIT,
        offset: int = 0,
        q: str | None = None,                 # 질문·유사 질문·답변 부분 일치
        category_id: int | None = None,       # 0 = 미분류만, 그 외 = 그 카테고리
        is_active: bool | None = None,
        tenant_id: str = Depends(get_tenant_id),
        session: AsyncSession = Depends(get_session),
):
    """FAQ 목록 — 페이징·검색·필터 (#278, FE 요청 6). 최신 등록이 위로 온다(문서 목록과 같은 정렬 기준).

    ⚠ #278에서 응답이 배열에서 `{items, total, has_more}`로 바뀌었다(FE 요청대로, 호출자는 FE 하나).
    파라미터 위생(limit 범위 밖은 상한으로, 음수 offset은 0, 빈 q는 미전송)은 문서 목록과 같다.
    """
    if not 1 <= limit <= DOC_LIST_MAX_LIMIT:
        limit = DOC_LIST_MAX_LIMIT
    if offset < 0:
        offset = 0
    q = (q or '').strip() or None      # 빈 문자열을 필터로 쓰면 '%%'가 되어 전건 매칭이다

    where = _list_where(tenant_id, q, category_id, is_active)
    total = (await session.execute(select(func.count(Faq.id)).where(where))).scalar_one()
    faqs = (await session.execute(
        select(Faq).where(where)
        .order_by(Faq.created_at.desc(), Faq.id.desc())
        .offset(offset).limit(limit)
    )).scalars().all()
    ref_counts = await _ref_counts(session, tenant_id, [f.id for f in faqs])
    return FaqListResponse(
        items=[_to_response(f, ref_counts.get(f.id, 0)) for f in faqs],
        total=total,
        has_more=offset + len(faqs) < total,   # len(faqs) — limit을 쓰면 마지막 부분 페이지에서 어긋난다
    )


@router.post('/faqs', response_model=FaqResponse)
async def create_faq(
        request: FaqCreateRequest,
        tenant_id: str = Depends(get_tenant_id),
        user_id: str | None = Depends(get_user_id),
        session: AsyncSession = Depends(get_session),
):
    # 공백 제거·빈 유사 질문 제거·길이 상한은 스키마가 했다(schemas/kms.py FAQ_*_MAX) — 여기선 빈 값만 본다.
    if not request.question or not request.answer:
        raise HTTPException(status_code=422, detail='질문/답변이 비어 있습니다.')
    await _check_category(session, tenant_id, request.category_id)
    await _reject_duplicate(session, tenant_id, [request.question, *request.variants])
    faq = Faq(
        tenant_id=tenant_id,
        question=request.question,
        variants=list(request.variants),
        answer=request.answer,
        category_id=request.category_id,
        created_by=user_id, updated_by=user_id,   # X-User-Id 미전송이면 둘 다 NULL(문서 uploaded_by와 같은 규칙)
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
    await session.refresh(faq)   # created_at·updated_at은 DB 기본값 — 응답에 실으려면 다시 읽는다
    return _to_response(faq, 0)


@router.patch('/faqs/{faq_id}', response_model=FaqResponse)
async def update_faq(
        faq_id: int,
        request: FaqUpdateRequest,
        tenant_id: str = Depends(get_tenant_id),
        user_id: str | None = Depends(get_user_id),
        session: AsyncSession = Depends(get_session),
):
    faq = await _get_faq(session, tenant_id, faq_id)

    # 빈 문자열 가드 (create는 스키마에서 막지만 PATCH는 optional이라 뚫려 있었음 — P1-15).
    # 빈 question/answer가 저장·재임베딩되면 검색에 쓰레기 항목이 남는다. strip·길이 상한은 스키마가 했다.
    if request.question is not None and not request.question:
        raise HTTPException(status_code=422, detail='질문은 비울 수 없습니다.')
    if request.answer is not None and not request.answer:
        raise HTTPException(status_code=422, detail='답변은 비울 수 없습니다.')
    # 카테고리는 null 전송(미분류로 이동)과 미전송(유지)을 가른다 — 문서 PATCH의 folder_id와 같은 3단.
    changing_category = 'category_id' in request.model_fields_set
    if changing_category:
        await _check_category(session, tenant_id, request.category_id)

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
    if changing_category:
        faq.category_id = request.category_id   # 관리용 분류 — 재임베딩·캐시 무효화 없음(검색·생성 입력이 아니다)
    # 마지막 변경자는 어떤 PATCH든 덮는다 — updated_at(어떤 UPDATE든 DB가 올림)과 같은 기준이어야 시각과 사람이 짝이 맞는다.
    # 헤더가 없으면 NULL로 덮는다(사용자 결정 5 — ICCS 연동에선 헤더가 항상 있다).
    faq.updated_by = user_id

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
    await session.refresh(faq)   # updated_at은 DB onupdate 값 — 응답에 실으려면 다시 읽는다
    ref_counts = await _ref_counts(session, tenant_id, [faq.id])
    return _to_response(faq, ref_counts.get(faq.id, 0))


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
