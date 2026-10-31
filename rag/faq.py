"""FAQ·FAQ 카테고리 관리 — 조회·중복 거절·생성·수정·삭제와 색인 대기열 적재. HTTP 변환은 라우터가 한다.

캐시 무효화 키: FAQ는 문서 id와 겹치지 않도록 음수 네임스페이스(-faq_id)를 쓴다 — semantic 캐시의 문서 집합
비교·무효화가 그대로 동작한다(rag/turn/service.py의 _source_doc_ids). 카테고리와 등록/변경자는 관리용이라
바꿔도 재임베딩·캐시 무효화가 없다 — 검색·생성 입력이 아니다.
"""

from typing import NamedTuple

from sqlalchemy import ColumnElement, Text, cast, exists, func, or_, select
from sqlalchemy import text as sql_text
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from rag import outbox
from rag.models import Faq, FaqCategory
from rag.turn import cache
from schemas.kms import FaqCreateRequest, FaqUpdateRequest
from text_norm import LIKE_ESCAPE_CHAR, like_pattern


class FaqNotFound(Exception):
    """내 테넌트에 없는 FAQ."""


class FaqCategoryNotFound(Exception):
    """내 테넌트에 없는 FAQ 카테고리."""


class FaqConflict(Exception):
    """중복 질문·같은 이름 카테고리·FAQ가 남은 카테고리 삭제. `str(e)`가 그대로 사용자에게 나간다."""


class FaqFilter(NamedTuple):
    """목록 조건. q는 질문·유사 질문·답변 부분 일치(빈 값은 호출부가 None으로), category_id 0은 미분류만."""

    q: str | None
    category_id: int | None
    is_active: bool | None


class FaqPage(NamedTuple):
    faqs: list[Faq]
    total: int
    ref_counts: dict[int, int]


def without_question(question: str, variants: list[str]) -> list[str]:
    """대표 질문과 같은 유사 질문은 뺀다 — 같은 문구가 두 자리에 있어도 검색엔 이득이 없다. 항목 간 중복은 스키마가 걷어냈다."""
    return [v for v in variants if v != question]


async def reject_duplicate(
    session: AsyncSession, tenant_id: str, phrases: list[str], exclude_id: int | None = None
) -> None:
    """대표 질문·유사 질문이 같은 테넌트의 다른 FAQ(대표 질문 또는 유사 질문)와 **완전 일치**하면 FaqConflict.

    완전 일치만 본다(공백 정리는 스키마가 했다) — 띄어쓰기·대소문자를 무시하면 정당한 변형 질문까지 막는다.
    같은 질문이 두 항목에 있으면 둘 다 후보에 올라 리랭커 점수로 순서가 정해지는데, 내용이 같으면
    점수도 거의 같아 어느 쪽이 먼저일지 비결정이다 — 등록 시점에 막는 이유.
    `?|`(jsonb_exists_any): variants 배열에 후보 문구 중 하나라도 있으면 참.
    """
    if not phrases:
        return
    stmt = (
        select(Faq.id, Faq.question, Faq.variants)
        .where(Faq.tenant_id == tenant_id)  # 격리 — WHERE 절 명시
        .where(or_(Faq.question.in_(phrases), func.jsonb_exists_any(Faq.variants, cast(phrases, ARRAY(Text)))))
    )
    if exclude_id is not None:
        stmt = stmt.where(Faq.id != exclude_id)
    for other_id, question, variants in (await session.execute(stmt)).all():
        hit = next(p for p in phrases if p == question or p in (variants or []))
        raise FaqConflict(f"같은 질문의 FAQ(#{other_id})가 이미 있습니다: '{hit}'")


async def get_faq(session: AsyncSession, tenant_id: str, faq_id: int) -> Faq:
    faq = (
        (
            await session.execute(
                select(Faq)
                .where(Faq.tenant_id == tenant_id)  # 격리 — WHERE 절 명시
                .where(Faq.id == faq_id)
            )
        )
        .scalars()
        .first()
    )
    if faq is None:
        raise FaqNotFound
    return faq


async def check_category(session: AsyncSession, tenant_id: str, category_id: int | None) -> None:
    """내 테넌트의 카테고리가 아니면 FaqCategoryNotFound — 문서가 남의 테넌트 폴더를 막는 것과 같은 규칙. None(미분류)은 통과."""
    if category_id is None:
        return
    found = (
        await session.execute(
            select(FaqCategory.id)
            .where(FaqCategory.tenant_id == tenant_id)  # 격리 — WHERE 절 명시
            .where(FaqCategory.id == category_id)
        )
    ).scalar_one_or_none()
    if found is None:
        raise FaqCategoryNotFound


async def ref_counts(session: AsyncSession, tenant_id: str, faq_ids: list[int]) -> dict[int, int]:
    """FAQ id별 인용 횟수 — 저장 시 확정된 실인용 목록(messages.cited_faq_ids)의 집계.

    문서의 ref_count(rag/document_admin.py ref_counts, 파일명 키)와 같은 정의(실인용·목록 페이지로만 좁힘)다.
    항목별 인용 도입 전 메시지는 FAQ 인용이 cited_docs의 'FAQ' 한 원소로만 남아 어느 항목인지 알 수 없다 — 안 센다(복원 불가).
    """
    if not faq_ids:
        return {}
    rows = (
        await session.execute(
            sql_text("""
        SELECT fid::bigint AS faq_id, count(*) AS cnt
        FROM messages, jsonb_array_elements_text(messages.cited_faq_ids) AS fid
        WHERE messages.tenant_id = :tenant_id
          AND messages.role = 'assistant'
          AND jsonb_typeof(messages.cited_faq_ids) = 'array'
          AND fid::bigint = ANY(:faq_ids)
        GROUP BY 1
    """),
            {"tenant_id": tenant_id, "faq_ids": faq_ids},
        )
    ).all()
    return {r.faq_id: r.cnt for r in rows}


def _list_where(tenant_id: str, f: FaqFilter) -> ColumnElement[bool]:
    """목록 조회 조건. count와 페이지 쿼리가 같은 것을 써야 has_more가 어긋나지 않는다 — 문서 목록과 같은 구조.

    유사 질문은 JSONB 배열을 **원소별로** 푼다(jsonb_array_elements_text) — 배열을 통째로 텍스트 캐스팅하면
    `["a","b"]`의 따옴표·쉼표까지 검색 대상이 되어 `"` 한 글자에 유사 질문 있는 FAQ가 전부 걸린다. 와일드카드
    이스케이프는 문서 목록과 같은 like_pattern. category_id 0은 문서 목록의 folder_id=0과 같은 약속(쿼리 파라미터엔 null이 없다).
    """
    where = Faq.tenant_id == tenant_id
    if f.q is not None:
        pattern = like_pattern(f.q)
        variant = func.jsonb_array_elements_text(Faq.variants).table_valued('value').alias('variant')
        variant_hit = exists(
            select(1).select_from(variant).where(variant.c.value.ilike(pattern, escape=LIKE_ESCAPE_CHAR)).correlate(Faq)
        )  # 바깥 faqs 행의 variants를 푼다 — correlate가 없으면 faqs를 한 번 더 조인한다
        where &= or_(
            Faq.question.ilike(pattern, escape=LIKE_ESCAPE_CHAR),
            Faq.answer.ilike(pattern, escape=LIKE_ESCAPE_CHAR),
            variant_hit,
        )
    if f.category_id is not None:
        where &= Faq.category_id.is_(None) if f.category_id == 0 else Faq.category_id == f.category_id
    if f.is_active is not None:
        where &= Faq.is_active == f.is_active
    return where


async def list_faqs(session: AsyncSession, tenant_id: str, f: FaqFilter, limit: int, offset: int) -> FaqPage:
    """최신 등록이 위로(문서 목록과 같은 정렬 기준). limit·offset 위생은 호출부가 했다."""
    where = _list_where(tenant_id, f)
    total = (await session.execute(select(func.count(Faq.id)).where(where))).scalar_one()
    faqs = list(
        (
            await session.execute(
                select(Faq).where(where).order_by(Faq.created_at.desc(), Faq.id.desc()).offset(offset).limit(limit)
            )
        )
        .scalars()
        .all()
    )
    return FaqPage(faqs, total, await ref_counts(session, tenant_id, [x.id for x in faqs]))


async def create_faq(session: AsyncSession, tenant_id: str, user_id: str | None, request: FaqCreateRequest) -> Faq:
    """빈 질문·답변은 호출부가 거절했다. 색인은 워커가 한다 — 같은 트랜잭션에 INDEX_FAQ 행만 남긴다."""
    await check_category(session, tenant_id, request.category_id)
    variants = without_question(request.question, request.variants)
    await reject_duplicate(session, tenant_id, [request.question, *variants])
    faq = Faq(
        tenant_id=tenant_id,
        question=request.question,
        variants=variants,
        answer=request.answer,
        category_id=request.category_id,
        created_by=user_id,
        updated_by=user_id,  # X-User-Id 미전송이면 둘 다 NULL(문서 uploaded_by와 같은 규칙)
    )
    session.add(faq)
    try:
        await session.flush()  # id 확보 (대기열 payload에 필요)
    except IntegrityError as e:
        # UNIQUE (tenant_id, question) — 위 검사와 커밋 사이에 같은 질문이 들어온 경쟁
        await session.rollback()
        raise FaqConflict(f"같은 질문의 FAQ가 이미 있습니다: '{request.question}'") from e

    # 생성 직후 최대 1분(재시도 포함 1~5분)은 검색에 안 잡힌다 — 제품 가이드 그대로.
    outbox.enqueue(session, tenant_id, outbox.INDEX_FAQ, faq_id=faq.id)
    await session.commit()
    await session.refresh(faq)  # created_at·updated_at은 DB 기본값 — 응답에 실으려면 다시 읽는다
    return faq


async def update_faq(session: AsyncSession, faq: Faq, user_id: str | None, request: FaqUpdateRequest) -> Faq:
    """빈 질문·답변은 호출부가 거절했다. 카테고리는 null 전송(미분류로 이동)과 미전송(유지)을 가른다."""
    tenant_id = faq.tenant_id
    changing_category = 'category_id' in request.model_fields_set
    if changing_category:
        await check_category(session, tenant_id, request.category_id)

    # 바뀐 뒤의 모습을 먼저 계산해 중복을 검사하고, 통과하면 그때 대입한다 — 대입을 먼저 하면 검사 SELECT 직전
    # autoflush가 UPDATE를 내보내 DB UNIQUE에서 터진다(409가 아니라 500).
    question = request.question if request.question is not None else faq.question
    variants = without_question(question, request.variants if request.variants is not None else (faq.variants or []))
    answer = request.answer if request.answer is not None else faq.answer
    content_changed = (question, variants, answer) != (faq.question, faq.variants or [], faq.answer)
    if content_changed:
        await reject_duplicate(session, tenant_id, [question, *variants], exclude_id=faq.id)  # 자기 자신은 제외
        faq.question, faq.variants, faq.answer = question, variants, answer
        try:
            await session.flush()  # UNIQUE 경쟁을 여기서 받는다 — 뒤의 캐시 무효화 질의가 autoflush로 먼저 터지지 않게
        except IntegrityError as e:
            await session.rollback()  # rollback이 faq를 만료시키므로 지역 변수를 쓴다
            raise FaqConflict(f"같은 질문의 FAQ가 이미 있습니다: '{question}'") from e

    turned_off = False
    if request.is_active is not None:
        turned_off = faq.is_active and not request.is_active
        faq.is_active = request.is_active
    if changing_category:
        faq.category_id = request.category_id
    # 마지막 변경자는 어떤 PATCH든 덮는다 — updated_at(어떤 UPDATE든 DB가 올림)과 같은 기준이어야 시각과 사람이 짝이 맞는다.
    faq.updated_by = user_id

    if content_changed or turned_off:
        # 이 항목을 근거로 만든 캐시는 즉시 무효화한다 — 색인 반영(아래 outbox)은 다음 회차라
        # 캐시까지 늦추면 낡은 답이 두 경로로 나간다.
        await cache.invalidate_source(session, tenant_id, -faq.id)  # 음수 = FAQ 네임스페이스

    # 내용 변경은 재임베딩·재색인(INDEX_FAQ), 활성 토글은 메타 부분 갱신(META_FAQS)이면 된다(재색인은 GPU를 태운다).
    if content_changed:
        outbox.enqueue(session, tenant_id, outbox.INDEX_FAQ, faq_id=faq.id)
    elif request.is_active is not None:
        outbox.enqueue(session, tenant_id, outbox.META_FAQS, faq_ids=[faq.id])
    await session.commit()
    await session.refresh(faq)  # updated_at은 DB onupdate 값 — 응답에 실으려면 다시 읽는다
    return faq


async def delete_faq(session: AsyncSession, faq: Faq) -> None:
    """청크는 FK cascade로 함께 삭제되고, 색인 삭제는 같은 트랜잭션의 DROP_FAQS 행으로 워커가 한다."""
    faq_id = faq.id
    await cache.invalidate_source(session, faq.tenant_id, -faq_id)
    await session.delete(faq)
    outbox.enqueue(session, faq.tenant_id, outbox.DROP_FAQS, faq_ids=[faq_id])
    await session.commit()


# ── 카테고리 — 폴더와 같은 1단 관리 분류. 검색에 영향이 없어 캐시 무효화도 색인 fan-out도 없다 ──


async def get_category(session: AsyncSession, tenant_id: str, category_id: int) -> FaqCategory:
    category = (
        (
            await session.execute(
                select(FaqCategory)
                .where(FaqCategory.tenant_id == tenant_id)  # 격리 — WHERE 절 명시
                .where(FaqCategory.id == category_id)
            )
        )
        .scalars()
        .first()
    )
    if category is None:
        raise FaqCategoryNotFound
    return category


async def faq_count(session: AsyncSession, tenant_id: str, category_id: int) -> int:
    return (
        await session.execute(
            select(func.count())
            .select_from(Faq)
            .where(Faq.tenant_id == tenant_id)
            .where(Faq.category_id == category_id)
        )
    ).scalar_one()


async def list_categories(session: AsyncSession, tenant_id: str) -> list[tuple[FaqCategory, int]]:
    """이름순 (카테고리, 소속 FAQ 수 — 활성·비활성 모두)."""
    rows = (
        await session.execute(
            select(FaqCategory, func.count(Faq.id))
            .outerjoin(Faq, (Faq.category_id == FaqCategory.id) & (Faq.tenant_id == FaqCategory.tenant_id))
            .where(FaqCategory.tenant_id == tenant_id)
            .group_by(FaqCategory.id)
            .order_by(FaqCategory.name)
        )
    ).all()
    return [(c, n) for c, n in rows]


async def create_category(session: AsyncSession, tenant_id: str, name: str) -> FaqCategory:
    category = FaqCategory(tenant_id=tenant_id, name=name)
    session.add(category)
    try:
        await session.commit()
    except IntegrityError as e:  # UNIQUE (tenant_id, name)
        await session.rollback()
        raise FaqConflict(f"'{name}' 카테고리가 이미 있습니다.") from e
    return category


async def rename_category(session: AsyncSession, category: FaqCategory, name: str) -> int:
    """이름을 바꾸고 소속 FAQ 수를 돌려준다(응답용)."""
    count = await faq_count(
        session, category.tenant_id, category.id
    )  # 대입 전에 센다 — 대입 뒤 SELECT는 autoflush로 UNIQUE에서 500이 난다
    category.name = name
    try:
        await session.commit()
    except IntegrityError as e:
        await session.rollback()  # rollback이 category를 만료시키므로 지역 변수만 쓴다
        raise FaqConflict(f"'{name}' 카테고리가 이미 있습니다.") from e
    return count


async def delete_category(session: AsyncSession, category: FaqCategory) -> None:
    """소속 FAQ가 있으면 FaqConflict — 폴더 삭제와 같은 규칙. FAQ를 옮기거나 지운 뒤 삭제한다."""
    count = await faq_count(session, category.tenant_id, category.id)
    if count:
        raise FaqConflict(
            f'카테고리에 FAQ {count}개가 있어 삭제할 수 없습니다. FAQ를 옮기거나 삭제한 뒤 다시 시도해 주세요.'
        )
    await session.delete(category)
    await session.commit()
