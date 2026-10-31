"""RAG 답변 캐시 — Postgres semantic 캐시의 조회·저장·무효화.

조회 = 재작성문(standalone query) 임베딩의 최근접 1건 → floor 게이트 → 근거 문서 집합 비교
→ LLM 재사용 판정. 판정 승인 없이는 서빙하지 않는다.

정확성 보장 지도:
- 문서 재업로드/삭제/비활성: 조회 시 doc집합 비교가 자가치유. invalidate_source는 죽은 row 청소용.
- FAQ 수정: faq_id가 불변이라 자가치유 불가 → 명시 무효화 + save_answer의 updated_at 낙관적 검증.
- 모델 교체: 보호하지 않는다 — 교체 시 캐시를 일괄 비운다.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256

from sqlalchemy import delete, func, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from config import settings
from rag.llm_schemas import ReuseJudgment, acomplete_validated
from rag.models import AnswerCache as AnswerCacheRow
from rag.models import Faq
from rag.prompts import build_cache_reuse_judge_messages
from rag.search.embeddings import embed_query
from schemas.kms import SourceCitation

logger = logging.getLogger(__name__)


@dataclass
class CacheHit:
    answer: str
    sources: list[SourceCitation]
    source_doc_ids: list[int]


async def _verify_reuse(
    llm, cached_query: str, new_query: str, cached_original: str | None = None, new_original: str | None = None
) -> bool:
    """캐시 후보의 재사용 가부를 LLM에 묻는다 — 모든 히트의 필요조건.

    판정 실패는 False — 잘못 승인(오답 재생)이 잘못 거절(생성 1회)보다 비싸서 불확실하면 거절한다.
    *_query는 재작성문, *_original은 사용자 원문(판정 재료로만 쓴다). None이면 '(없음)'으로 판정한다.
    """
    try:
        judgment = await acomplete_validated(
            llm, build_cache_reuse_judge_messages(cached_query, new_query, cached_original, new_original), ReuseJudgment
        )
        logger.info(
            '캐시 재사용 판정(%s): %s — %s',
            getattr(llm, 'model_name', '?'),
            judgment.same_answer,
            judgment.reason[:120],
        )  # 모델 이름 — 보조 클라이언트가 쓰였는지 확인용
        return judgment.same_answer
    except Exception:
        logger.exception('캐시 재사용 판정 실패 — 거절로 처리')
        return False


def _normalize_query(query: str) -> str:
    normalized = ' '.join(query.strip().split())
    return normalized.rstrip("?.!。？！").lower()


def _build_cache_digest(standalone_query: str) -> str:
    return sha256(_normalize_query(standalone_query).encode("utf-8")).hexdigest()


def _sources_to_json(sources: list[SourceCitation]) -> list[dict]:
    return [source.model_dump() for source in sources]


def _sources_from_json(raw_sources: list[dict]) -> list[SourceCitation]:
    return [SourceCitation(**source) for source in raw_sources]


async def snapshot_faq_versions(
    session: AsyncSession,
    tenant_id: str,
    source_doc_ids: list[int],
) -> dict[int, datetime]:
    """근거 FAQ들의 {faq_id: updated_at} 스냅샷 — save_answer의 낙관적 검증 기준값.

    검색 직후 찍어두고 save_answer에서 재조회와 비교해, 생성 중 FAQ가 수정·무효화된 뒤
    옛 내용 기반 답변을 다시 적재하는 레이스를 막는다.
    """
    faq_ids = [-i for i in source_doc_ids if i < 0]
    if not faq_ids:
        return {}
    rows = (
        await session.execute(
            select(Faq.id, Faq.updated_at).where(Faq.tenant_id == tenant_id).where(Faq.id.in_(faq_ids))
        )
    ).all()
    return {fid: ts for fid, ts in rows}


async def get_semantic(
    session: AsyncSession,
    tenant_id: str,
    query: str,
    current_source_doc_ids: list[int],
    query_embedding: list[float] | None = None,
    llm=None,
    original_query: str | None = None,
) -> CacheHit | None:
    """semantic 캐시를 조회한다. 실패는 miss로 삼킨다 — 캐시가 요청을 죽이면 안 된다.

    hit 조건(순서대로): 최근접 후보 유사도 ≥ floor → 검색 doc집합 동일 → LLM 재사용 판정 승인.
    유사도만으로 서빙하는 경로는 없다 — llm이 None이면 항상 miss.

    query_embedding: 검색이 이미 만든 질의 벡터를 재사용한다(None이면 직접 임베딩). 쿼리에는
    색인용 프리픽스가 없어 같은 벡터이고, TEI 자체의 호출 간 비결정성 폭 안이다 —
    이 재사용 근거의 정의점이 여기다.
    original_query: 판정기 입력으로만 쓴다 — 조회 키·임베딩은 query(재작성문) 기준.
    """
    try:
        if query_embedding is None:
            query_embedding = (await embed_query(query)).dense

        distance = AnswerCacheRow.query_embedding.cosine_distance(query_embedding).label("distance")

        stmt = select(AnswerCacheRow, distance).where(AnswerCacheRow.tenant_id == tenant_id).order_by(distance).limit(1)
        row = (await session.execute(stmt)).first()

        if row is None:
            return None

        cache_row, cosine_distance = row
        similarity = 1 - cosine_distance

        # floor 미만은 판정 비용을 쓸 가치가 없는 잡음이다.
        if llm is None or similarity < settings.semantic_cache_floor:
            return None

        # 질문이 비슷해도 근거 문서가 다르면 다른 답이다. LLM 판정보다 먼저 — 싼 검사로 거른다.
        if set(cache_row.source_doc_ids) != set(current_source_doc_ids):
            return None

        # 원문도 넘긴다 — 같은 입력이 이력 차이로 다르게 재작성된 경우를 구제한다.
        if not await _verify_reuse(llm, cache_row.query_text, query, cache_row.original_query, original_query):
            logger.info('판정기가 캐시 재사용 거절 — miss (tenant=%s, sim=%.4f)', tenant_id, similarity)
            return None

        # DB 쪽 증가식이라 동시 히트에서도 카운터가 유실되지 않는다.
        await session.execute(
            update(AnswerCacheRow)
            .where(AnswerCacheRow.id == cache_row.id)
            .values(
                hit_count=AnswerCacheRow.hit_count + 1,
                last_hit_at=func.now(),
            )
        )
        return CacheHit(
            answer=cache_row.answer,
            sources=_sources_from_json(cache_row.sources),
            source_doc_ids=cache_row.source_doc_ids,
        )
    except Exception:
        logger.exception('semantic 캐시 조회 실패 — miss로 진행 (tenant=%s)', tenant_id)
        return None


async def save_answer(
    session: AsyncSession,
    tenant_id: str,
    query: str,
    answer: str,
    sources: list[SourceCitation],
    source_doc_ids: list[int],
    faq_versions: dict[int, datetime] | None = None,
    query_embedding: list[float] | None = None,
    original_query: str | None = None,
) -> None:
    """답변을 semantic 캐시에 upsert한다. 실패는 로그만 남기고 삼킨다 — 이미 나간 답변을 failed로 만들지 않게.

    faq_versions: snapshot_faq_versions의 스냅샷. None이면 FAQ 낙관적 검증을 생략한다.
    query_embedding: get_semantic과 같은 계약. original_query: 나중 턴의 판정기가 읽는다 —
    cache_key·임베딩은 query(재작성문) 기준이다(원문을 키에 섞으면 upsert 충돌 판정이 바뀐다).
    """
    try:
        # 생성 중 근거 FAQ가 바뀌었으면 저장하지 않는다. FOR SHARE 재조회라 동시 FAQ UPDATE는
        # 이 트랜잭션 커밋까지 대기하고, 그 뒤의 무효화가 방금 쓴 row를 지운다.
        if faq_versions is not None and not await _faqs_unchanged(session, tenant_id, source_doc_ids, faq_versions):
            logger.info('근거 FAQ가 생성 중 변경됨 — 캐시 저장 스킵 (tenant=%s)', tenant_id)
            return

        if query_embedding is None:
            query_embedding = (await embed_query(query)).dense

        # cache_key = 정규화 query의 digest — UNIQUE(tenant_id, cache_key)의 충돌 판정 키.
        stmt = pg_insert(AnswerCacheRow).values(
            tenant_id=tenant_id,
            cache_key=_build_cache_digest(query),
            query_text=query,
            original_query=original_query,
            query_embedding=query_embedding,
            answer=answer,
            sources=_sources_to_json(sources),
            source_doc_ids=source_doc_ids,
            model=settings.llm_model,
        )

        # hit_count=0 리셋은 의도다 — 답변이 교체됐으니 옛 답의 적중 통계를 물려받지 않는다.
        stmt = stmt.on_conflict_do_update(
            index_elements=["tenant_id", "cache_key"],
            set_={
                "query_text": query,
                "original_query": original_query,
                "query_embedding": query_embedding,
                "answer": answer,
                "sources": _sources_to_json(sources),
                "source_doc_ids": source_doc_ids,
                "model": settings.llm_model,
                "last_hit_at": func.now(),
                "hit_count": 0,
            },
        )
        await session.execute(stmt)
    except Exception:
        logger.exception('semantic 캐시 저장 실패 — 스킵 (tenant=%s)', tenant_id)


async def _faqs_unchanged(
    session: AsyncSession,
    tenant_id: str,
    source_doc_ids: list[int],
    snapshot: dict[int, datetime],
) -> bool:
    """근거 FAQ들이 스냅샷 이후 그대로인가. 소실·비활성·스냅샷에 없던 id는 '변경'으로 본다."""
    faq_ids = [-i for i in source_doc_ids if i < 0]
    if not faq_ids:
        return True
    rows = (
        await session.execute(
            select(Faq.id, Faq.updated_at, Faq.is_active)
            .where(Faq.tenant_id == tenant_id)
            .where(Faq.id.in_(faq_ids))
            .with_for_update(read=True)  # FOR SHARE — 커밋까지 동시 FAQ UPDATE 차단
        )
    ).all()
    current = {fid: (ts, active) for fid, ts, active in rows}
    for fid in faq_ids:
        got = current.get(fid)
        if got is None or not got[1] or snapshot.get(fid) != got[0]:
            return False
    return True


async def sweep_stale(session: AsyncSession) -> int:
    """cache_retention_days 동안 히트 없는 row를 전 테넌트에서 지우고 건수를 반환한다.

    doc집합이 낡아 영영 못 맞는 죽은 row를 걷는다 — last_hit_at은 히트마다 갱신된다.
    """
    result = await session.execute(
        delete(AnswerCacheRow).where(
            AnswerCacheRow.last_hit_at < func.now() - func.make_interval(0, 0, 0, settings.cache_retention_days)
        )
    )
    return result.rowcount


async def invalidate_source(
    session: AsyncSession,
    tenant_id: str,
    source_id: int,
) -> None:
    """source_id를 근거로 만든 캐시 row를 지운다. 양수 = document_id, 음수(-faq_id) = FAQ.

    FAQ 수정·삭제·토글은 자가치유되지 않아 반드시 이 함수로 무효화해야 한다.
    """
    await session.execute(
        delete(AnswerCacheRow)
        .where(AnswerCacheRow.tenant_id == tenant_id)
        .where(AnswerCacheRow.source_doc_ids.any(source_id))
    )
