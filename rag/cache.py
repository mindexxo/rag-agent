"""RAG 답변 캐시 — Postgres semantic 단층.

조회 = standalone query 임베딩의 pgvector 최근접 1건 → threshold → 근거 문서 집합 비교.
캐시 키는 질의 재작성된 standalone query 기준. (과거의 Redis exact 계층은 제거됨)

정확성 보장 지도 (#16 분석 — 어떤 변경이 어떤 메커니즘으로 잡히는가):
- 문서 재업로드/삭제/비활성: document_id가 바뀌거나 검색에서 빠짐 → 조회 시 doc집합
  비교(get_semantic)가 자가치유. 명시 무효화(invalidate_document)는 죽은 row 청소용.
- FAQ 수정: faq_id 불변이라 자가치유 불가 → routers/faqs.py의 명시 무효화 +
  생성 중(in-flight) write-back 레이스는 set의 updated_at 낙관적 검증이 차단.
- 모델 교체: 미보호 — 8B 전환 전 조회 model 필터 또는 일괄 flush 필요 (백로그).
"""
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from hashlib import sha256
from sqlalchemy import select, update, func, delete
from sqlalchemy.ext.asyncio import AsyncSession

from config import settings
from rag.embeddings import embed_query
from schemas.kms import SourceCitation
from rag.models import AnswerCache as AnswerCacheRow, Faq
from sqlalchemy.dialects.postgresql import insert as pg_insert

logger = logging.getLogger(__name__)


@dataclass
class CacheHit:
    answer: str
    sources: list[SourceCitation]
    source_doc_ids: list[int]
    kind: Literal["semantic"]

def normalize_query(query: str) -> str:
    """캐시 키 생성을 위해 질의를 정규화한다."""
    normalized = ' '.join(query.strip().split())
    return normalized.rstrip("?.!。？！").lower()

def build_cache_digest(standalone_query: str) -> str:
    """정규화된 standalone_query의 sha256 digest를 만든다."""
    return sha256(normalize_query(standalone_query).encode("utf-8")).hexdigest()


def sources_to_json(sources: list[SourceCitation]) -> list[dict]:
    """SourceCitation 리스트를 JSON 저장 가능한 dict 리스트로 변환한다."""
    return [source.model_dump() for source in sources]

def sources_from_json(raw_sources: list[dict]) -> list[SourceCitation]:
    """JSON dict 리스트를 SourceCitation 리스트로 복원한다."""
    return [SourceCitation(**source) for source in raw_sources]


async def snapshot_faq_versions(
        session: AsyncSession, tenant_id: str, source_doc_ids: list[int],
) -> dict[int, datetime]:
    """근거에 포함된 FAQ들의 {faq_id: updated_at} 스냅샷 — set의 낙관적 검증 기준값.

    prepare(검색 직후) 시점에 찍어두고, 생성이 끝난 set 시점에 재조회와 등치 비교한다.
    FAQ는 수정돼도 id가 불변이라 doc집합 비교가 자가치유하지 못하는 유일한 출처 —
    무효화 커밋과 생성 구간이 겹치는 write-back 레이스를 이 검증이 막는다 (#16).
    """
    faq_ids = [-i for i in source_doc_ids if i < 0]
    if not faq_ids:
        return {}
    rows = (await session.execute(
        select(Faq.id, Faq.updated_at)
        .where(Faq.tenant_id == tenant_id)     # 격리 — 모든 조회에 tenant WHERE (프로젝트 원칙)
        .where(Faq.id.in_(faq_ids))
    )).all()
    return {fid: ts for fid, ts in rows}

class AnswerCache:
    """Postgres semantic 답변 캐시."""

    async def get_semantic(self, session: AsyncSession, tenant_id: str, query: str, current_source_doc_ids: list[int]) -> CacheHit | None:
        """Postgres semantic cache를 조회한다.

        질문 임베딩 유사도가 충분히 높고, 현재 검색된 문서 집합이 캐시 엔트리의
        문서 집합과 같을 때만 hit로 인정한다.

        fail-open(#16): 캐시는 어떤 경우에도 요청을 죽이면 안 된다 — 조회 실패는 miss 취급.
        (실측 실패 모드: 임베딩 TEI 블립. 검색까지 성공한 요청이 캐시 조회에서 500 나던 문제)
        """
        try:
            return await self._get_semantic(session, tenant_id, query, current_source_doc_ids)
        except Exception:
            logger.exception('semantic 캐시 조회 실패 — miss로 진행 (tenant=%s)', tenant_id)
            return None

    async def _get_semantic(self, session: AsyncSession, tenant_id: str, query: str, current_source_doc_ids: list[int]) -> CacheHit | None:
        # 1. 새 질문을 임베딩해서 기존 캐시 질문들과 벡터 유사도 비교에 사용한다.
        query_embedding = (await embed_query(query)).dense

        # 2. pgvector cosine distance 식을 만든다.
        # distance는 작을수록 유사하다. similarity로 바꿀 때는 1 - distance를 사용한다.
        distance = AnswerCacheRow.query_embedding.cosine_distance(query_embedding).label("distance")

        # 3. 같은 tenant의 캐시 중 새 질문과 가장 가까운 캐시 row 1개만 조회한다.
        stmt = (
            select(AnswerCacheRow, distance)
            .where(AnswerCacheRow.tenant_id == tenant_id)
            .order_by(distance)
            .limit(1)
        )
        row = (await session.execute(stmt)).first()

        # 4. 캐시 row가 하나도 없으면 semantic cache miss.
        if row is None:
            return None

        cache_row, cosine_distance = row
        similarity = 1 - cosine_distance

        # 5. 가장 가까운 캐시라도 threshold보다 낮으면 의미가 충분히 같지 않다고 본다
        if similarity < settings.semantic_cache_threshold:
            return None

        # 6. 질문 의미가 비슷해도, 현재 검색 결과의 근거 문서 집합이 다르면 miss 처리한다.
        # 비슷하지만 다른 정책/문서에 답해야 하는 케이스에서 오답 재사용을 막기 위함이다.
        if set(cache_row.source_doc_ids) != set(current_source_doc_ids):
            return None

        # 7. semantic hit 통계 갱신.
        # hit_count = hit_count + 1 형태라 동시 요청에서도 카운터 손실이 x.
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
            sources=sources_from_json(cache_row.sources),
            source_doc_ids=cache_row.source_doc_ids,
            kind="semantic",
        )

    async def set(
            self,
            session: AsyncSession,
            tenant_id: str,
            query: str,
            answer: str,
            sources: list[SourceCitation],
            source_doc_ids: list[int],
            faq_versions: dict[int, datetime] | None = None,
    ) -> None:
        """LLM 응답을 Postgres semantic cache에 저장한다.

        faq_versions: prepare 시점의 {faq_id: updated_at} 스냅샷(snapshot_faq_versions).
        None이면 검증 생략(FAQ 근거가 없거나 테스트 등 직접 호출).

        fail-open(#16): 저장 실패는 삼키고 로그만 — 이미 완성돼 클라이언트로 나간 답변이
        캐시 저장 실패 때문에 failed로 기록되던 문제 방지. 단 DB 오류로 세션이 오염된
        경우는 이후 커밋이 어차피 실패하므로 여기서 감추지 않는 편이 낫지만, 주 실패
        모드(임베딩 TEI 블립)는 DB 접근 전이라 세션이 깨끗하다.
        """
        try:
            await self._set(session, tenant_id, query, answer, sources, source_doc_ids, faq_versions)
        except Exception:
            logger.exception('semantic 캐시 저장 실패 — 스킵 (tenant=%s)', tenant_id)

    async def _set(
            self,
            session: AsyncSession,
            tenant_id: str,
            query: str,
            answer: str,
            sources: list[SourceCitation],
            source_doc_ids: list[int],
            faq_versions: dict[int, datetime] | None,
    ) -> None:
        # 낙관적 검증(#16): 생성하는 동안 근거 FAQ가 수정/삭제/비활성됐으면 저장을 스킵.
        # FAQ PATCH의 무효화 커밋 '이후'에 옛 내용 기반 답변을 다시 심는 레이스 차단.
        # FOR SHARE로 재조회 — 검증과 INSERT 커밋 사이의 ms 창까지 닫는다: 동시 FAQ UPDATE는
        # 이 트랜잭션 커밋까지 대기하고, 대기 후 실행되는 무효화가 방금 쓴 row를 지운다.
        if faq_versions is not None and not await self._faqs_unchanged(
                session, tenant_id, source_doc_ids, faq_versions):
            logger.info('근거 FAQ가 생성 중 변경됨 — 캐시 저장 스킵 (tenant=%s)', tenant_id)
            return

        # semantic cache용으로 query embedding을 저장한다.
        # 이후 비슷한 질문이 들어오면 pgvector cosine distance로 이 row를 찾는다.
        query_embedding = (await embed_query(query)).dense

        # 4. Postgres answer_cache에 저장할 INSERT 문을 만든다.
        # cache_key는 Redis key 전체가 아니라 정규화 query digest만 저장한다.
        stmt = pg_insert(AnswerCacheRow).values(
            tenant_id=tenant_id,
            cache_key=build_cache_digest(query),
            query_text=query,
            query_embedding=query_embedding,
            answer=answer,
            sources=sources_to_json(sources),
            source_doc_ids=source_doc_ids,
            model=settings.vllm_model,
        )

        # 5. 같은 tenant + cache_key가 이미 있으면 UPDATE한다.
        # Redis TTL 만료 후 DB row가 남아 있거나, 동시 요청이 같은 캐시를 쓰는 경우
        # unique 제약 오류 대신 최신 답변으로 갱신하기 위함이다.
        stmt = stmt.on_conflict_do_update(
            index_elements=["tenant_id", "cache_key"],
            set_={
                "query_text": query,
                "query_embedding": query_embedding,
                "answer": answer,
                "sources": sources_to_json(sources),
                "source_doc_ids": source_doc_ids,
                "model": settings.vllm_model,
                "last_hit_at": func.now(),
                "hit_count": 0,
            },
        )
        await session.execute(stmt)

    async def _faqs_unchanged(
            self,
            session: AsyncSession,
            tenant_id: str,
            source_doc_ids: list[int],
            snapshot: dict[int, datetime],
    ) -> bool:
        """근거 FAQ들이 prepare 스냅샷 이후 안 바뀌었는가 — 등치 비교(순서 아님).

        row 소실·비활성도 '변경'으로 간주. 스냅샷에 없던 faq_id가 근거에 있으면
        (스냅샷 조회와 검색 사이 정합 깨짐) 보수적으로 변경 취급.
        """
        faq_ids = [-i for i in source_doc_ids if i < 0]
        if not faq_ids:
            return True
        rows = (await session.execute(
            select(Faq.id, Faq.updated_at, Faq.is_active)
            .where(Faq.tenant_id == tenant_id)
            .where(Faq.id.in_(faq_ids))
            .with_for_update(read=True)          # FOR SHARE — 커밋까지 동시 FAQ UPDATE 차단
        )).all()
        current = {fid: (ts, active) for fid, ts, active in rows}
        for fid in faq_ids:
            got = current.get(fid)
            if got is None or not got[1] or snapshot.get(fid) != got[0]:
                return False
        return True

    async def sweep_stale(self, session: AsyncSession) -> int:
        """cache_retention_days 동안 히트 없는 row 삭제 → 삭제 건수 반환 (arq 크론이 일 1회 호출).

        last_hit_at은 생성 시 now(), 히트마다 갱신 — 자주 맞는 답변은 계속 살고
        죽은 row(문서 개정 등으로 doc집합이 낡아 영영 못 맞는 것)만 걷힌다.
        전 테넌트 일괄 — 보존 기간은 테넌트 무관 위생 정책.
        """
        result = await session.execute(
            delete(AnswerCacheRow)
            .where(AnswerCacheRow.last_hit_at
                   < func.now() - func.make_interval(0, 0, 0, settings.cache_retention_days))
        )
        return result.rowcount

    async def invalidate_document(
            self,
            session: AsyncSession,
            tenant_id: str,
            document_id: int,
    ) -> None:
        """특정 문서에 의존한 semantic 캐시(PG)를 무효화한다.

        문서 재업로드/삭제/비활성화 시 그 document_id를 근거로 만든 semantic 캐시 row 삭제.
        (exact 캐시는 제거됨. semantic은 조회 시 doc집합 비교로도 자가치유하나, 여기서 명시
         삭제해 재업로드/삭제 후 죽은 row가 쌓이지 않게 한다.)
        """
        # Postgres semantic cache 삭제
        await session.execute(
            delete(AnswerCacheRow)
            .where(AnswerCacheRow.tenant_id == tenant_id)
            .where(AnswerCacheRow.source_doc_ids.any(document_id))
        )

