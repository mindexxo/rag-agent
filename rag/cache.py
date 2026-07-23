"""RAG 답변 캐시.

Postgres semantic cache를 다룬다 (Redis exact 계층은 제거됨 — doc 역인덱스 등 잔여 유틸만 Redis).
캐시 키는, 질의 재작성 된 query 기준
"""
from dataclasses import dataclass
from typing import Literal

from hashlib import sha256
from redis.asyncio import Redis
from sqlalchemy import select, update, func, delete
from sqlalchemy.ext.asyncio import AsyncSession

from config import settings
from rag.embeddings import embed_query
from schemas.kms import SourceCitation
from rag.models import AnswerCache as AnswerCacheRow
from sqlalchemy.dialects.postgresql import insert as pg_insert


@dataclass
class CacheHit:
    answer: str
    sources: list[SourceCitation]
    source_doc_ids: list[int]
    kind: Literal["exact", "semantic"]

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

class AnswerCache:
    """Redis exact + Postgres semantic 2계층 답변 캐시."""
    def __init__(self, redis: Redis | None = None):
        # 기본은 공용 싱글톤 재사용 (요청마다 새 커넥션 풀 생성 방지 — P1-12)
        from rag.clients import cache_redis
        self.redis = redis or cache_redis

    async def get_semantic(self, session: AsyncSession, tenant_id: str, query: str, current_source_doc_ids: list[int]) -> CacheHit | None:
        """Postgres semantic cache를 조회한다.

        질문 임베딩 유사도가 충분히 높고, 현재 검색된 문서 집합이 캐시 엔트리의
        문서 집합과 같을 때만 hit로 인정한다.
        """

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
    ) -> None:
        """LLM 응답을 Postgres semantic cache에 저장한다."""

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

