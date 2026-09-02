"""RAG 답변 캐시 — Postgres semantic 단층. 상태 없는 모듈 함수 모음.

조회 = standalone query 임베딩의 pgvector 최근접 1건 → floor(후보 게이트) → 근거 문서
집합 비교 → LLM 재사용 판정(_verify_reuse). **판정 승인 없이는 서빙이 없다** (#113 —
자동 서빙 임계와 규칙 기반 기계 가드는 제거됐다: 판정기가 전 negative를 잡는 것이 실측된
순간 규칙 사전(부정 마커·시간 어휘)은 유지보수 부채이고, 실제 오차단도 냈다).
캐시 키는 질의 재작성된 standalone query 기준. (과거의 Redis exact 계층은 제거됨)

공개 표면: CacheHit · snapshot_faq_versions · get_semantic · save_answer ·
invalidate_source · sweep_stale. 나머지는 내부 헬퍼(_).
(과거엔 상태 없는 AnswerCache 클래스였고 호출부마다 인스턴스를 새로 만들었다 —
 #49에서 모듈 함수로 해체. REVIEW_FINDINGS #12의 "요청마다 생성" 지적은 인스턴스라는
 개념이 사라지며 함께 해소. rag.models의 동명 모델과 헷갈리던 AnswerCacheRow 별칭
 관행도 이제 이 파일 안의 문제일 뿐이다.)

정확성 보장 지도 (#16 분석 — 어떤 변경이 어떤 메커니즘으로 잡히는가):
- 문서 재업로드/삭제/비활성: document_id가 바뀌거나 검색에서 빠짐 → 조회 시 doc집합
  비교(get_semantic)가 자가치유. 명시 무효화(invalidate_source)는 죽은 row 청소용.
- FAQ 수정: faq_id 불변이라 자가치유 불가 → routers/faqs.py의 명시 무효화 +
  생성 중(in-flight) write-back 레이스는 save_answer의 updated_at 낙관적 검증이 차단.
- 모델 교체: 미보호 — 교체 시 일괄 flush로 처리한다 (조회 model 필터는 기각 #113:
  답을 바꾸는 미보호 축이 여럿이라 그 하나만 막는 건 착시).
"""
import logging
from dataclasses import dataclass
from datetime import datetime

from hashlib import sha256
from sqlalchemy import select, update, func, delete
from sqlalchemy.ext.asyncio import AsyncSession

from config import settings
from rag.embeddings import embed_query
from rag.llm_schemas import acomplete_validated, ReuseJudgment
from rag.prompts import build_cache_reuse_judge_messages
from schemas.kms import SourceCitation
from rag.models import AnswerCache as AnswerCacheRow, Faq
from sqlalchemy.dialects.postgresql import insert as pg_insert

logger = logging.getLogger(__name__)


@dataclass
class CacheHit:
    answer: str
    sources: list[SourceCitation]
    source_doc_ids: list[int]


async def _verify_reuse(llm, cached_query: str, new_query: str) -> bool:
    """판정기(#113) — 캐시 후보의 재사용 가부를 LLM에 묻는다. 모든 히트의 필요조건이다.

    True=재사용 승인. 판정 실패(스키마 불신·호출 실패)는 False — 오판 비용 비대칭
    (잘못 승인=오답 재생, 잘못 거절=생성 1회)이라 모든 불확실성은 거절로 수렴시킨다.
    프롬프트·검증 실측은 rag/prompt_texts.py의 CACHE_REUSE_JUDGE_* 참조 (40쌍 39/40·
    위험 방향 0·3회 반복 흔들림 0 — 그래서 운영은 1콜).
    """
    try:
        judgment = await acomplete_validated(
            llm, build_cache_reuse_judge_messages(cached_query, new_query), ReuseJudgment)
        logger.info('캐시 재사용 판정: %s — %s', judgment.same_answer, judgment.reason[:120])
        return judgment.same_answer
    except Exception:
        logger.exception('캐시 재사용 판정 실패 — 거절로 처리')
        return False


def _normalize_query(query: str) -> str:
    """캐시 키 생성을 위해 질의를 정규화한다."""
    normalized = ' '.join(query.strip().split())
    return normalized.rstrip("?.!。？！").lower()


def _build_cache_digest(standalone_query: str) -> str:
    """정규화된 standalone_query의 sha256 digest를 만든다."""
    return sha256(_normalize_query(standalone_query).encode("utf-8")).hexdigest()


def _sources_to_json(sources: list[SourceCitation]) -> list[dict]:
    """SourceCitation 리스트를 JSON 저장 가능한 dict 리스트로 변환한다."""
    return [source.model_dump() for source in sources]


def _sources_from_json(raw_sources: list[dict]) -> list[SourceCitation]:
    """JSON dict 리스트를 SourceCitation 리스트로 복원한다."""
    return [SourceCitation(**source) for source in raw_sources]


async def snapshot_faq_versions(
        session: AsyncSession, tenant_id: str, source_doc_ids: list[int],
) -> dict[int, datetime]:
    """근거에 포함된 FAQ들의 {faq_id: updated_at} 스냅샷 — save_answer의 낙관적 검증 기준값.

    prepare(검색 직후) 시점에 찍어두고, 생성이 끝난 save_answer 시점에 재조회와 등치 비교한다.
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


async def get_semantic(
        session: AsyncSession, tenant_id: str, query: str,
        current_source_doc_ids: list[int],
        query_embedding: list[float] | None = None,
        llm=None,
) -> CacheHit | None:
    """Postgres semantic cache를 조회한다.

    hit 조건(#113, 순서대로): 최근접 후보의 유사도 ≥ floor(후보 게이트) → 검색 doc집합
    동일 → **LLM 재사용 판정 승인**. 자동 서빙 임계는 없다 — 실측에서 유사도 0.96~0.99
    쌍조차 답이 반대인 경우가 4건 나와, 유사도 단독으로는 어떤 값에서도 서빙을 정당화할
    수 없었다. 판정은 모든 히트의 필요조건이고, 그래서 **llm이 None이면 항상 miss**다
    (판정 없는 서빙 경로 자체가 없다).

    query_embedding: 검색이 이미 만든 원본 쿼리 벡터(#50). 있으면 그대로 쓴다 —
    검색·캐시 조회·캐시 저장이 같은 문자열을 각자 임베딩해 TEI를 3번 때리던 것을 1번으로
    줄인다. 재사용이 안전한 근거: 벡터의 출처가 물리적으로 같다(embed_query는
    embed_texts([text])[0] 래퍼고, 쿼리에는 색인 때 붙는 '파일명>헤딩' 프리픽스가 없다).
    게다가 TEI는 호출마다 비결정적(실측 1.4e-4)이라 "매번 새로 임베딩"해도 애초에 같은
    벡터가 아니었다 — 재사용본도 그 노이즈 폭 안이고, floor 게이트 판정을 흔들
    수준이 아니다. **이 rationale의 단일 정의점이 여기다** — retriever·service는 참조만 한다.

    None이면 예전처럼 embed_query(query)로 직접 만든다. 벡터를 손에 들지 않은 호출부
    (테스트, eval/cache_eval.py)를 위한 폴백이다 — 운영 경로는 항상 벡터를 넘긴다.

    fail-open(#16): 캐시는 어떤 경우에도 요청을 죽이면 안 된다 — 조회 실패는 miss 취급.
    (실측 실패 모드: 임베딩 TEI 블립. 검색까지 성공한 요청이 캐시 조회에서 500 나던 문제.
    #50 이후 운영 경로는 여기서 임베딩을 안 하므로 남는 실패 모드는 DB 쪽이다.)
    try가 본문 전체를 감싼다 — reranker.rerank_scores의 graceful degrade와 같은 모양.
    (구조상 public/_private 2겹이었던 것을 단일 함수로 — 내부판을 직접 부르는 곳이 없었다)
    """
    try:
        # 1. 질의 벡터 확보 — 검색이 넘겨줬으면 재사용, 없으면 직접 임베딩 (#50).
        if query_embedding is None:
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

        # 5. 후보 게이트 — floor 미만은 판정 비용을 쓸 가치가 없는 잡음이다.
        # llm이 없으면 판정이 불가능하므로 서빙도 없다 (판정 없는 히트 경로는 없다 #113).
        if llm is None or similarity < settings.semantic_cache_floor:
            return None

        # 6. 질문 의미가 비슷해도, 현재 검색 결과의 근거 문서 집합이 다르면 miss 처리한다.
        # 비슷하지만 다른 정책/문서에 답해야 하는 케이스에서 오답 재사용을 막기 위함이다.
        # 판정보다 먼저 — LLM 콜 없이 걸러지는 후보에 콜을 쓰지 않는다.
        if set(cache_row.source_doc_ids) != set(current_source_doc_ids):
            return None

        # 7. 재사용 판정(#113) — 모든 히트의 필요조건. 판정 실패(예외 포함)는 miss.
        if not await _verify_reuse(llm, cache_row.query_text, query):
            logger.info('판정기가 캐시 재사용 거절 — miss (tenant=%s, sim=%.4f)',
                        tenant_id, similarity)
            return None

        # 8. semantic hit 통계 갱신.
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
) -> None:
    """LLM 응답을 Postgres semantic cache에 저장한다.

    (구 이름 set — 모듈 함수가 되며 개명. 최상위 `def set`은 내장 set()을 가려
    get_semantic의 doc집합 비교 `set(a) != set(b)`를 조용히 깨뜨린다.)

    faq_versions: prepare 시점의 {faq_id: updated_at} 스냅샷(snapshot_faq_versions).
    None이면 검증 생략(FAQ 근거가 없거나 테스트 등 직접 호출).

    query_embedding: get_semantic과 같은 계약 — 있으면 재사용, None이면 embed_query 폴백.
    사유·안전성 근거는 get_semantic docstring 참조 (#50).

    fail-open(#16): 저장 실패는 삼키고 로그만 — 이미 완성돼 클라이언트로 나간 답변이
    캐시 저장 실패 때문에 failed로 기록되던 문제 방지. 단 DB 오류로 세션이 오염된
    경우는 이후 커밋이 어차피 실패하므로 여기서 감추지 않는 편이 낫지만, 주 실패
    모드(임베딩 TEI 블립)는 DB 접근 전이라 세션이 깨끗하다.
    (#50 이후 운영 경로는 벡터를 받아 오므로 그 실패 모드 자체가 여기서 사라진다.)
    """
    try:
        # 낙관적 검증(#16): 생성하는 동안 근거 FAQ가 수정/삭제/비활성됐으면 저장을 스킵.
        # FAQ PATCH의 무효화 커밋 '이후'에 옛 내용 기반 답변을 다시 심는 레이스 차단.
        # FOR SHARE로 재조회 — 검증과 INSERT 커밋 사이의 ms 창까지 닫는다: 동시 FAQ UPDATE는
        # 이 트랜잭션 커밋까지 대기하고, 대기 후 실행되는 무효화가 방금 쓴 row를 지운다.
        if faq_versions is not None and not await _faqs_unchanged(
                session, tenant_id, source_doc_ids, faq_versions):
            logger.info('근거 FAQ가 생성 중 변경됨 — 캐시 저장 스킵 (tenant=%s)', tenant_id)
            return

        # semantic cache용으로 query embedding을 저장한다.
        # 이후 비슷한 질문이 들어오면 pgvector cosine distance로 이 row를 찾는다.
        # 검색이 넘겨줬으면 재사용, 없으면 직접 임베딩 (#50).
        if query_embedding is None:
            query_embedding = (await embed_query(query)).dense

        # cache_key = 정규화 query의 digest — UNIQUE(tenant_id, cache_key)의 충돌 판정 키.
        stmt = pg_insert(AnswerCacheRow).values(
            tenant_id=tenant_id,
            cache_key=_build_cache_digest(query),
            query_text=query,
            query_embedding=query_embedding,
            answer=answer,
            sources=_sources_to_json(sources),
            source_doc_ids=source_doc_ids,
            model=settings.vllm_model,
        )

        # 같은 tenant + cache_key가 이미 있으면 UPDATE — 동시 요청이 같은 질의를 쓰거나,
        # 무효화 후 재생성된 경우 unique 제약 오류 대신 최신 답변으로 갱신한다.
        # hit_count=0 리셋은 의도다: 답변이 교체됐으니 옛 답의 적중 통계를 물려받지 않는다.
        stmt = stmt.on_conflict_do_update(
            index_elements=["tenant_id", "cache_key"],
            set_={
                "query_text": query,
                "query_embedding": query_embedding,
                "answer": answer,
                "sources": _sources_to_json(sources),
                "source_doc_ids": source_doc_ids,
                "model": settings.vllm_model,
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


async def sweep_stale(session: AsyncSession) -> int:
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


async def invalidate_source(
        session: AsyncSession,
        tenant_id: str,
        source_id: int,
) -> None:
    """source_id를 근거로 만든 semantic 캐시 row를 무효화한다.

    source_id는 service._source_doc_ids와 같은 네임스페이스다 — **양수 = document_id,
    음수(-faq_id) = FAQ.** 문서 재업로드/삭제/비활성화와 FAQ 수정/삭제(routers/faqs.py)가
    전부 이 함수를 탄다. (구 이름 invalidate_document — FAQ도 타는데 이름이 반쪽 진실이었다)

    semantic은 조회 시 doc집합 비교로도 자가치유하지만, 여기서 명시 삭제해
    재업로드/삭제 후 죽은 row가 쌓이지 않게 한다.
    """
    await session.execute(
        delete(AnswerCacheRow)
        .where(AnswerCacheRow.tenant_id == tenant_id)
        .where(AnswerCacheRow.source_doc_ids.any(source_id))
    )
