"""리랭커 — 검색 후보를 cross-encoder로 재정렬한다. 공급자는 설정으로 고른다(사내 TEI 또는 LiteLLM 경유 상용 API).

후보의 순서만 바꾼다(집합 불변). 실패하면 원 순서를 유지해 검색이 멈추지 않게 한다.
점수 눈금은 벤더마다 달라, TEI 분포로 맞춘 settings.ident_boost는 벤더를 바꾸면 다시 재야 한다.
`litellm`은 함수 안에서만 import한다(118MB 패키지 — TEI만 쓰는 프로세스가 비용을 안 내게).
"""

import asyncio
import logging

from config import settings
from rag.index_text import build_index_text

logger = logging.getLogger(__name__)

# 상용 벤더 → LiteLLM 모델 접두. 응답 index는 보낸 documents 기준이다.
_VENDORS = {"cohere": "cohere", "voyage": "voyage", "jina": "jina_ai"}


def _rerank_text(chunk) -> str:
    """cross-encoder 입력 — 인제스션의 임베딩 입력과 같은 형태(파일명·헤딩 머리줄)에 폴더 설명을 더한다.

    FAQ 청크는 본문 그대로다 — heading_path가 질문 그 자체라 붙이면 질문이 두 번 들어간다.
    """
    if chunk.faq_id:
        return chunk.text
    folder = ' — '.join(x for x in (chunk.folder_name, chunk.folder_description) if x) or None
    return build_index_text(chunk.text, chunk.filename, chunk.heading_path, folder=folder)


async def rerank_scores(query: str, chunks: list) -> list[float] | None:
    """query ↔ 각 청크의 cross-encoder 원점수(chunks와 같은 순서). 실패 시 None."""
    if not chunks:
        return []
    try:
        from rag.embeddings import (
            MAX_CLIENT_BATCH,
        )  # 리랭커 TEI도 같은 배치 상한 32(초과 시 422) — 쌍마다 독립 채점이라 나눠 불러도 점수가 같다

        scores = [0.0] * len(chunks)
        for start in range(0, len(chunks), MAX_CLIENT_BATCH):
            texts = [_rerank_text(c) for c in chunks[start : start + MAX_CLIENT_BATCH]]
            # index는 이 배치 안의 위치라 start를 더해 원본 인덱스로 환산한다.
            for i, score in await _score_batch(query, texts):
                scores[start + i] = score
        return scores
    except Exception as e:
        logger.warning("rerank 점수 실패: %s", e)
        return None


async def _score_batch(query: str, texts: list[str]) -> list[tuple[int, float]]:
    """배치 하나 채점 — 공급자 분기는 여기뿐. 반환은 (배치 안 index, 점수)."""
    if settings.rerank_provider == "tei":
        from rag.clients import http_async

        resp = await http_async.post(
            f"{settings.rerank_base_url}/rerank",
            json={"query": query, "texts": texts},
            timeout=settings.rerank_timeout,
        )
        resp.raise_for_status()
        return [(item["index"], item["score"]) for item in resp.json()]  # TEI /rerank → [{"index", "score"}]
    import litellm

    resp = await litellm.arerank(
        model=f"{_VENDORS[settings.rerank_provider]}/{settings.rerank_model}",
        query=query,
        documents=texts,
        api_key=settings.rerank_api_key,
        return_documents=False,  # 본문은 우리가 들고 있다 — 응답 크기만 키운다
        timeout=settings.rerank_timeout,
    )
    return [(r["index"], r["relevance_score"]) for r in resp.results]


async def rerank_maxpool(queries: list[str], chunks: list) -> tuple[list, list[float] | None, list[int] | None]:
    """쿼리별로 채점해 청크별 최고점(max-pool)으로 정렬 — 단일 쿼리도 이 함수다(원소 1개의 max-pool = 단일 정렬).

    (정렬된 chunks, 채택 점수, 그 점수를 낸 쿼리 인덱스) — 인덱스는 0=원본, 1..=확장 변형, 동점이면 앞 쿼리.
    실패해도 chunks는 넘겨받은 순서 그대로 돌려주고 점수·인덱스만 None이다.
    부분 성공을 쓰지 않는다 — 쿼리 하나라도 실패하면 통째로 None(성공분만 합치면 점수가 서로 비교 불가능해진다).
    """
    matrix = await asyncio.gather(*(rerank_scores(q, chunks) for q in queries))
    if not all(s is not None for s in matrix):
        return chunks, None, None
    per_chunk = list(zip(*matrix))  # [청크][쿼리] 점수
    best_qi = [max(range(len(col)), key=col.__getitem__) for col in per_chunk]
    best = [col[qi] for col, qi in zip(per_chunk, best_qi)]  # == max(col) — 값 불변
    order = sorted(range(len(chunks)), key=lambda i: -best[i])
    return [chunks[i] for i in order], [best[i] for i in order], [best_qi[i] for i in order]


async def rerank(query: str, chunks: list) -> list:
    """단일 쿼리 재정렬 — rerank_maxpool([query], ...)의 위임 래퍼(eval 호출부용)."""
    reranked, _, _ = await rerank_maxpool([query], chunks)
    return reranked
