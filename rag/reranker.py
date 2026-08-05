"""리랭커 — TEI `/rerank` 로 검색 후보를 cross-encoder로 재정렬 (F99).

bi-encoder(임베딩)로 넓게 추린 후보를, query+청크를 함께 넣어 직접 관련도를 매겨
정밀하게 재정렬한다. on/off = settings.rerank_enabled (retriever가 참조).

- retrieve_candidates() 후보의 '순서만' 바꾼다 (집합·Recall@N 불변).
- 실패(서버 다운 등) 시 원 순서 유지 → graceful degrade (검색이 멈추지 않게).
- 기존 인프로세스(FlagReranker) 구현은 하단 주석 보존 (서버 없이 돌리던 실험용).

2026-07-17 비동기 전환 완료 — 공용 AsyncClient(rag.clients) 사용, 호출부는 await.
"""
import logging

from config import settings
from rag.index_text import build_index_text

logger = logging.getLogger(__name__)


def _rerank_text(chunk) -> str:
    """cross-encoder에 넣을 텍스트 — 인제스션의 임베딩 입력과 같은 형태로 맞춘다.

    본문에 없는 단어(문서 제목·상위 섹션)로 들어온 질의를 리랭커도 볼 수 있게 한다.

    FAQ 청크는 제외한다 — 본문이 'Q: 질문 / A: 답' 형태로 이미 자기설명적이고,
    heading_path가 질문 그 자체라(rag/faq_indexing) 붙이면 질문만 두 번 들어간다.
    인제스션(reindex_faq)도 FAQ 임베딩엔 prefix를 붙이지 않으므로 형태가 맞는다.
    """
    if chunk.faq_id:
        return chunk.text
    return build_index_text(chunk.text, chunk.filename, chunk.heading_path)


async def rerank(query: str, chunks: list, model_name: str | None = None) -> list:
    """query 기준 cross-encoder 점수로 chunks 재정렬. 실패 시 원본 순서 그대로.

    async(공용 AsyncClient) — 리랭크 대기 중 이벤트 루프 비블로킹.
    model_name: TEI는 컨테이너당 모델 고정이라 무시 (구 eval 호출 호환용 인자).
    """
    if not chunks:
        return chunks
    try:
        from rag.clients import http_async
        from rag.embeddings import MAX_CLIENT_BATCH   # 리랭커 TEI도 같은 상한 32 (실측 2026-08-04)

        # 상한 초과 시 422. cross-encoder는 query-text 쌍마다 독립 채점이라 나눠 불러도
        # 점수가 같다 → 전체 점수를 모아 한 번에 정렬하면 결과가 동일하다.
        scored: list[tuple[int, float]] = []
        for start in range(0, len(chunks), MAX_CLIENT_BATCH):
            batch = chunks[start:start + MAX_CLIENT_BATCH]
            resp = await http_async.post(
                f"{settings.rerank_base_url}/rerank",
                json={"query": query, "texts": [_rerank_text(c) for c in batch]},
                timeout=settings.rerank_timeout,
            )
            resp.raise_for_status()
            # TEI /rerank → [{"index": i, "score": s}, ...]. index는 **이 배치 안**의 위치라
            # start를 더해 원본 인덱스로 환산한다 (빠뜨리면 엉뚱한 청크가 상위로 올라간다).
            scored.extend((start + item["index"], item["score"]) for item in resp.json())

        ranked = sorted(scored, key=lambda x: -x[1])   # score 내림차순
        return [chunks[idx] for idx, _ in ranked]
    except Exception as e:
        logger.warning("rerank 실패 — 원 순서 유지: %s", e)
        return chunks


# =====================================================================
# [보관] 기존 인프로세스 FlagReranker (서버 없이 로컬 GPU/CPU로 돌리던 실험용).
# 서버 대신 로컬 로드로 돌리려면 이 블록 활성화 + 위 rerank 교체.
# =====================================================================
# from functools import lru_cache
# from FlagEmbedding import FlagReranker
# _DEFAULT_MODEL = 'BAAI/bge-reranker-v2-m3'
#
# @lru_cache(maxsize=2)
# def _model(name: str):
#     return FlagReranker(name, use_fp16=False)
#
# def rerank(query, chunks, model_name=_DEFAULT_MODEL):
#     if not chunks:
#         return chunks
#     scores = _model(model_name).compute_score([[query, c.text] for c in chunks])
#     if not isinstance(scores, list):
#         scores = [scores]
#     ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
#     return [c for c, _ in ranked]
