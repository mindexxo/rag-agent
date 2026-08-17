"""리랭커 — TEI `/rerank` 로 검색 후보를 cross-encoder로 재정렬 (F99).

bi-encoder(임베딩)로 넓게 추린 후보를, query+청크를 함께 넣어 직접 관련도를 매겨
정밀하게 재정렬한다. on/off = settings.rerank_enabled (retriever가 참조).

- retrieve_candidates() 후보의 '순서만' 바꾼다 (집합·Recall@N 불변).
- 실패(서버 다운 등) 시 원 순서 유지 → graceful degrade (검색이 멈추지 않게).

정렬은 rerank_maxpool() 하나다 — 단일 쿼리는 원소 1개짜리 쿼리 리스트의 max-pool과 수학적으로
동일해서(#54, 골든 140케이스 대조) retriever는 항상 rerank_maxpool을 부른다. rerank()는
그 사실을 모르는 기존 호출부(eval 2곳)를 위한 위임 래퍼다.
otel 계측은 호출부(retriever)가 맡는다: 이 모듈은 관측 의존이 없다.

2026-07-17 비동기 전환 완료 — 공용 AsyncClient(rag.clients) 사용, 호출부는 await.
"""
import asyncio
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
    folder = ' — '.join(x for x in (chunk.folder_name, chunk.folder_description) if x) or None
    return build_index_text(chunk.text, chunk.filename, chunk.heading_path, folder=folder)


async def rerank_scores(query: str, chunks: list) -> list[float] | None:
    """query ↔ 각 청크의 cross-encoder 원점수 (chunks와 같은 순서). 실패 시 None.

    쿼리 확장 max-pool(#5)이 쿼리별 점수를 직접 합산해야 해서 순서 대신 점수를 노출.
    rerank()도 이 점수로 정렬한다 — 채점 경로는 하나.
    """
    if not chunks:
        return []
    try:
        from rag.clients import http_async
        from rag.embeddings import MAX_CLIENT_BATCH   # 리랭커 TEI도 같은 상한 32 (실측 2026-08-04)

        # 상한 초과 시 422. cross-encoder는 query-text 쌍마다 독립 채점이라 나눠 불러도
        # 점수가 같다 → 전체 점수를 모아 한 번에 정렬하면 결과가 동일하다.
        scores = [0.0] * len(chunks)
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
            for item in resp.json():
                scores[start + item["index"]] = item["score"]
        return scores
    except Exception as e:
        logger.warning("rerank 점수 실패: %s", e)
        return None


async def rerank_maxpool(queries: list[str], chunks: list) -> tuple[list, list[float] | None]:
    """쿼리 확장(#5) 채점 — 쿼리별로 각자 채점해 청크별 최고점(max-pool)으로 정렬.

    (chunks, 그 순서에 대응하는 채택 점수) 반환. **실패해도 chunks는 항상 돌려준다** —
    점수 자리만 None이 되고 순서는 넘겨받은 그대로(RRF)다. rerank()가 실패 시 원 순서를
    돌려주는 것과 같은 모양이라, 호출부가 실패 분기에서 리스트를 따로 챙길 필요가 없다.

    RRF→원본 쿼리 채점 방식은 변형이 찾아온 청크를 원본 어휘로 다시 채점해 이득이
    소멸했다 (mt 90문항 실측: RRF+원본채점 = 풀확장+원본채점 = 개선 0, max-pool +4.5pp).

    **부분 성공을 쓰지 않는다** — 쿼리 하나라도 실패하면 통째로 None이다. 성공분만
    합치면 청크마다 max를 취한 쿼리 수가 달라져 점수가 서로 비교 불가능해지고,
    어느 쿼리가 실패했느냐에 따라 순서가 달라져 재현성이 깨진다.
    """
    matrix = await asyncio.gather(*(rerank_scores(q, chunks) for q in queries))
    if not all(s is not None for s in matrix):
        return chunks, None
    best = [max(col) for col in zip(*matrix)]
    order = sorted(range(len(chunks)), key=lambda i: -best[i])
    return [chunks[i] for i in order], [best[i] for i in order]


async def rerank(query: str, chunks: list, model_name: str | None = None) -> list:
    """단일 쿼리 재정렬 — rerank_maxpool([query], ...)의 위임 래퍼 (#54). 실패 시 원본 순서.

    쿼리 1개의 max-pool은 그 쿼리 점수 정렬과 동일하다(원소 1개 max = 그 값, 같은 sorted 호출).
    model_name: TEI는 컨테이너당 모델 고정이라 무시 (구 eval 호출 호환용 인자).
    """
    reranked, _ = await rerank_maxpool([query], chunks)
    return reranked
