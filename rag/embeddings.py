"""BGE-M3 임베딩 — TEI 원격 서버 호출 (dense-only, F99 전환).

기존엔 인프로세스(FlagEmbedding)로 dense+sparse를 뽑았으나, F99에서
  - 임베딩을 표준 TEI 서버(worker15:38889)로 분리 (앱에서 GPU/모델 의존 제거)
  - 하이브리드 어블레이션 결과 sparse 순기여 미미 + 리랭커가 진짜 지렛대
    → dense-only 전환 (eval/report_retrieval_ablation_v1.md)
로 바꿨다. sparse·인프로세스 구현은 파일 하단에 주석 보존 (하이브리드 원복 청사진).

2026-07-17 비동기 전환: embed_texts/embed_query는 async(공용 AsyncClient)가 정식.
동기 컨텍스트(RAGAS 래퍼·EPCov·스크립트)는 *_sync 변형을 쓴다 — 운영 경로 사용 금지.
"""
from dataclasses import dataclass

import httpx

from config import settings


@dataclass
class Embedding:
    """단일 텍스트의 임베딩 결과. dense-only (sparse 제거 — 하단 주석 참고)."""
    dense: list[float]                # 1024차원 의미 벡터


async def embed_texts(texts: list[str]) -> list[Embedding]:
    """여러 텍스트를 TEI 서버로 한번에 임베딩. dense만 반환. (정식 — async 경로용)

    TEI `/embed` 응답 형식: [[float, ...], ...] (입력 순서 보존).
    공용 AsyncClient(rag.clients) 사용 — 커넥션 풀 재사용 + 이벤트 루프 비블로킹.
    """
    if not texts:
        return []
    from rag.clients import http_async   # 지연 import — 동기 스크립트가 이 모듈만 쓸 때 클라이언트 미생성
    resp = await http_async.post(
        f"{settings.embed_base_url}/embed",
        json={"inputs": texts},
        timeout=settings.embed_timeout,
    )
    resp.raise_for_status()
    return [Embedding(dense=vec) for vec in resp.json()]


async def embed_query(text: str) -> Embedding:
    """단일 query 편의 함수. (정식 — async)"""
    return (await embed_texts([text]))[0]


def embed_texts_sync(texts: list[str]) -> list[Embedding]:
    """동기 컨텍스트 전용 (RAGAS 래퍼·EPCov 채점기·일회성 스크립트).
    운영 async 경로에서 호출 금지 — 이벤트 루프를 막는다.
    """
    if not texts:
        return []
    resp = httpx.post(
        f"{settings.embed_base_url}/embed",
        json={"inputs": texts},
        timeout=settings.embed_timeout,
    )
    resp.raise_for_status()
    return [Embedding(dense=vec) for vec in resp.json()]


def embed_query_sync(text: str) -> Embedding:
    """단일 query 편의 함수 (동기 — 스크립트 전용)."""
    return embed_texts_sync([text])[0]


# =====================================================================
# [원복 청사진] 기존 인프로세스 BGE-M3 (dense + sparse) — 하이브리드 복원 시 참고.
# 복원하려면: 이 블록 활성화 + Embedding에 sparse 필드 복구 + chunks.sparse 컬럼 재생성
#            + 각 인제스트의 sparse write 복구 + retriever sparse 검색/RRF 복구 + 전체 재인제스트.
# =====================================================================
# from functools import lru_cache
# from FlagEmbedding import BGEM3FlagModel
#
# @lru_cache(maxsize=1)
# def get_model() -> BGEM3FlagModel:
#     return BGEM3FlagModel('BAAI/bge-m3', use_fp16=False)  # GPU면 use_fp16=True
#
# def embed_texts(texts: list[str]) -> list[Embedding]:
#     if not texts:
#         return []
#     output = get_model().encode(texts, return_dense=True, return_sparse=True, return_colbert_vecs=False)
#     dense_arr = output['dense_vecs']          # numpy (N, 1024)
#     sparse_list = output['lexical_weights']   # list[dict[str -> float]]
#     return [
#         Embedding(
#             dense=dense_arr[i].tolist(),
#             sparse={int(tid): float(w) for tid, w in sparse_list[i].items()},
#         )
#         for i in range(len(texts))
#     ]
