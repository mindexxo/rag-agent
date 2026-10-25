"""임베딩(dense) — 공급자를 설정으로 고른다: 사내 TEI(기본) 또는 LiteLLM 경유 상용 API.

- 차원은 1024 고정(EMBED_DIM): 상용엔 항상 dimensions=1024를 보내고, 응답 길이가 다르면 색인·검색이 조용히
  깨지므로 ValueError로 멈춘다.
- 비대칭 모델은 질의와 문서를 `kind`로 구분한다. 저장 벡터는 전부 "document"라 공급자·모델을 바꾸면 재인제스트 필수.
- `litellm`은 함수 안에서만 import한다(118MB 패키지 — TEI 경로는 안 쓴다).
- 운영 경로는 async가 정식이고, *_sync 변형은 동기 컨텍스트(RAGAS·EPCov·스크립트) 전용이다.
"""

from dataclasses import dataclass
from typing import Literal

import httpx

from config import settings

Kind = Literal["query", "document"]

# 저장 벡터 차원 — rag/os_client.DIM·rag/models.AnswerCache.query_embedding과 같아야 한다.
EMBED_DIM = 1024


@dataclass
class Embedding:
    dense: list[float]


# TEI의 요청당 입력 개수 상한(max_client_batch_size, 초과 시 422) — 나눠 보내도 서버가 다시 합쳐 배치해 손실이 없다.
MAX_CLIENT_BATCH = 32

# 상용 벤더표 — prefix는 LiteLLM 모델 접두, kind는 용도 인자 이름과 값(None이면 안 보냄).
_VENDORS: dict[str, dict] = {
    "openai": {"prefix": "openai", "kind": None},
    "gemini": {
        "prefix": "gemini",
        "kind": ("input_type", {"query": "RETRIEVAL_QUERY", "document": "RETRIEVAL_DOCUMENT"}),
    },
    "voyage": {"prefix": "voyage", "kind": ("input_type", {"query": "query", "document": "document"})},
}


def _batches(texts: list[str]) -> list[list[str]]:
    return [texts[i : i + MAX_CLIENT_BATCH] for i in range(0, len(texts), MAX_CLIENT_BATCH)]


def _payload(batch: list[str]) -> dict:
    """TEI `/embed` 요청 본문. embed_dimensions가 있으면 절단·재정규화를 서버(TEI)에 맡긴다."""
    body: dict = {"inputs": batch}
    if settings.embed_dimensions:
        body["dimensions"] = settings.embed_dimensions
    return body


def _vendor_kwargs(kind: Kind) -> dict:
    v = _VENDORS[settings.embed_provider]
    kw = {"model": f"{v['prefix']}/{settings.embed_model}", "api_key": settings.embed_api_key, "dimensions": EMBED_DIM}
    if v["kind"]:
        name, values = v["kind"]
        kw[name] = values[kind]
    return kw


def _from_vendor(resp, n: int) -> list[Embedding]:
    """LiteLLM 응답 → 입력 순서의 Embedding. 응답 index로 정렬하고 개수·차원을 검사한다."""
    rows = sorted(resp.data, key=lambda d: d["index"])
    if len(rows) != n:
        raise ValueError(f"임베딩 응답 개수 불일치: 입력 {n}, 응답 {len(rows)}")
    out = []
    for d in rows:
        vec = d["embedding"]
        if len(vec) != EMBED_DIM:
            raise ValueError(f"임베딩 차원 불일치: {len(vec)} (저장 벡터는 {EMBED_DIM}) — embed_model·dimensions 확인")
        out.append(Embedding(dense=vec))
    return out


async def embed_texts(texts: list[str], kind: Kind = "document") -> list[Embedding]:
    """MAX_CLIENT_BATCH개씩 순차 호출해 입력 순서대로 이어붙인다. kind: 질의면 "query", 색인 본문이면 "document"."""
    if not texts:
        return []
    out: list[Embedding] = []
    if settings.embed_provider == "tei":
        from rag.clients import http_async  # 지연 import — 동기 스크립트가 이 모듈만 쓸 때 클라이언트 미생성

        for batch in _batches(texts):
            resp = await http_async.post(
                f"{settings.embed_base_url}/embed",
                json=_payload(batch),
                timeout=settings.embed_timeout,
            )
            resp.raise_for_status()
            out.extend(Embedding(dense=vec) for vec in resp.json())
        return out
    import litellm

    kw = _vendor_kwargs(kind)
    for batch in _batches(texts):
        resp = await litellm.aembedding(input=batch, timeout=settings.embed_timeout, **kw)
        out.extend(_from_vendor(resp, len(batch)))
    return out


async def embed_query(text: str) -> Embedding:
    return (await embed_texts([text], kind="query"))[0]


def embed_texts_sync(texts: list[str], kind: Kind = "document") -> list[Embedding]:
    """동기 컨텍스트 전용 — 운영 async 경로에서 호출 금지(이벤트 루프를 막는다)."""
    if not texts:
        return []
    out: list[Embedding] = []
    if settings.embed_provider == "tei":
        for batch in _batches(texts):
            resp = httpx.post(
                f"{settings.embed_base_url}/embed",
                json=_payload(batch),
                timeout=settings.embed_timeout,
            )
            resp.raise_for_status()
            out.extend(Embedding(dense=vec) for vec in resp.json())
        return out
    import litellm

    kw = _vendor_kwargs(kind)
    for batch in _batches(texts):
        resp = litellm.embedding(input=batch, timeout=settings.embed_timeout, **kw)
        out.extend(_from_vendor(resp, len(batch)))
    return out


def embed_query_sync(text: str) -> Embedding:
    return embed_texts_sync([text], kind="query")[0]
