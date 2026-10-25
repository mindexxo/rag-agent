"""임베딩 — 공급자 하나를 설정으로 고른다(#289): 사내 TEI(BGE-M3, 기본) 또는 상용 API(LiteLLM 경유).

F99에서 인프로세스(FlagEmbedding) → TEI 원격 서버로, 하이브리드 어블레이션 결과 sparse 순기여가
미미해 dense-only로 전환했다(근거 eval/report_retrieval_ablation_v1.md). #289는 그 위에 상용 벤더를
얹었다 — 함수 계약(embed_texts/embed_query)은 그대로고 안에서 settings.embed_provider로 갈린다.

- **차원은 1024 고정(EMBED_DIM)**: 저장 벡터(OpenSearch 매핑 rag/os_client.DIM·의미 캐시 Vector(1024))가
  이 값이라, 상용엔 항상 dimensions=1024를 보내고 응답 길이를 검사한다 — 다른 길이면 색인·검색이
  조용히 깨지므로 ValueError로 멈춘다(#289 결정 3). 벤더 기본 차원(OpenAI 3072·Gemini 3072·Voyage 1024)에
  기대지 않는다.
- **용도(kind) 전달**: 비대칭 모델(Gemini RETRIEVAL_QUERY/DOCUMENT·Voyage query/document)은 질의와 문서를
  다른 인자로 받는다. 호출부가 `kind=`로 알려 주고(embed_query는 "query"), TEI(BGE-M3 대칭)·OpenAI는
  무시한다(#289 결정 4). 저장 벡터는 전부 "document"로 만든 것이라, 공급자·모델을 바꾸면 **재인제스트 필수**.
- 배치 상한 32(MAX_CLIENT_BATCH)는 공급자 무관하게 같다 — 상용 입력 상한(OpenAI 2048·Gemini 100·Voyage 1000)
  안에 들고, 나눠 보내도 결과가 같다(입력별 독립 연산).
- `litellm`은 함수 안에서만 import한다(118MB 패키지 — rag/llm/litellm.py와 같은 규율, TEI 경로는 안 쓴다).

2026-07-17 비동기 전환: embed_texts/embed_query는 async(공용 AsyncClient)가 정식.
동기 컨텍스트(RAGAS 래퍼·EPCov·스크립트)는 *_sync 변형을 쓴다 — 운영 경로 사용 금지.
"""

from dataclasses import dataclass
from typing import Literal

import httpx

from config import settings

Kind = Literal["query", "document"]

# 저장 벡터 차원 — rag/os_client.DIM·rag/models.AnswerCache.query_embedding과 같아야 한다(tests/test_embed_rerank_providers.py가 고정).
EMBED_DIM = 1024


@dataclass
class Embedding:
    """단일 텍스트의 임베딩 결과. dense-only."""

    dense: list[float]  # 1024차원 의미 벡터


# TEI가 요청 하나에 받는 입력 개수 상한 (`/info`의 max_client_batch_size, 실측 2026-08-04).
# 초과하면 422. GPU 용량이 아니라 큐 독점 방지용 상한이라, 나눠 보내도 서버가
# max_batch_tokens(16384) 안에서 다시 합쳐 배치하므로 처리량 손실이 없다.
# 서버 설정을 올리는 대신 클라이언트에서 쪼갠다 — 문서가 커지면 어떤 상한이든 다시 넘기 때문.
MAX_CLIENT_BATCH = 32

# 상용 벤더표 (#289) — prefix는 LiteLLM 모델 접두, kind는 용도 인자 이름과 값(없으면 안 보냄).
# 검증(로컬 스텁 캡처 2026-10-24, litellm 1.95.0): openai→`dimensions`, gemini→batchEmbedContents의
# `outputDimensionality`+`taskType`(input_type 인자에서 변환), voyage→`output_dimension`+`input_type` 그대로.
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
    """TEI `/embed` 요청 본문. settings.embed_dimensions가 있으면 MRL 절단을 서버에 맡긴다.
    (TEI 1.9.3 실측: `dimensions` 수용 + 절단 후 L2 재정규화까지 서버가 처리 — 클라이언트 절단 불필요.)
    """
    body: dict = {"inputs": batch}
    if settings.embed_dimensions:
        body["dimensions"] = settings.embed_dimensions
    return body


def _vendor_kwargs(kind: Kind) -> dict:
    """상용 호출 공통 인자 — 모델 접두·키·차원 고정·용도. 모르는 공급자는 config Literal이 기동 시 막는다."""
    v = _VENDORS[settings.embed_provider]
    kw = {"model": f"{v['prefix']}/{settings.embed_model}", "api_key": settings.embed_api_key, "dimensions": EMBED_DIM}
    if v["kind"]:
        name, values = v["kind"]
        kw[name] = values[kind]
    return kw


def _from_vendor(resp, n: int) -> list[Embedding]:
    """LiteLLM EmbeddingResponse → 입력 순서의 Embedding. 응답 index로 정렬하고 길이·차원을 검사한다."""
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
    """여러 텍스트를 임베딩. dense만 반환. (정식 — async 경로용)

    kind: 질의면 "query"(검색·캐시), 색인 본문이면 "document"(기본). 비대칭 벤더만 구분한다(모듈 docstring).
    MAX_CLIENT_BATCH개씩 나눠 순차 호출하고 결과를 입력 순서대로 이어붙인다 —
    입력별 독립 연산이라 한 번에 보낸 것과 결과가 동일하다.
    TEI 경로: `/embed` 응답 [[float, ...], ...](입력 순서 보존), 공용 AsyncClient(rag.clients). 상용 경로: litellm.aembedding → _from_vendor.
    """
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
    """단일 query 편의 함수. (정식 — async) — kind="query"."""
    return (await embed_texts([text], kind="query"))[0]


def embed_texts_sync(texts: list[str], kind: Kind = "document") -> list[Embedding]:
    """동기 컨텍스트 전용 (RAGAS 래퍼·EPCov 채점기·일회성 스크립트). async 경로와 같은 분기·분할.
    운영 async 경로에서 호출 금지 — 이벤트 루프를 막는다.
    """
    if not texts:
        return []
    out: list[Embedding] = []
    if settings.embed_provider == "tei":
        for batch in _batches(texts):  # async 경로와 같은 이유로 분할 (MAX_CLIENT_BATCH 주석 참조)
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
    """단일 query 편의 함수 (동기 — 스크립트 전용) — kind="query"."""
    return embed_texts_sync([text], kind="query")[0]
