"""임베딩·리랭크 공급자 분기 계약 (#289) — 키·네트워크 없이, TEI는 HTTP를·상용은 litellm을 가짜로 바꿔 요청과 해석을 고정한다.

고정하는 것(#289 결정):
1. 함수 계약 불변(결정 2): embed_texts/embed_query/rerank_scores 시그니처 그대로, 분기는 모듈 안. TEI 요청 본문은 kind와 무관하게 전과 같다.
2. 차원 1024 고정(결정 3): 상용엔 dimensions=1024를 보내고, 응답 벡터가 그 길이가 아니면 ValueError — 저장 벡터(os_client.DIM)와 같은 값.
3. 용도 전달(결정 4): gemini는 RETRIEVAL_QUERY/RETRIEVAL_DOCUMENT, voyage는 query/document, openai는 안 보냄. embed_query는 "query".
4. 배치 32 분할·응답 index 정렬·개수 검사는 상용에도 같다. 리랭크는 배치 안 index를 원본 위치로 환산하고 실패 시 None(graceful degrade).
5. 설정: 상용 공급자인데 모델명이 비면 기동 시 ValueError.
실제 벤더 요청 모양은 로컬 스텁 캡처로 확인했다 — 품질·눈금은 라이브 측정 몫.
"""
from types import SimpleNamespace

import pytest

import rag.search.embeddings as emb
import rag.search.reranker as rr
from config import settings
from rag.search.embeddings import EMBED_DIM, MAX_CLIENT_BATCH, Embedding, embed_query, embed_texts, embed_texts_sync
from rag.search.reranker import rerank_scores


def _vec(i, n=EMBED_DIM):
    return [float(i)] * n


class _Litellm:
    """litellm 모듈 대역 — 호출 인자를 기록하고 정해진 응답을 돌려준다."""

    def __init__(self, dim=EMBED_DIM, shuffle=False, fail=False):
        self.calls, self.dim, self.shuffle, self.fail = [], dim, shuffle, fail

    def _embed_resp(self, input):
        rows = [{"index": i, "embedding": _vec(hash(t) % 97, self.dim)} for i, t in enumerate(input)]
        return SimpleNamespace(data=rows[::-1] if self.shuffle else rows)

    async def aembedding(self, **kw):
        self.calls.append(kw)
        return self._embed_resp(kw["input"])

    def embedding(self, **kw):
        self.calls.append(kw)
        return self._embed_resp(kw["input"])

    async def arerank(self, **kw):
        self.calls.append(kw)
        if self.fail:
            raise RuntimeError("429")
        # 배치 안 index 기준, 뒤 문서일수록 점수 높게 — 환산을 안 하면 순서가 틀어진다
        return SimpleNamespace(results=[{"index": i, "relevance_score": 0.1 * (i + 1)} for i in range(len(kw["documents"]))][::-1])


@pytest.fixture
def fake_litellm(monkeypatch):
    import sys
    fake = _Litellm()
    monkeypatch.setitem(sys.modules, "litellm", fake)
    return fake


@pytest.fixture
def vendor(monkeypatch):
    def _set(embed=None, rerank=None):
        if embed:
            monkeypatch.setattr(settings, "embed_provider", embed)
            monkeypatch.setattr(settings, "embed_model", "m-embed")
            monkeypatch.setattr(settings, "embed_api_key", "k-embed")
        if rerank:
            monkeypatch.setattr(settings, "rerank_provider", rerank)
            monkeypatch.setattr(settings, "rerank_model", "m-rerank")
            monkeypatch.setattr(settings, "rerank_api_key", "k-rerank")
    return _set


def test_dim_matches_storage():
    """상용에 요청하는 차원 = OpenSearch 매핑 차원. 어긋나면 색인이 조용히 실패한다."""
    from rag.search.os_client import DIM
    assert EMBED_DIM == DIM == 1024


class TestTeiUnchanged:
    """기본 공급자(tei) 경로는 #289 전과 같은 요청 본문 — kind는 무시."""

    @pytest.mark.anyio
    async def test_async_body_and_kind_ignored(self, monkeypatch):
        import rag.clients
        sent = []

        async def _post(url, json, timeout):
            sent.append((url, json))
            return SimpleNamespace(raise_for_status=lambda: None, json=lambda: [_vec(1)] * len(json["inputs"]))

        monkeypatch.setattr(rag.clients.http_async, "post", _post)
        assert settings.embed_provider == "tei"
        out = await embed_texts(["a", "b"], kind="query")
        assert [e.dense for e in out] == [_vec(1)] * 2
        assert sent == [(f"{settings.embed_base_url}/embed", {"inputs": ["a", "b"]})]   # kind 흔적 없음

    def test_sync_body(self, monkeypatch):
        import httpx
        sent = []

        def _post(url, json, timeout):
            sent.append(json)
            return SimpleNamespace(raise_for_status=lambda: None, json=lambda: [_vec(2)] * len(json["inputs"]))

        monkeypatch.setattr(httpx, "post", _post)
        assert embed_texts_sync(["x"], kind="query")[0].dense == _vec(2)
        assert sent == [{"inputs": ["x"]}]

    @pytest.mark.anyio
    async def test_async_batch_split_and_order(self, monkeypatch):
        monkeypatch.setattr(settings, "embed_provider", "tei")
        import rag.clients
        sent = []

        async def _post(url, json, timeout):
            sent.append(json["inputs"])
            return SimpleNamespace(raise_for_status=lambda: None, json=lambda: [_vec(int(t)) for t in json["inputs"]])

        monkeypatch.setattr(rag.clients.http_async, "post", _post)
        texts = [str(i) for i in range(MAX_CLIENT_BATCH + 1)]
        out = await embed_texts(texts)
        assert [len(b) for b in sent] == [MAX_CLIENT_BATCH, 1]
        assert [e.dense for e in out] == [_vec(i) for i in range(MAX_CLIENT_BATCH + 1)]

    def test_sync_batch_split_and_order(self, monkeypatch):
        monkeypatch.setattr(settings, "embed_provider", "tei")
        import httpx
        sent = []

        def _post(url, json, timeout):
            sent.append(json["inputs"])
            return SimpleNamespace(raise_for_status=lambda: None, json=lambda: [_vec(int(t)) for t in json["inputs"]])

        monkeypatch.setattr(httpx, "post", _post)
        out = embed_texts_sync([str(i) for i in range(MAX_CLIENT_BATCH + 1)])
        assert [len(b) for b in sent] == [MAX_CLIENT_BATCH, 1]
        assert [e.dense for e in out] == [_vec(i) for i in range(MAX_CLIENT_BATCH + 1)]

    @pytest.mark.anyio
    async def test_rerank_body_and_index_offset(self, monkeypatch):
        import rag.clients
        sent = []

        async def _post(url, json, timeout):
            sent.append(json)
            n = len(json["texts"])
            return SimpleNamespace(raise_for_status=lambda: None, json=lambda: [{"index": i, "score": 0.5} for i in range(n)])

        monkeypatch.setattr(rag.clients.http_async, "post", _post)
        chunks = [SimpleNamespace(faq_id=1, text=f"t{i}") for i in range(MAX_CLIENT_BATCH + 1)]
        scores = await rerank_scores("q", chunks)
        assert scores == [0.5] * (MAX_CLIENT_BATCH + 1)
        assert [len(b["texts"]) for b in sent] == [MAX_CLIENT_BATCH, 1] and sent[0]["query"] == "q"


class TestVendorEmbedding:
    @pytest.mark.anyio
    @pytest.mark.parametrize("provider,prefix,kind_arg", [
        ("openai", "openai", None),
        ("gemini", "gemini", ("input_type", "RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY")),
        ("voyage", "voyage", ("input_type", "document", "query")),
    ])
    async def test_request_shape_per_vendor(self, fake_litellm, vendor, provider, prefix, kind_arg):
        vendor(embed=provider)
        await embed_texts(["a"])                 # 기본 kind = document
        await embed_query("q")                   # 편의 함수 = query
        doc, qry = fake_litellm.calls
        for call in (doc, qry):
            assert call["model"] == f"{prefix}/m-embed" and call["api_key"] == "k-embed"
            assert call["dimensions"] == EMBED_DIM and call["timeout"] == settings.embed_timeout
        if kind_arg is None:
            assert "input_type" not in doc and "input_type" not in qry
        else:
            name, d, q = kind_arg
            assert doc[name] == d and qry[name] == q
        assert doc["input"] == ["a"] and qry["input"] == ["q"]

    @pytest.mark.anyio
    async def test_batch_split_and_order(self, fake_litellm, vendor):
        """33개 → 32+1 두 호출, 응답이 index 역순으로 와도 입력 순서로 복원."""
        vendor(embed="openai")
        fake_litellm.shuffle = True
        texts = [f"t{i}" for i in range(MAX_CLIENT_BATCH + 1)]
        out = await embed_texts(texts)
        assert [len(c["input"]) for c in fake_litellm.calls] == [MAX_CLIENT_BATCH, 1]
        assert [e.dense[0] for e in out] == [float(hash(t) % 97) for t in texts]

    @pytest.mark.anyio
    async def test_wrong_dimension_raises(self, fake_litellm, vendor):
        vendor(embed="gemini")
        fake_litellm.dim = 3072                  # 벤더 기본 차원 — dimensions가 안 먹은 상황
        with pytest.raises(ValueError, match="차원 불일치"):
            await embed_texts(["a"])

    @pytest.mark.anyio
    async def test_count_mismatch_raises(self, fake_litellm, vendor):
        vendor(embed="openai")

        async def _short(**kw):
            return SimpleNamespace(data=[{"index": 0, "embedding": _vec(1)}])
        fake_litellm.aembedding = _short
        with pytest.raises(ValueError, match="개수 불일치"):
            await embed_texts(["a", "b"])

    def test_sync_wrong_dimension_raises(self, fake_litellm, vendor):
        vendor(embed="openai")
        fake_litellm.dim = 3072
        with pytest.raises(ValueError, match="차원 불일치"):
            embed_texts_sync(["a"])

    def test_sync_uses_same_branch(self, fake_litellm, vendor):
        vendor(embed="voyage")
        out = embed_texts_sync(["a", "b"], kind="query")
        assert len(out) == 2 and isinstance(out[0], Embedding)
        assert fake_litellm.calls[0]["input_type"] == "query" and fake_litellm.calls[0]["model"] == "voyage/m-embed"

    @pytest.mark.anyio
    async def test_empty_input_no_call(self, fake_litellm, vendor):
        vendor(embed="openai")
        assert await embed_texts([]) == [] and fake_litellm.calls == []


class TestVendorRerank:
    @pytest.mark.anyio
    @pytest.mark.parametrize("provider,prefix", [("cohere", "cohere"), ("voyage", "voyage"), ("jina", "jina_ai")])
    async def test_request_and_index_offset(self, fake_litellm, vendor, provider, prefix):
        vendor(rerank=provider)
        chunks = [SimpleNamespace(faq_id=1, text=f"t{i}") for i in range(MAX_CLIENT_BATCH + 2)]
        scores = await rerank_scores("q", chunks)
        # 각 배치 안 i번째 문서 점수 = 0.1*(i+1): 원본 위치로 환산됐으면 둘째 배치(2개)는 0.1, 0.2
        assert scores[:MAX_CLIENT_BATCH] == pytest.approx([0.1 * (i + 1) for i in range(MAX_CLIENT_BATCH)])
        assert scores[MAX_CLIENT_BATCH:] == pytest.approx([0.1, 0.2])
        c0, c1 = fake_litellm.calls
        assert c0["model"] == f"{prefix}/m-rerank" and c0["api_key"] == "k-rerank" and c0["query"] == "q"
        assert c0["return_documents"] is False and c0["timeout"] == settings.rerank_timeout
        assert c0["documents"] == [f"t{i}" for i in range(MAX_CLIENT_BATCH)] and c1["documents"] == [f"t{MAX_CLIENT_BATCH}", f"t{MAX_CLIENT_BATCH + 1}"]

    @pytest.mark.anyio
    async def test_failure_degrades_to_none(self, fake_litellm, vendor):
        vendor(rerank="cohere")
        fake_litellm.fail = True
        assert await rerank_scores("q", [SimpleNamespace(faq_id=1, text="t")]) is None


class TestSettings:
    @pytest.mark.parametrize("field", ["embed", "rerank"])
    def test_vendor_requires_model(self, monkeypatch, field):
        from config import Settings
        monkeypatch.delenv(f"{field.upper()}_MODEL", raising=False)
        vendor_name = "openai" if field == "embed" else "cohere"
        with pytest.raises(ValueError, match=f"{field.upper()}_MODEL"):
            Settings(**{f"{field}_provider": vendor_name, f"{field}_model": ""})

    def test_tei_needs_no_model(self):
        from config import Settings
        assert Settings(embed_provider="tei", embed_model="", rerank_provider="tei", rerank_model="").embed_provider == "tei"
