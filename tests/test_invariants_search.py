"""불변식 대장(INVARIANTS.md)의 부분 칸 보강 — 검색 축(엔진 필터·리랭커 집합·식별자 가산 순서).

엔진·TEI 없이 돈다. 엔진 클라이언트와 리랭커 채점만 가짜로 바꾸고, 질의 조립·후보 조합·가산은 운영 코드를 그대로 탄다.
"""

import ast
import random
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from config import settings
from rag.identifiers import normalize_identifiers
from rag.search import os_search, reranker, retriever
from rag.search.retrieval_types import DenseHit, DenseHits, RetrievedChunk

ROOT = Path(__file__).resolve().parent.parent


def _chunk(cid: int, identifiers: list[str] | None = None) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=cid,
        document_id=cid,
        text=f'본문{cid}',
        heading_path=[f'제{cid}장'],
        page=None,
        filename=f'규정{cid}.pdf',
        version=1,
        identifiers=identifiers or [],
    )


# ── INV-03 테넌트·검색가능 필터는 엔진 질의 안에 ───────────────────────────────────────────────────


class _RecordingClient:
    """search 본문을 기록하는 엔진 대역 — 빈 결과를 돌려준다."""

    def __init__(self) -> None:
        self.bodies: list[dict] = []

    async def search(self, *, index: str, body: dict) -> dict:
        self.bodies.append(body)
        return {'hits': {'hits': []}}


def _search_callers(path: Path) -> set[str]:
    """그 모듈에서 `.search(`를 부르는 함수 이름."""
    tree = ast.parse(path.read_text(encoding='utf-8'))
    return {
        fn.name
        for fn in ast.walk(tree)
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef))
        for c in ast.walk(fn)
        if isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute) and c.func.attr == 'search'
    }


@pytest.mark.asyncio
async def test_INV03_검색_경로의_모든_엔진_질의가_테넌트_필터를_엔진_안에_건다(monkeypatch):
    """INV-03 테넌트·검색가능 필터는 엔진에서 건다 — kNN은 knn.filter 안에, 어휘·식별자는 bool.filter 안에.
    post_filter(검색 후 거르기)는 어디에도 없어야 한다. 엔진 질의 함수가 늘면 여기 대상에 넣으라고 실패한다."""
    fake = _RecordingClient()
    monkeypatch.setattr(os_search, 'client', lambda: fake)
    want = os_search.tenant_filter('t-a')

    await os_search.search_dense_per_query('t-a', [SimpleNamespace(dense=[0.1, 0.2])], 5)
    await os_search.search_lexical('t-a', '환불 기간', 5)
    await os_search.search_identifiers('t-a', ['3조'], 5)
    dense, lexical, ident = fake.bodies

    assert dense['query']['knn']['dense']['filter'] == want
    assert want in lexical['query']['bool']['filter']
    assert want in ident['query']['bool']['filter']
    assert all('post_filter' not in b for b in fake.bodies)

    tested = {'search_dense_per_query', 'search_lexical', 'search_identifiers'}
    assert _search_callers(ROOT / 'rag' / 'search' / 'os_search.py') == tested, '새 엔진 질의 함수를 이 테스트에 더하라'


# ── INV-55·54 하네스: 엔진·임베딩을 대역으로 둔 retrieve_candidates ─────────────────────────────────


class _Harness:
    """풀 [1, 2, 3] — 3번만 질문의 식별자를 가진다. 리랭커 점수는 1 > 2 > 3, 가산이 3을 맨 앞으로 올린다."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.token = normalize_identifiers('제3조 내용 알려줘')[0]
        self.chunks = {1: _chunk(1), 2: _chunk(2), 3: _chunk(3, [self.token])}
        self.rerank_inputs: list[tuple[str, list[tuple]]] = []
        scores = {1: 0.6, 2: 0.5, 3: 0.4}  # 가산 0.3이면 3이 0.7로 1위

        async def embed(texts: list[str], kind: str = 'passage') -> list:
            return [SimpleNamespace(dense=[0.0]) for _ in texts]

        async def dense(tenant_id: str, q_embs: list, n: int) -> DenseHits:
            return DenseHits([[1, 2, 3]], [DenseHit(1, 0.1)])

        async def lexical(tenant_id: str, query: str, limit: int) -> list[int]:
            return []

        async def ident(tenant_id: str, tokens: list[str], limit: int) -> list[int]:
            return [3]  # 이미 풀에 있다 — 채널을 켜고 꺼도 리랭커 풀은 같다

        async def chunk_map(ids: list[int]) -> dict:
            return {i: self.chunks[i] for i in ids}

        async def rerank_scores(query: str, chunks: list) -> list[float]:
            self.rerank_inputs.append(
                (query, [(c.chunk_id, reranker._rerank_text(c), c.text, c.matched_ident) for c in chunks])
            )
            return [scores[c.chunk_id] for c in chunks]

        monkeypatch.setattr(retriever, 'embed_texts', embed)
        monkeypatch.setattr(os_search, 'search_dense_per_query', dense)
        monkeypatch.setattr(os_search, 'search_lexical', lexical)
        monkeypatch.setattr(os_search, 'search_identifiers', ident)
        monkeypatch.setattr(os_search, 'fetch_chunk_map', chunk_map)
        monkeypatch.setattr(reranker, 'rerank_scores', rerank_scores)
        monkeypatch.setattr(settings, 'rerank_enabled', True)

    async def run(self, monkeypatch: pytest.MonkeyPatch, *, ident_on: bool) -> list[RetrievedChunk]:
        monkeypatch.setattr(settings, 'ident_channel_enabled', ident_on)
        got = await retriever.retrieve_candidates(None, 't-a', '제3조 내용 알려줘')
        return got.chunks


@pytest.mark.asyncio
async def test_INV55_식별자_가산은_리랭커_뒤에_하고_리랭커_입력은_채널과_무관하게_같다(monkeypatch):
    """INV-55 식별자 가산은 리랭커 뒤에 하고 리랭커 입력(질의·조각 글)은 바꾸지 않는다.
    채널을 켠 실행과 끈 실행에서 리랭커가 받은 것이 같아야 하고, 가산 표식(matched_ident)은 리랭커 입력에 없다."""
    h = _Harness(monkeypatch)
    on = await h.run(monkeypatch, ident_on=True)
    off = await h.run(monkeypatch, ident_on=False)

    rerank_on, rerank_off = h.rerank_inputs
    assert rerank_on == rerank_off, '식별자 채널이 리랭커 입력을 바꿨다'
    query, docs = rerank_on
    assert query == '제3조 내용 알려줘'
    assert all(matched is None for *_, matched in docs)
    assert [c.chunk_id for c in off] == [1, 2, 3]  # 리랭커 순서
    assert [c.chunk_id for c in on] == [3, 1, 2], '가산이 리랭커 점수 뒤에 적용되지 않았다'
    assert on[0].matched_ident == h.token


# ── INV-54 리랭커는 순서만 바꾼다 ────────────────────────────────────────────────────────────────


def _assert_permutation(before: list, after: list) -> None:
    assert len(after) == len(before)
    assert Counter(map(id, after)) == Counter(map(id, before)), '리랭크가 후보를 더하거나 빼거나 바꿨다'


@pytest.mark.asyncio
@pytest.mark.parametrize('seed', range(20))
async def test_INV54_maxpool은_어떤_점수에서도_후보의_순열만_돌려준다(monkeypatch, seed):
    """INV-54 리랭커는 후보 순서만 바꾼다(집합 불변) — 동점·음수·쿼리 여러 개를 섞은 점수 행렬로."""
    rng = random.Random(seed)
    chunks = [_chunk(i) for i in range(rng.randint(1, 70))]
    n_queries = rng.randint(1, 3)
    matrix = iter([[rng.choice([-1.0, 0.0, 0.5, rng.random()]) for _ in chunks] for _ in range(n_queries)])

    async def fake(query: str, chs: list) -> list[float]:
        return next(matrix)

    monkeypatch.setattr(reranker, 'rerank_scores', fake)
    ordered, best, best_qi = await reranker.rerank_maxpool([f'q{i}' for i in range(n_queries)], chunks)
    _assert_permutation(chunks, ordered)
    assert best == sorted(best, reverse=True) and len(best_qi) == len(chunks)


@pytest.mark.asyncio
async def test_INV54_TEI_응답이_섞인_순서로_와도_배치를_넘겨_집합이_같다(monkeypatch):
    """TEI /rerank는 점수순으로 index를 섞어 돌려준다 — 배치 상한(32)을 넘는 후보도 원래 조각에 점수가 붙어 순열이 된다."""
    chunks = [_chunk(i) for i in range(70)]

    async def score_batch(query: str, texts: list[str]) -> list[tuple[int, float]]:
        return sorted(((i, float(sum(map(ord, t)) % 97)) for i, t in enumerate(texts)), key=lambda r: -r[1])

    monkeypatch.setattr(reranker, '_score_batch', score_batch)
    scores = await reranker.rerank_scores('q', chunks)
    assert scores == [float(sum(map(ord, reranker._rerank_text(c))) % 97) for c in chunks]
    ordered, _, _ = await reranker.rerank_maxpool(['q'], chunks)
    _assert_permutation(chunks, ordered)


def test_INV54_식별자_가산도_후보_집합을_바꾸지_않는다():
    token = '3조'
    chunks = [_chunk(i, [token] if i % 2 else []) for i in range(10)]
    out, scores, qi = retriever.apply_identifier_boost(chunks, [float(i) for i in range(10)], [0] * 10, [token], 0.3)
    assert sorted(c.chunk_id for c in out) == list(range(10))
    assert len(scores) == len(qi) == 10


@pytest.mark.asyncio
async def test_INV54_검색_경로에서도_리랭크_전후_후보_집합이_같다(monkeypatch):
    h = _Harness(monkeypatch)
    got = await h.run(monkeypatch, ident_on=True)
    assert sorted(c.chunk_id for c in got) == [1, 2, 3]
