"""검색 순위 결정 단위 테스트 — _rank_single / _rank_multi / rerank_maxpool (#38).

DB·TEI 없이 도는 순수 테스트. retrieve_candidates에서 분해해 나온 조각들이라,
분해가 동작을 바꾸지 않았음을 이 파일이 함수 단위로 못박는다.

**단일/멀티 비대칭을 고정한다** — `_rank_single`은 top_n으로 자르고 `_rank_multi`는
자르지 않는다. 의도된 설계가 아니라 #5가 멀티 경로를 얹으며 생긴 부작용이고,
고치는 것은 #39 소관이다. 여기서는 "지금 이렇다"를 고정해, 리팩터링이 이 차이를
모르고 없애버리는 일을 막는다.
"""
import pytest

from rag.retriever import RetrievedChunk, _rank_multi, _rank_single, _rrf_fuse


def _chunk(chunk_id: int) -> RetrievedChunk:
    return RetrievedChunk(chunk_id=chunk_id, document_id=1, text=f'본문{chunk_id}',
                          heading_path=[], page=None, filename='정책.pdf', version=1)


class TestRankSingle:
    """단일 쿼리 순위 — distance 순 그대로, top_n으로 자른다."""

    def test_top_n으로_자른다(self):
        assert _rank_single([1, 2, 3, 4, 5], top_n=3) == [1, 2, 3]

    def test_순서를_바꾸지_않는다(self):
        # dense distance 순으로 이미 정렬돼 들어온다 — 재정렬하면 안 된다
        assert _rank_single([9, 4, 7, 1], top_n=4) == [9, 4, 7, 1]

    def test_top_n보다_적으면_그대로(self):
        assert _rank_single([1, 2], top_n=20) == [1, 2]

    def test_빈_입력(self):
        assert _rank_single([], top_n=20) == []


class TestRankMulti:
    """쿼리 확장(#5) 순위 — RRF로 정렬하되 **자르지 않는다**."""

    def test_자르지_않는다(self):
        """비대칭의 핵심. 여기에 슬라이스가 생기면 리랭커 max-pool이 볼 후보가 줄어든다.

        멀티 경로는 union 전체를 리랭커에 넘기고 top_n은 리랭크 뒤에 확정한다
        (retrieve_candidates). _rank_multi가 미리 자르면 그 설계가 조용히 깨진다.
        """
        per_query = [list(range(1, 31)), list(range(31, 61)), list(range(61, 91))]
        assert len(_rank_multi(per_query)) == 90       # 겹침 없음 → union 90개 전부

    def test_RRF_점수_내림차순이다(self):
        per_query = [[10, 20, 30], [20, 30, 10]]
        ranked = _rank_multi(per_query)
        scores = _rrf_fuse(per_query)
        assert ranked == sorted(ranked, key=lambda cid: -scores[cid])

    def test_여러_쿼리에서_잡힌_청크가_앞선다(self):
        # 20은 두 리스트 모두 상위 → 한 리스트에서만 1위인 10보다 합산 점수가 높다
        assert _rank_multi([[10, 20], [20, 99]])[0] == 20

    def test_중복_제거된다(self):
        # union이므로 같은 청크가 여러 쿼리에 걸려도 한 번만
        assert sorted(_rank_multi([[1, 2], [2, 3]])) == [1, 2, 3]

    def test_한_리스트만_와도_동작(self):
        assert _rank_multi([[5, 6, 7]]) == [5, 6, 7]


class TestRerankMaxpool:
    """멀티쿼리 채점 — 청크별 최고점(max-pool)으로 정렬. rag.reranker로 이관(#38)."""

    @pytest.fixture
    def chunks(self):
        return [_chunk(1), _chunk(2), _chunk(3)]

    async def _run(self, monkeypatch, matrix, queries, chunks):
        """rerank_scores를 고정 행렬로 대체 — TEI 없이 조합 로직만 검증."""
        import rag.reranker as rr
        calls = iter(matrix)

        async def _fake(query, chs):
            return next(calls)

        monkeypatch.setattr(rr, 'rerank_scores', _fake)
        return await rr.rerank_maxpool(queries, chunks)

    @pytest.mark.asyncio
    async def test_청크별_최고점으로_정렬(self, monkeypatch, chunks):
        # 청크1 max=0.1, 청크2 max=0.9, 청크3 max=0.5 → 2, 3, 1 순
        result = await self._run(monkeypatch,
                                 [[0.1, 0.2, 0.5], [0.0, 0.9, 0.3]],
                                 ['q1', 'q2'], chunks)
        ordered, best = result
        assert [c.chunk_id for c in ordered] == [2, 3, 1]
        assert best == [0.9, 0.5, 0.1]        # 점수도 같은 순서로 대응해야 otel 기록이 맞는다

    @pytest.mark.asyncio
    async def test_한_쿼리라도_실패하면_None(self, monkeypatch, chunks):
        """부분 성공을 쓰지 않는다 — 청크마다 max를 취한 쿼리 수가 달라져 비교 불가."""
        result = await self._run(monkeypatch,
                                 [[0.1, 0.2, 0.5], None],
                                 ['q1', 'q2'], chunks)
        assert result is None

    @pytest.mark.asyncio
    async def test_전부_실패하면_None(self, monkeypatch, chunks):
        assert await self._run(monkeypatch, [None, None], ['q1', 'q2'], chunks) is None

    @pytest.mark.asyncio
    async def test_동점이면_원_순서_유지(self, monkeypatch, chunks):
        # sorted가 안정 정렬이므로 동점 청크의 상대 순서는 입력 순서 그대로
        result = await self._run(monkeypatch, [[0.5, 0.5, 0.5]], ['q1'], chunks)
        ordered, _ = result
        assert [c.chunk_id for c in ordered] == [1, 2, 3]
