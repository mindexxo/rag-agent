"""하이브리드 가중치 검증(#248) 재생 계산부 — 운영 조립과의 동치·진단·부트스트랩. DB·엔진 불필요(운영 단계 함수는 합성 값으로 대체).

핵심은 동치 테스트다: 같은 합성 값을 운영 rag/retriever.retrieve_candidates에 끼워 **실제로 돌린 결과**와 eval/hybrid_replay.assemble이
같아야 하고, 재생을 일부러 망가뜨리면 그 비교가 반드시 실패해야 한다(#128 1차 측정은 조립이 틀린 채 기준선과 같은 값이 나왔다).
"""
import pytest

import rag.os_search
import rag.reranker
import rag.retriever
from config import settings
from eval import hybrid_replay as hr
from rag.embeddings import Embedding
from rag.identifiers import normalize_identifiers
from rag.retrieval_types import RetrievedChunk


def _chunk(cid, *, table_sheet=None, identifiers=(), text="본문"):
    return RetrievedChunk(chunk_id=cid, document_id=cid // 1000, text=text, heading_path=[table_sheet] if table_sheet else ["절"],
                          page=None, filename=f"d{cid // 1000}.pdf", version=1, is_table=table_sheet is not None,
                          sheet_text=f"시트 {table_sheet} 전체" if table_sheet else None, identifiers=list(identifiers))


def _record(query, dense30, bm25_30, ident, rerank, gold, *, set_="v2", token=None):
    return {"set": set_, "qid": "q1", "tenant": "t", "type": "single_fact", "query": query, "gold_ids": gold,
            "token": token, "dense30": [[c, 0.1 * i] for i, c in enumerate(dense30)],
            "dense100": [[c, 0.1 * i] for i, c in enumerate(dense30)], "bm25_30": bm25_30,
            "bm25_100": [[c, 10.0 - i] for i, c in enumerate(bm25_30)],
            "ident_tokens": normalize_identifiers(query), "ident": ident, "rerank": {str(k): v for k, v in rerank.items()}}


# 시나리오: 벡터 30 / BM25 30(벡터와 일부 겹침, 비중복 20개라 15개만 주입) / 식별자 일치 / 동점 / 두 시트 표 / 가산으로 역전
DENSE = list(range(1000, 1030))
BM25 = [1005, 1010] + list(range(2000, 2028))                    # 앞 둘은 벡터와 겹친다
IDENT = [3000, 1001]
CHUNKS = {c: _chunk(c) for c in DENSE + BM25[2:] + [3000]}
CHUNKS[1001] = _chunk(1001, identifiers=["11조"])
CHUNKS[3000] = _chunk(3000, identifiers=["11조"])
CHUNKS[2003] = _chunk(2003, table_sheet="시트A")
CHUNKS[2004] = _chunk(2004, table_sheet="시트B")                # 다른 시트 — 한 시트만 남긴다
RERANK = {c: 0.5 - 0.001 * i for i, c in enumerate(DENSE + BM25[2:] + [3000])}
RERANK.update({2003: 0.95, 2004: 0.94, 3000: 0.70, 1001: 0.69, 1007: 0.69, 2010: 0.69})   # 동점 묶음 포함
CASES = {
    "식별자_있음": _record("가나규칙 11조 뭐야", DENSE, BM25, IDENT, RERANK, [3000]),
    "식별자_없음": _record("교환 기한 알려줘", DENSE, BM25, [], RERANK, [2003]),
    "벡터_비어있음": _record("교환 기한", [], BM25, [], RERANK, [2003]),
}


@pytest.fixture
def production(monkeypatch):
    """운영 단계 함수를 레코드 값으로 대체 — retrieve_candidates 본체(조립·정렬·가산·자르기)는 운영 코드 그대로 돈다.
    대체하는 이음새 목록은 eval/hybrid_weights._patched와 같다 — 운영 이름이 바뀌면 둘을 함께 고칠 것."""
    state = {}

    async def _embed(texts):
        assert len(texts) == 1, "단일 질의 경로만 재생한다"
        return [Embedding(dense=[0.0])]

    async def _dense(tenant, q_embs, n):
        assert n == 30
        rec = state["rec"]
        return [[c for c, _ in rec["dense30"]]], [tuple(x) for x in rec["dense30"]]

    async def _lexical(tenant, query, limit):
        assert limit == 30
        return list(state["rec"]["bm25_30"])

    async def _ident(tenant, tokens, limit):
        return list(state["rec"]["ident"])[:limit]

    async def _chunk_map(ids):
        return {c: CHUNKS[c] for c in ids if c in CHUNKS}

    async def _scores(query, chunks):
        return [state["rec"]["rerank"][str(c.chunk_id)] for c in chunks]

    monkeypatch.setattr(rag.retriever, "embed_texts", _embed)
    monkeypatch.setattr(rag.retriever, "_dense_candidates", _dense)
    monkeypatch.setattr(rag.retriever, "_lexical_candidates", _lexical)
    monkeypatch.setattr(rag.os_search, "search_identifiers", _ident)
    monkeypatch.setattr(rag.retriever, "_chunk_map", _chunk_map)
    monkeypatch.setattr(rag.reranker, "rerank_scores", _scores)
    for k, v in {"rerank_enabled": True, "hybrid_lexical_enabled": True, "hybrid_lexical_inject": 15,
                 "ident_channel_enabled": True, "ident_inject": 10, "ident_boost": 0.3}.items():
        monkeypatch.setattr(settings, k, v)

    async def run(rec):
        state["rec"] = rec
        cands = await rag.retriever.retrieve_candidates(None, rec["tenant"], rec["query"], top_n=hr.TOP_N)
        return [c.chunk_id for c in cands.chunks]

    return run


def test_조건_기본값이_운영_설정과_같다():
    from config import Settings
    d = Settings.model_fields
    assert (hr.BASELINE.lex_inject, hr.BASELINE.ident_inject, hr.BASELINE.ident_boost) == (
        d["hybrid_lexical_inject"].default, d["ident_inject"].default, d["ident_boost"].default)


@pytest.mark.asyncio
@pytest.mark.parametrize("name", list(CASES))
async def test_재생은_운영_retrieve_candidates와_같다(production, name):
    rec = CASES[name]
    assert [c.chunk_id for c in hr.assemble(rec, CHUNKS)] == await production(rec)


@pytest.mark.asyncio
@pytest.mark.parametrize("broken", [
    lambda rec, cond=hr.BASELINE: [c for c, _ in rec["dense30"]] + rec["bm25_30"][:15],              # 중복 제거 안 함
    lambda rec, cond=hr.BASELINE: [c for c, _ in rec["dense30"]] + [c for c in rec["bm25_30"] if c not in
                                   {d for d, _ in rec["dense30"]}][:20] + rec["ident"],              # 주입 상한 틀림
    lambda rec, cond=hr.BASELINE: [c for c, _ in rec["dense30"]] + [c for c in rec["bm25_30"] if c not in
                                   {d for d, _ in rec["dense30"]}][:15],                             # 식별자 주입 빠짐
])
async def test_재생을_망가뜨리면_동치_검사가_실패한다(production, monkeypatch, broken):
    # 동치 검사가 '구성상 늘 같음'이 아님을 보증한다 — 망가진 재생 중 하나라도 운영과 같게 나오면 검사가 무력하다
    monkeypatch.setattr(hr, "build_pool", broken)
    rec = CASES["식별자_있음"]
    assert [c.chunk_id for c in hr.assemble(rec, CHUNKS)] != await production(rec)


@pytest.mark.asyncio
async def test_가산이_빠진_재생도_동치_검사에서_걸린다(production, monkeypatch):
    # 식별자 가산(+0.3)이 3000(리랭커 0.70)을 시트 표 조각(0.95) 위로 올린다 — 가산을 빼면 순서가 달라져야 한다
    monkeypatch.setattr(hr, "apply_identifier_boost", lambda c, s, q, t, b: (c, s, q))
    rec = CASES["식별자_있음"]
    assert [c.chunk_id for c in hr.assemble(rec, CHUNKS)] != await production(rec)


def test_한_시트만_남고_top_n은_시트_필터_앞에서_자른다():
    out = [c.chunk_id for c in hr.assemble(CASES["식별자_없음"], CHUNKS)]
    assert 2003 in out and 2004 not in out and len(out) == hr.TOP_N - 1   # 버린 자리는 채우지 않는다


def test_캡처에_없는_후보를_요구하면_멈춘다():
    rec = dict(CASES["식별자_없음"], rerank={})
    with pytest.raises(hr.PoolCoverageError):
        hr.assemble(rec, CHUNKS)
    with pytest.raises(hr.PoolCoverageError):
        hr.build_pool(dict(rec, dense100=[[i, 0.0] for i in range(100)]), hr.Condition("넓음", dense_n=150))


def test_진단은_풀_누락과_리랭커_누락을_가른다():
    base = CASES["식별자_없음"]
    hit = hr.diagnose(base, CHUNKS)                                    # 2003: 풀 안·1위
    pool_miss = hr.diagnose(dict(base, gold_ids=[2027]), CHUNKS)       # BM25 30 안이지만 비중복 16번째 이후 → 풀 밖
    rerank_miss = hr.diagnose(dict(base, gold_ids=[1029]), CHUNKS)     # 풀 안이지만 점수 낮아 근거 5 밖
    assert (hit["status"], pool_miss["status"], rerank_miss["status"]) == ("hit", "pool_miss", "rerank_miss")
    assert pool_miss["in_bm25_30"] and not pool_miss["in_dense30"]
    assert hit["lexical_saves"] == 1          # 2003은 BM25 주입으로만 풀에 들어온다 — 주입을 끄면 근거에서 빠진다
    assert hit["bm25_rank"] == 6 and hit["dense_rank"] is None   # BM25 목록: 1005·1010(벡터와 겹침)·2000·2001·2002·2003
    summary = hr.summarize_diagnosis([dict(hit, type="a"), dict(pool_miss, type="a"), dict(rerank_miss, type="b")])
    assert summary["(전체)"]["lexical_saves"] == 1
    assert summary["(전체)"]["n"] == 3 and summary["a"]["upside_bm25_30"] == 1
    assert sum(summary["(전체)"][s] for s in ("hit", "rerank_miss", "pool_miss")) == 3


def test_키워드_축은_근거_글로_판정한다():
    rec = dict(CASES["식별자_없음"], set="keyword", token="시트A", gold_ids=[2003])
    assert hr.evaluate(rec, hr.assemble(rec, CHUNKS), CHUNKS)["in_evidence"] == 1.0


def test_같은_시트의_형제_조각이_근거면_정답을_본_것이다():
    # 정답 행(2005)과 같은 시트A의 2003이 근거로 접혀 넘어가면 모델은 시트 전체를 받는다 — id로는 2005가 근거에 없어도 적중
    chunks = {**CHUNKS, 2005: _chunk(2005, table_sheet="시트A")}       # 2005·2003은 같은 문서(id//1000)·같은 시트
    rec = dict(CASES["식별자_없음"], gold_ids=[2005])
    assert hr.evaluate(rec, hr.assemble(rec, chunks), chunks)["in_evidence"] == 1.0
    assert hr.evaluate(dict(rec, gold_ids=[1029]), hr.assemble(rec, chunks), chunks)["in_evidence"] == 0.0


def test_부트스트랩():
    same = hr.paired_bootstrap([1, 0, 1, 1], [1, 0, 1, 1])
    assert same["mean_diff"] == 0 and same["ci95"] == (0.0, 0.0) and same["ties"] == 4
    better = hr.paired_bootstrap([0] * 50 + [1] * 50, [1] * 100, n=2000)
    assert better["ci95"][0] > 0 and (better["wins"], better["losses"]) == (50, 0)
    assert hr.paired_bootstrap([0, 1], [1, 0], n=500) == hr.paired_bootstrap([0, 1], [1, 0], n=500)   # 시드 고정
    with pytest.raises(ValueError):
        hr.paired_bootstrap([1], [1, 0])


def test_혼합_가중치_0은_기준선과_같다():
    for name, rec in CASES.items():
        zero = hr.Condition("zero", blend="add", w_key=0.0, w_desc=0.0)
        assert [c.chunk_id for c in hr.assemble(rec, CHUNKS, zero)] == [c.chunk_id for c in hr.assemble(rec, CHUNKS)], name


def test_혼합은_BM25_상위_후보를_올린다():
    # 식별자_없음: 2000(BM25 비중복 1위, 리랭커 0.5-...)이 혼합으로 올라온다
    rec = CASES["식별자_없음"]
    base = [c.chunk_id for c in hr.assemble(rec, CHUNKS)]
    blended = [c.chunk_id for c in hr.assemble(rec, CHUNKS, hr.Condition("b", blend="add", w_key=0.3, w_desc=0.3))]
    pos = lambda ids: ids.index(2000) if 2000 in ids else len(ids)
    assert pos(blended) < pos(base)


def test_키워드형_질문_판정():
    assert hr.is_keyword_query("AR-2003") and hr.is_keyword_query("취업규칙 44조 뭐야") and hr.is_keyword_query("짧은 질문")
    assert not hr.is_keyword_query("냉동 닭 시켰는데 다 녹아서 물이 흥건하게 왔어요")


def test_모르는_혼합_방식은_멈춘다():
    with pytest.raises(ValueError):
        hr.assemble(CASES["식별자_없음"], CHUNKS, hr.Condition("오타", blend="covex", w_key=0.1, w_desc=0.1))
