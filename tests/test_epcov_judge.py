"""EPCov 2단계 판정 (#221) — 결정적 부분만 고정한다. 실판정자 품질은 `python -m eval.epcov_judge`(selftest)로 본다.

가짜 판정자로 LLM·Claude를 부르지 않는다. 임베딩(TEI)도 타지 않게 입력을 고른다 — 규칙 단계에서
부분문자열로 끝나거나(적중), 금액 게이트에서 막히거나, rule_hits를 직접 넘긴다.
"""
import asyncio
import inspect
import json

import pytest

from eval import epcov_judge as E
from eval import generation as G
from rag.llm_schemas import LlmJudgmentFailed


class _FakeJudge:
    """user 메시지의 '기대 포인트: …' 첫 줄로 라벨을 고른다. fail에 든 포인트는 판정 실패."""

    def __init__(self, labels: dict[str, str], fail: set[str] = frozenset()):
        self.labels, self.fail, self.calls = labels, fail, []

    async def acomplete(self, messages, **_):
        point = messages[-1]["content"].split("\n", 1)[0].removeprefix("기대 포인트: ")
        self.calls.append(point)
        if point in self.fail:
            raise LlmJudgmentFailed("judge down")
        return json.dumps({"label": self.labels[point], "reason": "fake"})


def _run(coro):
    return asyncio.run(coro)


class TestConfirmPoints:
    def test_대상이_없으면_판정자를_부르지_않는다(self):
        judge = _FakeJudge({})
        final, audit = _run(E.confirm_points(judge, "답", ["a"], [True], [], asyncio.Semaphore(2)))
        assert final == [True] and audit == [] and judge.calls == []

    def test_covered만_적중으로_뒤집고_not_covered는_유지(self):
        judge = _FakeJudge({"p1": "covered", "p2": "not_covered"})
        final, audit = _run(E.confirm_points(judge, "답", ["p0", "p1", "p2"], [True, False, False], [1, 2],
                                             asyncio.Semaphore(2)))
        assert final == [True, True, False]
        assert sorted(judge.calls) == ["p1", "p2"]                     # 규칙 적중 p0은 묻지 않는다
        assert {a["k"]: a["label"] for a in audit} == {1: "covered", 2: "not_covered"}

    def test_판정_실패는_미적중_유지하고_사유를_남긴다(self):
        # fail-closed — 판정 못 한 것을 '담김'으로 추측하면 지표가 실패를 숨긴다(misinfo_judge와 같은 원칙)
        judge = _FakeJudge({}, fail={"p"})
        final, audit = _run(E.confirm_points(judge, "답", ["p"], [False], [0], asyncio.Semaphore(1)))
        assert final == [False]
        assert audit[0]["label"] is None and audit[0]["error"].startswith("LlmJudgmentFailed")


class TestEpcovScores:
    def test_금액이_빠진_미적중은_심판에_넘기지_않는다(self):
        # vLLM 판정자가 '9,600원 기대 / 9,000원 답'을 담김으로 본 오판(#221 실측) — 게이트가 판정 전에 막는다
        judge = _FakeJudge({"적립금은 9,600원": "covered"})
        s = _run(G._epcov_scores(judge, asyncio.Semaphore(1), "기본 적립율 3%라 9,000원이 적립됩니다.",
                                 ["적립금은 9,600원"], [False]))
        assert judge.calls == []
        assert s["expected_points_coverage"] == 0.0 and s["expected_points_audit"] == []

    def test_최종과_규칙_커버리지를_둘_다_남긴다(self):
        # 이중 기록 — 규칙만의 값으로 #221 이전 수치와 비교할 수 있어야 한다. 두 포인트 모두 정상 판정돼야 한다
        # (리뷰: 두 번째 라벨이 없어 판정 '실패'로도 같은 0.5가 나와 우연히 통과하던 것을 고침)
        judge = _FakeJudge({"데님 이염은 보상 대상이 아닙니다": "covered", "개별 심사 가능": "not_covered"})
        s = _run(G._epcov_scores(judge, asyncio.Semaphore(1), "따라서 이염은 보상 대상이 아닙니다.",
                                 ["데님 이염은 보상 대상이 아닙니다", "개별 심사 가능"], [False, False]))
        assert s["expected_points_coverage"] == 0.5
        assert s["expected_points_coverage_rule"] == 0.0
        assert sorted(a["label"] for a in s["expected_points_audit"]) == ["covered", "not_covered"]
        assert s["expected_points_judge_errors"] == 0

    def test_판정_실패_수를_행에_남긴다(self):
        # 한도·로그인 만료로 전부 실패하면 '심판 반영'처럼 보이는 규칙 결과가 된다 — 실패는 보여야 한다(리뷰)
        judge = _FakeJudge({}, fail={"p1", "p2"})
        s = _run(G._epcov_scores(judge, asyncio.Semaphore(2), "답", ["p1", "p2"], [False, False]))
        assert s["expected_points_judge_errors"] == 2
        assert s["expected_points_coverage"] == s["expected_points_coverage_rule"] == 0.0

    def test_판정자가_없으면_규칙_값_그대로(self):
        # run_mode(epcov_judging=False) — 생성 일관성 축이 쓰는 경로
        s = _run(G._epcov_scores(None, asyncio.Semaphore(1), "답", ["a", "b"], [True, False]))
        assert s["expected_points_coverage"] == s["expected_points_coverage_rule"] == 0.5
        assert s["expected_points_audit"] == []

    def test_포인트가_없으면_None(self):
        s = _run(G._epcov_scores(_FakeJudge({}), asyncio.Semaphore(1), "답", [], []))
        assert s["expected_points_coverage"] is None and s["expected_points_coverage_rule"] is None


class TestRuleStage:
    def test_마크다운_굵게_표시가_부분문자열_일치를_깨지_않는다(self):
        # '총 … = **240,000원**입니다'를 기대 '240,000원입니다'가 놓쳤다(#221 실측, 사람 판정 '담김')
        assert G.expected_points_hits("합계는 **240,000원**입니다.", ["240,000원입니다"]) == [True]

    def test_골드_포인트_쪽의_굵게_표시도_지운다(self):
        # 골드 5행은 포인트에 '**SK-250 베이비**'가 있다 — 답변만 지우면 부분문자열 단계에서 영영 안 맞았다(리뷰·확인)
        assert G.expected_points_hits("제일 가벼운 제품은 SK-250 베이비입니다",
                                      ["제일 가벼운 제품은 **SK-250 베이비**입니다"]) == [True]
        assert G.point_amounts_present("14,400원입니다", "결제 금액은 **14,400원**")

    def test_금액_게이트_공개_진입점(self):
        assert G.point_amounts_present("결제 금액은 **14,400원**입니다", "결제 금액은 14,400원")
        assert G.point_amounts_present("3만원을 환불합니다", "환불받는 금액은 30,000원")
        assert not G.point_amounts_present("9,000원이 적립됩니다", "적립금은 9,600원")
        assert G.point_amounts_present("아무 말", "금액 없는 포인트")          # 금액 없으면 통과


class TestJudgeSwitch:
    def test_기본은_Claude_버전_고정_모델(self, monkeypatch):
        monkeypatch.delenv("EPCOV_JUDGE", raising=False)
        monkeypatch.delenv("EPCOV_JUDGE_MODEL", raising=False)
        assert E.judge_spec() == ("claude", "claude-sonnet-4-5-20250929")
        assert E.judge_label() == f"claude:claude-sonnet-4-5-20250929:{E.JUDGE_PROMPT_VERSION}"

    def test_vllm으로_바꿀_수_있고_다른_심판_스위치와_독립(self, monkeypatch):
        # 전용 스위치 — ABSENCE_JUDGE(부재·오정보 심판)를 claude로 바꿔도 EPCov는 따로 간다
        monkeypatch.setenv("EPCOV_JUDGE", "vllm")
        monkeypatch.setenv("ABSENCE_JUDGE", "claude")
        assert E.judge_spec()[0] == "vllm"

    def test_claude_CLI가_없으면_바로_실패하고_vllm을_안내한다(self, monkeypatch):
        # 조용히 규칙만으로 진행하지 않는다 — 다른 눈금이 '심판 반영'으로 착각되는 것을 막는다(리뷰 결정 a)
        import shutil
        monkeypatch.delenv("EPCOV_JUDGE", raising=False)
        monkeypatch.setattr(shutil, "which", lambda name: None)
        with pytest.raises(SystemExit, match="EPCOV_JUDGE=vllm"):
            E.judge_llm()

    def test_모르는_값은_거절(self, monkeypatch):
        monkeypatch.setenv("EPCOV_JUDGE", "openai")
        with pytest.raises(SystemExit):
            E.judge_spec()


class TestFixturesAndPrompt:
    def test_fixture는_양쪽_라벨을_다_갖고_공개_가능한_모의_사례만(self):
        labels = {f["expected"] for f in E.FIXTURES}
        assert labels == {"covered", "not_covered"}
        assert not any("인티큐브" in f["answer"] or "inticube" in f["id"] for f in E.FIXTURES)

    def test_label에_기본값이_없다(self):
        assert "label" in E.CoverageJudgment.model_json_schema()["required"]

    def test_메시지_조립(self):
        m = E._build_messages("답변", "포인트")
        assert [x["role"] for x in m] == ["system", "user"]
        assert m[1]["content"] == "기대 포인트: 포인트\n\n답변:\n답변"


def test_생성_일관성_축은_EPCov_심판을_끈다(monkeypatch, tmp_path):
    # 규칙 판정의 런간 일치를 재는 축 — 심판을 R회 부르면 비용이 R배이고 눈금도 달라진다.
    # 소스 문자열이 아니라 실제 run_mode 호출 인자를 본다(리뷰: 주석만 있어도 통과하던 테스트)
    from contextlib import asynccontextmanager
    from eval import consistency as C

    seen = []

    async def fake_run_mode(session, llm, mode, rows, resolved, **kw):
        seen.append(kw)
        return [{"id": r["id"], "answer": "포인트 a 포함", "scores": {}} for r in rows]

    @asynccontextmanager
    async def fake_session():
        yield None

    gtype = sorted(G.GEN_TYPES)[0]
    monkeypatch.setattr(C, "load_gold", lambda: [{"id": "x1", "type": gtype, "query": "q", "expected_points": ["포인트 a"]}])
    monkeypatch.setattr(C, "run_mode", fake_run_mode)
    monkeypatch.setattr(C, "AsyncSessionLocal", fake_session)
    monkeypatch.setattr(C, "create_llm", lambda: None)
    monkeypatch.setattr(C, "RUNS", 2)
    monkeypatch.setattr(C, "OUT", tmp_path / "c.jsonl")
    monkeypatch.setattr(C, "RESULT_DIR", tmp_path)
    asyncio.run(C.main())
    assert seen and all(kw.get("epcov_judging") is False for kw in seen)


def test_재채점은_심판_반영분을_지우지_않는다(monkeypatch, tmp_path):
    # rescore_v2: 규칙만으로 다시 계산해 덮으면 심판 반영분이 사라진다 — 최종 = 새 규칙 적중 ∪ 감사의 covered (리뷰)
    from eval import rescore_v2 as R
    row = {"id": "q1", "answer": "포인트 a 포함", "epcov_judge": "claude:x:v1",
           "scores": {"citation_accuracy": 1.0, "expected_points_coverage": 1.0, "expected_points_coverage_rule": 0.5,
                      "expected_points_audit": [{"k": 1, "label": "covered", "reason": "", "error": None}]}}
    (tmp_path / "results").mkdir()
    for mode in ("oracle", "retrieved"):
        (tmp_path / "results" / f"generation_{mode}.jsonl").write_text(json.dumps(row, ensure_ascii=False))
    monkeypatch.setattr(R, "ROOT", tmp_path)
    monkeypatch.setattr(R, "GOLD", {"q1": {"expected_docs": [], "expected_points": ["포인트 a", "다른 말로 담긴 것"]}})
    monkeypatch.setattr(R, "citation_accuracy", lambda answer, docs: 1.0)
    monkeypatch.setattr(R, "expected_points_hits", lambda answer, points: [True, False])   # 임베딩(TEI) 안 타게
    R.main()
    out = json.loads((tmp_path / "results" / "generation_retrieved.jsonl").read_text())
    assert out["scores"]["expected_points_coverage_rule"] == 0.5
    assert out["scores"]["expected_points_coverage"] == 1.0


def test_요약은_판정자와_판정_실패를_크게_찍는다(capsys):
    # 전부 실패하면 EPCov = EPCov규칙인데 '심판 반영'처럼 보인다 — 요약에서 바로 보여야 한다(리뷰)
    rows = [{"type": "single_fact", "epcov_judge": "claude:m:v1",
             "scores": {"expected_points_coverage": 0.5, "expected_points_coverage_rule": 0.5,
                        "expected_points_audit": [{"k": 1, "label": None, "reason": None, "error": "RuntimeError: limit"}],
                        "expected_points_judge_errors": 1, "citation_accuracy": 1.0, "must_not_contain_violations": None}}]
    G.summarize(rows)
    out = capsys.readouterr().out
    assert "claude:m:v1" in out and "심판 실패 1/1" in out
