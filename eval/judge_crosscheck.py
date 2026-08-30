"""judge 교차검증 (#103) — 같은 답변을 vLLM judge와 Claude judge로 이중 채점해 불일치를 뽑는다.

이 이슈의 진짜 산출물: "우리 judge(Qwen3-14B 자기채점)를 얼마나 믿어도 되는가"의 수치.
불일치 목록을 사람이 읽고(측정 규율) 어느 쪽이 맞는지 표본 확인하는 게 최종 판단.
Claude judge도 LLM이라 순환을 "끊는" 게 아니라 "다변화" — 최종 기준은 #98 사람 평가.

**부재단정 축**만 다룬다(저렴 — 인용 0건 답변 ~50건). RAGAS 축 교차검증은 전체가 19~38시간
이라 스모크 규모로만 별도 실행(ragas_eval.py RAGAS_JUDGE=claude), 여기서 안 묶는다.

입력 두 경로:
- --audit <파일>: 기존 refusal 실행이 남긴 absence_judge_*.jsonl을 재활용 —
  그 안의 answer에 **Claude judge만** 새로 적용(vLLM 판정은 파일의 absence 컬럼). GPU 불필요.
- (--audit 없음): refusal.compute()를 즉석 실행해 answer+vLLM판정을 얻고 Claude를 적용.
  생성·vLLM judge를 새로 태우므로 GPU 필요.

같은 answer에 두 judge를 적용하는 게 핵심 — 생성을 두 번 하면 비결정성이 섞여 순수한
judge 비교가 안 된다. 그래서 audit 재활용(같은 answer 고정)이 정석이다.

실행: python -m eval.judge_crosscheck --audit eval/results/absence_judge_XXXX.jsonl
"""
import argparse
import asyncio
import json
from pathlib import Path

from eval.absence_judge import judge_absence
from eval.claude_client import ClaudeCliClient
from rag.llm_schemas import LlmJudgmentFailed

CONCURRENCY = 4   # claude -p 콜=서브프로세스 (longcontext_claude 선례)
RESULT_DIR = Path(__file__).resolve().parent / "results"


async def _claude_label(claude: ClaudeCliClient, sem: asyncio.Semaphore,
                        query: str, answer: str) -> tuple[bool | None, str | None]:
    """Claude judge 판정 → (부재단정 여부, 사유). 판정 불가는 (None, error)."""
    async with sem:
        try:
            v = await judge_absence(claude, query, answer)
            return v.label == "absence_assertion", v.reason
        except LlmJudgmentFailed as exc:
            return None, f"판정 불가: {exc}"


async def _rows_from_audit(path: Path) -> list[dict]:
    """기존 audit 재활용 — answer는 그대로, vLLM 판정은 absence 컬럼."""
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    return [{"id": r["id"], "query": r["query"], "answer": r["answer"],
             "vllm": r.get("absence")} for r in rows if r.get("answer")]


async def _rows_from_refusal() -> list[dict]:
    """audit이 없으면 refusal을 즉석 실행 — 인용 0건 답변에 대한 vLLM 판정을 얻는다 (GPU 필요)."""
    from eval.refusal import compute
    r = await compute()
    # compute()가 부재단정 판정한 행(absence is not None)만 교차검증 대상
    return [{"id": m["id"], "query": m["query"], "answer": m["query"],  # audit 없으면 answer 미보존
             "vllm": None} for m in r.get("absence_misses", [])]


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", help="기존 absence_judge_*.jsonl (재활용, GPU 불필요)")
    args = ap.parse_args()

    if args.audit:
        rows = await _rows_from_audit(Path(args.audit))
        print(f"[교차검증] audit 재활용 {len(rows)}건 — Claude judge만 적용 (vLLM 판정은 파일값)")
    else:
        rows = await _rows_from_refusal()
        print(f"[교차검증] refusal 즉석 실행 {len(rows)}건")

    claude = ClaudeCliClient()
    sem = asyncio.Semaphore(CONCURRENCY)
    claude_results = await asyncio.gather(*(
        _claude_label(claude, sem, r["query"], r["answer"]) for r in rows))

    compared, agree, disagree, failed = [], 0, 0, 0
    for r, (claude_label, reason) in zip(rows, claude_results):
        v = r["vllm"]
        if claude_label is None:
            failed += 1
            row = {**r, "claude": None, "claude_reason": reason, "match": None}
        elif v is None:
            row = {**r, "claude": claude_label, "claude_reason": reason, "match": None}
        else:
            match = (v == claude_label)
            agree += match
            disagree += not match
            row = {**r, "claude": claude_label, "claude_reason": reason, "match": match}
        compared.append(row)

    RESULT_DIR.mkdir(exist_ok=True)
    out = RESULT_DIR / "judge_crosscheck.jsonl"
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in compared) + "\n")

    total = agree + disagree
    print(f"\n부재단정 축 — vLLM↔Claude 판정")
    if total:
        print(f"  일치 {agree}/{total} = {agree/total:.1%} · 불일치 {disagree} = {disagree/total:.1%}")
    print(f"  Claude 판정 불가 {failed}건")
    if disagree:
        print(f"\n[불일치 — 사람이 원문을 읽어 어느 쪽이 맞는지 확인할 것]")
        for r in compared:
            if r["match"] is False:
                print(f"  {r['id']}: vLLM={'단정' if r['vllm'] else '정상'} / "
                      f"Claude={'단정' if r['claude'] else '정상'}")
                print(f"    Q: {r['query'][:50]}")
                print(f"    A: {r['answer'][:90]!r}")
                print(f"    Claude 사유: {(r['claude_reason'] or '')[:80]}")
    print(f"\n→ {out}")


if __name__ == "__main__":
    asyncio.run(main())
