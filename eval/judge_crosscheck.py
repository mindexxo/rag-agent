"""judge 교차검증 (#103) — 같은 답변을 vLLM judge와 Claude judge로 이중 채점해 불일치를 뽑는다.

이 이슈의 진짜 산출물: "우리 judge(Qwen3-14B 자기채점)를 얼마나 믿어도 되는가"의 수치.
불일치 목록을 사람이 읽고(측정 규율) 어느 쪽이 맞는지 표본 확인하는 게 최종 판단.
Claude judge도 LLM이라 순환을 "끊는" 게 아니라 "다변화" — 최종 기준은 #98 사람 평가.

**부재단정 축**만 다룬다(저렴 — 인용 0건 답변 ~50건). RAGAS 축 교차검증은 전체가 19~38시간
이라 스모크 규모로만 별도 실행(ragas_eval.py RAGAS_JUDGE=claude), 여기서 안 묶는다.

**입력은 audit 재활용 한 경로뿐이다.** 기존 refusal 실행이 남긴 absence_judge_*.jsonl의
answer에 **Claude judge만** 새로 적용하고, vLLM 판정은 파일의 absence 컬럼을 쓴다. 같은
answer에 두 judge를 적용하는 게 순수 비교의 핵심 — 생성을 두 번 하면 비결정성이 섞여
judge 비교가 오염된다(그래서 refusal 즉석 실행 경로는 없앴다: absence_misses가 answer를
보존하지 않아 무의미했다, 리뷰 발견). GPU 불필요(Claude 좌석만).

**judge 비결정성**: Claude도 LLM이라 1회 판정은 흔들릴 수 있다(absence_judge selftest가
그 흔들림을 실측). 그래서 불일치로 판정된 건만 Claude 재판정 1회를 더 해 flaky를 표시한다
(refusal.compute의 assertion 재판정 패턴과 같은 근거) — 불일치가 견해차인지 그 순간의
flaky인지 결과에 남긴다.

실행: python -m eval.judge_crosscheck --audit eval/results/absence_judge_XXXX.jsonl
"""
import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path

from eval.absence_judge import judge_absence
from eval.claude_client import ClaudeCliClient

CONCURRENCY = 4   # claude -p 콜=서브프로세스 (longcontext_claude 선례)
RESULT_DIR = Path(__file__).resolve().parent / "results"


async def _claude_label(claude: ClaudeCliClient, sem: asyncio.Semaphore,
                        query: str, answer: str) -> tuple[bool | None, str | None]:
    """Claude judge 판정 → (부재단정 여부, 사유). 어떤 실패든 (None, error)로 흡수한다.

    LlmJudgmentFailed(JSON 파싱 실패)뿐 아니라 RuntimeError(타임아웃·좌석 한도, claude_client가
    시끄럽게 던짐)도 잡는다 — 좁게 잡으면 다건 중 1건 실패가 gather를 뚫어 이미 계산된
    나머지 결과가 통째로 유실된다(리뷰 발견). refusal._one·absence.selftest와 같은 넓은 캐치.
    """
    async with sem:
        try:
            v = await judge_absence(claude, query, answer)
            return v.label == "absence_assertion", v.reason
        except Exception as exc:
            return None, f"판정 불가: {type(exc).__name__}: {exc}"


def _load_audit(path: Path) -> list[dict]:
    """audit 재활용 — answer는 그대로, vLLM 판정은 absence 컬럼, judge 버전 보존.

    여러 버전이 섞인 파일이면 불일치가 judge 차이인지 프롬프트 버전 차이인지 구분 불가라
    중단한다(리뷰 발견 — 지금은 전부 v1이지만 재사용 시 조용히 오염될 구조였다).
    """
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    kept = [{"id": r["id"], "query": r["query"], "answer": r["answer"],
             "vllm": r.get("absence"), "vllm_judge_version": r.get("judge_prompt_version")}
            for r in rows if r.get("answer")]
    versions = {r["vllm_judge_version"] for r in kept}
    if len(versions) > 1:
        raise SystemExit(f"audit에 judge 버전이 섞여 있다 {versions} — 버전별로 분리해 실행할 것.")
    return kept


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", required=True,
                    help="기존 absence_judge_*.jsonl (answer 보존분 재활용, GPU 불필요)")
    args = ap.parse_args()

    rows = _load_audit(Path(args.audit))
    ver = rows[0]["vllm_judge_version"] if rows else None
    print(f"[교차검증] audit 재활용 {len(rows)}건 — Claude judge 적용 (vLLM 판정 버전 {ver})")

    claude = ClaudeCliClient()
    sem = asyncio.Semaphore(CONCURRENCY)
    claude_results = await asyncio.gather(*(
        _claude_label(claude, sem, r["query"], r["answer"]) for r in rows))

    compared, agree, disagree, failed = [], 0, 0, 0
    for r, (claude_label, reason) in zip(rows, claude_results):
        v = r["vllm"]
        if claude_label is None:
            failed += 1
            compared.append({**r, "claude": None, "claude_reason": reason, "match": None})
        elif v is None:
            compared.append({**r, "claude": claude_label, "claude_reason": reason, "match": None})
        else:
            match = (v == claude_label)
            agree += match
            disagree += not match
            compared.append({**r, "claude": claude_label, "claude_reason": reason, "match": match})

    # 불일치 건만 Claude 재판정 1회 — 견해차 vs flaky 구분 (비용 미미, 불일치 수만큼)
    for r in compared:
        if r["match"] is False:
            again, _ = await _claude_label(claude, sem, r["query"], r["answer"])
            r["claude_flaky"] = (again != r["claude"])   # 재판정에서 뒤집히면 flaky

    RESULT_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M")   # 고정 이름 덮어쓰기 사고 방지 (save_audit 관례)
    out = RESULT_DIR / f"judge_crosscheck_{stamp}_n{len(compared)}.jsonl"
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in compared) + "\n")

    total = agree + disagree
    print(f"\n부재단정 축 — vLLM(judge {ver}) ↔ Claude 판정")
    if total:
        print(f"  일치 {agree}/{total} = {agree/total:.1%} · 불일치 {disagree} = {disagree/total:.1%}")
    print(f"  Claude 판정 불가 {failed}건")
    if disagree:
        print(f"\n[불일치 — 사람이 원문을 읽어 어느 쪽이 맞는지 확인할 것]")
        for r in compared:
            if r["match"] is False:
                flaky = " [flaky — 재판정에서 뒤집힘]" if r.get("claude_flaky") else ""
                print(f"  {r['id']}: vLLM={'단정' if r['vllm'] else '정상'} / "
                      f"Claude={'단정' if r['claude'] else '정상'}{flaky}")
                print(f"    Q: {r['query'][:50]}")
                print(f"    A: {r['answer'][:90]!r}")
                print(f"    Claude 사유: {(r['claude_reason'] or '')[:80]}")
    print(f"\n→ {out}")


if __name__ == "__main__":
    asyncio.run(main())
