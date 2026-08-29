"""생성 일관성 (#97 ②) — 같은 질문을 여러 번 물어도 같은 사실을 답하는가.

왜 재나: condense는 3런 flaky를 재는데 생성축은 1런뿐이었다. 상담 도구의 급소는
"물을 때마다 답이 다르면 못 쓴다"이고, 발표 예상 질문이기도 하다.

측정 정의:
- 표본: 골드셋 생성 타입에서 타입 균등 N문항(기본 50, _smoke_sample — 무작위 없음·재현 가능).
- 각 문항을 retrieved 모드(운영 경로 그대로: condense→검색→리랭크→생성)로 R회(기본 5) 반복.
  검색까지 매번 다시 탄다 — 재는 것은 생성 LLM 단독이 아니라 **시스템 전체의 재현성**이다
  (TEI 임베딩·리랭커도 콜마다 비결정적임이 실측돼 있다: 1.4e-4 오더).
- 지표(포인트 단위 — 판정은 eval.generation.expected_points_hits 재사용, 기준 복제 금지):
  - **포인트 일치율**: (문항, 포인트) 쌍 중 R런 판정이 전부 같은 비율 (전부 O or 전부 X).
  - **flaky 포인트율**: 런마다 갈리는 쌍의 비율 (= 1 - 일치율). condense flaky와 같은 관점.
  - 참고로 런별 EPCov 평균도 찍는다 — 평균은 같아도 개별 포인트가 출렁일 수 있어서
    평균만 보면 안 된다(그걸 보이는 게 이 축의 존재 이유다).

**GPU가 한가할 때 돌릴 것** — N×R 생성 콜(기본 250)이라 eval 풀런과 겹치면 서로 느려진다.
의존: DB + vLLM + TEI. 실행: python -m eval.consistency  [N=50 RUNS=5 환경변수]
"""
import asyncio
import json
import os
from collections import defaultdict
from pathlib import Path

from database import AsyncSessionLocal
from eval.generation import (GEN_TYPES, GOLD, RESULT_DIR, _smoke_sample,
                             expected_points_hits, run_mode)
from eval.retrieval import Resolved
from rag.llm import LlmClient

N = int(os.getenv("N", "50"))
RUNS = int(os.getenv("RUNS", "5"))
OUT = RESULT_DIR / "consistency.jsonl"


async def main() -> None:
    gold = [json.loads(l) for l in GOLD.read_text().splitlines() if l.strip()]
    sample = _smoke_sample([g for g in gold if g["type"] in GEN_TYPES], N)
    points_by_id = {g["id"]: g.get("expected_points", []) for g in sample}

    llm = LlmClient()
    all_rows = []
    async with AsyncSessionLocal() as session:
        for run in range(1, RUNS + 1):
            print(f"=== run {run}/{RUNS} ({len(sample)}문항, retrieved) ===")
            # retrieved 모드는 resolved(oracle 전용 재료)를 안 쓴다 — 빈 값으로 충분
            rows = await run_mode(session, llm, "retrieved", sample, Resolved())
            for r in rows:
                r["run"] = run
                r["point_hits"] = expected_points_hits(r["answer"], points_by_id[r["id"]])
            all_rows.extend(rows)

    RESULT_DIR.mkdir(exist_ok=True)
    OUT.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in all_rows))

    # ── 집계: (문항, 포인트) 쌍 단위 런간 일치 ──
    by_id: dict[str, list[list[bool]]] = defaultdict(list)
    for r in all_rows:
        by_id[r["id"]].append(r["point_hits"])

    stable = flaky = 0
    flaky_detail: list[str] = []
    for qid, runs_hits in by_id.items():
        if len(runs_hits) < RUNS:      # 일부 런에서 스킵된 문항은 일치율 분모에서 제외
            continue
        for pi, outcomes in enumerate(zip(*runs_hits)):
            if len(set(outcomes)) == 1:
                stable += 1
            else:
                flaky += 1
                flaky_detail.append(
                    f"{qid} 포인트[{pi}] {''.join('O' if o else 'X' for o in outcomes)}"
                    f" — {points_by_id[qid][pi][:60]}")

    total = stable + flaky
    ep_by_run = defaultdict(list)
    for r in all_rows:
        if r["point_hits"]:
            ep_by_run[r["run"]].append(sum(r["point_hits"]) / len(r["point_hits"]))

    print(f"\n포인트 일치율 {stable}/{total} = {stable/total:.3f} · flaky 포인트율 {flaky/total:.3f}"
          f"  (문항 {len(by_id)} × {RUNS}런)")
    print("런별 EPCov 평균:", " ".join(
        f"r{k}={sum(v)/len(v):.3f}" for k, v in sorted(ep_by_run.items())))
    if flaky_detail:
        print("\nflaky 상세 (사람이 원문을 읽을 것 — 수치만 보고 끝내지 않는다):")
        for d in flaky_detail:
            print(" ", d)
    print(f"→ {OUT}")


if __name__ == "__main__":
    asyncio.run(main())
