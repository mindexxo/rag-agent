"""통합 평가 러너 — 검색축 · 생성축(RAGAS) · 인텐트를 한 번에 돌려 요약 리포트.

각 축 스크립트의 compute()를 호출해 헤드라인 지표만 뽑고, 직전 실행과 비교(±)해
콘솔 표로 출력한다. 요약은 eval/results/history/summary_<stamp>.json에 누적 —
이게 프롬프트·모델 변경 시 회귀를 잡는 안전망이자 추적(항목5)의 씨앗.

평가 축 (파이프라인 단계별 독립 채점):
  - 인텐트축(--intent): 관문. 입력을 올바른 경로로 보내는지 — KNOWLEDGE(지식질문)/
      OTHER(대화성)/차단(인젝션·PII)의 분류 정확도. 여기서 틀리면 뒤가 다 무의미.
  - condense축(--condense): 질의 재작성. 후속질문을 독립질문으로 바꿀 때 현재 질문의
      조건·수치를 보존하고 이전 답변을 주입하지 않는지 (케이스×N회 → 일관성 포함). LLM만.
  - 거절축(--refusal): 관문 판정. 근거 없는 질문은 거절하고(no_evidence) 근거 있는
      유도질문은 답하는지(trap). 미거부(환각)/오거부(유용성) 두 방향. prepare+generate 실행.
  - 캐시축(--cache): 시맨틱 캐시가 같은 질문은 재사용, 다른 질문은 안 섞는지. 오탐(오답재생)/
      과잉거절(효용손실) 두 방향. ※정확도 단일수치는 오해 소지 — 보수적 설계라 낮게 나옴. 임베딩+DB만.
  - OTHER축(--other): OTHER 경로(인사·잡답)가 경계를 지키는지 — 역할밖 작업 거절(off_scope)·
      없는기능 환각방지(capability). 경계위반만 규칙채점(품질·톤은 미채점=judge영역). prepare+generate.
  - 검색축(--retrieval): 근거 찾기. 질문에 맞는 문서 청크를 상위로 올리는지 —
      R@5/R@20/Hit@1/MRR. LLM 답변 생성 없이 검색 품질만.
  - 생성축(--ragas): 답 쓰기. 찾아온 근거로 만든 답의 품질 — faithfulness(환각 없나)/
      answer_relevancy(질문에 맞나). judge=OpenAI, 임베딩=TEI.

사용:
  python -m eval.run_all                 # 세 축 전부
  python -m eval.run_all --retrieval     # 검색축만 (DB+TEI, 빠름·키 불필요)
  python -m eval.run_all --intent        # 인텐트만 (LLM)
  python -m eval.run_all --ragas --smoke 0   # 생성축 전체(기본 SMOKE=3)

참고: --ragas 심판은 기본 사내 vLLM(RAGAS_JUDGE=vllm) — 비용·rate limit 없음. 외부 심판은 RAGAS_JUDGE=openai.
      전 축은 사내망(DB·TEI·vLLM) 접근이 필요하다.

생성축(--ragas)은 '미리 생성해둔 답변'(generation_retrieved.jsonl)을 채점만 한다
(생성↔채점 분리 — 비싼 생성과 자주 바뀌는 채점을 떼어둔 설계).
프롬프트·모델을 바꿨다면 답변부터 재생성해야 지표에 반영된다:
  python -m eval.generation        # 답변 재생성 (DB+TEI+vLLM, 무거움)
  python -m eval.run_all --ragas   # 그 답변을 채점
러너는 답변 파일이 오래됐으면 경고만 하고 자동 재생성은 하지 않는다.
"""
import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path

HISTORY = Path(__file__).resolve().parent / "results" / "history"

# (축, 지표키, 표시라벨) — 리포트 행 정의. 높을수록 좋은 지표라 하락이 회귀.
ROWS = [
    ("intent", "accuracy", "인텐트 정확도"),
    ("condense", "accuracy", "질의재작성 정확도"),
    ("refusal", "accuracy", "거절 정확도"),
    ("cache", "accuracy", "캐시히트 정확도"),
    ("other", "accuracy", "OTHER 경계준수"),
    ("retrieval", "recall_at_5", "검색  R@5"),
    ("retrieval", "r5_easy", "검색  R@5(쉬움)"),
    ("retrieval", "r5_medium", "검색  R@5(중간)"),
    ("retrieval", "r5_hard", "검색  R@5(어려움)"),
    ("retrieval", "recall_at_20", "검색  R@20"),
    ("retrieval", "hit_at_1", "검색  Hit@1"),
    ("retrieval", "mrr", "검색  MRR"),
    ("ragas", "faithfulness", "생성  faithfulness"),
    ("ragas", "answer_relevancy", "생성  relevancy"),
]
REGRESSION_EPS = 0.01   # 이보다 크게 떨어지면 회귀 경고


def _warn_stale_generation() -> None:
    """생성축 채점 대상(generation_retrieved.jsonl)의 나이를 확인해 경고.

    RAGAS는 '미리 생성해둔 답변'을 채점만 한다(생성↔채점 분리 설계).
    프롬프트·모델을 바꿨다면 이 파일이 옛 답변이라 변경이 지표에 안 잡힌다 —
    최신 반영은 `python -m eval.generation`으로 답변을 먼저 재생성해야 한다.
    """
    import os
    from datetime import datetime

    gen = Path(__file__).resolve().parent / "results" / "generation_retrieved.jsonl"
    if not gen.exists():
        print("⚠ generation_retrieved.jsonl 없음 — RAGAS 채점 불가. 먼저 `python -m eval.generation` 실행.")
        return
    age_days = (datetime.now().timestamp() - os.path.getmtime(gen)) / 86400
    when = datetime.fromtimestamp(os.path.getmtime(gen)).strftime("%Y-%m-%d %H:%M")
    print(f"ℹ 채점 대상 답변: {when} 생성분 ({age_days:.1f}일 전).")
    if age_days >= 1:
        print("  ⚠ 프롬프트·모델을 바꿨다면 옛 답변이라 변경이 안 잡힙니다.")
        print("    최신 반영: python -m eval.generation  (답변 재생성 후 다시 --ragas)")


def _prev_summary() -> dict | None:
    """직전(가장 최근) 요약 JSON을 반환. 없으면 None."""
    if not HISTORY.exists():
        return None
    files = sorted(HISTORY.glob("summary_*.json"))
    if not files:
        return None
    return json.loads(files[-1].read_text())


# 시계열 뷰 컬럼 — (지표키 경로, 짧은 헤더). ROWS와 같은 축이나 폭 위해 축약.
HIST_COLS = [
    ("intent", "accuracy", "intent"),
    ("condense", "accuracy", "cond"),
    ("refusal", "accuracy", "refus"),
    ("cache", "accuracy", "cache"),
    ("other", "accuracy", "other"),
    ("retrieval", "recall_at_5", "R@5"),
    ("retrieval", "r5_hard", "R@5hard"),
    ("ragas", "faithfulness", "faith"),
]


def _history_view() -> None:
    """history의 모든 요약을 읽어 시계열 매트릭스로 출력 (버전 라벨 포함).

    각 축을 안 돌린 실행은 빈칸(-). 라벨로 '무엇을 바꿨나'를 시각적으로 추적.
    """
    if not HISTORY.exists() or not sorted(HISTORY.glob("summary_*.json")):
        print("이력 없음 — python -m eval.run_all 로 먼저 측정하세요.")
        return
    summaries = [json.loads(f.read_text()) for f in sorted(HISTORY.glob("summary_*.json"))]

    print(f"\n평가 이력 (시계열, {len(summaries)}회)\n")
    header = f"{'날짜':<12}{'라벨':<22}" + "".join(f"{h:>7}" for _, _, h in HIST_COLS)
    print(header)
    print("─" * len(header))
    for s in summaries:
        stamp = (s.get("stamp") or "")[5:]   # 'MM-DD HH:MM' (연도 생략)
        label = (s.get("label") or "")[:20]
        cells = ""
        for axis, key, _ in HIST_COLS:
            v = _get(s, axis, key)
            cells += f"{v:>7.3f}" if isinstance(v, (int, float)) else f"{'-':>7}"
        print(f"{stamp:<12}{label:<22}{cells}")


def _get(summary: dict | None, axis: str, key: str):
    if not summary:
        return None
    return (summary.get(axis) or {}).get(key)


def _report(cur: dict, prev: dict | None) -> None:
    stamp = cur["stamp"]
    prev_note = f"(직전: {prev['stamp']} 대비)" if prev else "(첫 실행 — 비교 대상 없음)"
    print(f"\n평가 요약 — {stamp}  {prev_note}")
    if cur.get("label"):
        print(f"라벨: {cur['label']}")
    print("─" * 52)
    warnings = []
    for axis, key, label in ROWS:
        axis_data = cur.get(axis)
        if axis_data is None:
            continue  # 축 자체를 안 돌림 → 행 생략
        if key not in axis_data:
            continue  # 축은 돌렸으나 이 세부지표는 해당 없음(예: 빈 난이도 그룹) → 행 생략
        v = axis_data[key]
        if v is None:
            # 키는 있는데 값이 None = 진짜 측정 실패(RAGAS NaN 등) — '해당 없음'과 구분해 명시
            print(f"{label:<20}{'실패':>8}   (측정 불가)")
            warnings.append(f"{label} 측정 실패")
            continue
        pv = _get(prev, axis, key)
        if pv is None:
            delta = "     ―   "
        else:
            d = v - pv
            mark = " ⚠" if d < -REGRESSION_EPS else ""
            delta = f"({d:+.3f}){mark}"
            if mark:
                warnings.append(f"{label} {d:+.3f}")
        print(f"{label:<20}{v:>8.3f}   {delta}")
    print("─" * 52)
    if warnings:
        print("⚠ 회귀 감지:")
        for w in warnings:
            print(f"   - {w}")
    else:
        print("회귀 없음.")


async def _run_async(name: str):
    """async compute() 축(retrieval·intent)을 실행."""
    if name == "retrieval":
        from eval.retrieval_v2 import compute
        r = await compute()
        # 전체 평균 + 난이도 층화(easy/medium/hard의 R@5).
        # 문항 없는 난이도 그룹은 키를 아예 넣지 않는다 → _report가 '해당 없음'으로 행 생략
        # (None으로 넣으면 '측정 실패'로 오인돼 가짜 회귀가 뜸).
        d = r["by_difficulty"]
        out = dict(r["overall"])
        for grp, key in (("easy", "r5_easy"), ("medium", "r5_medium"), ("hard", "r5_hard")):
            if grp in d:
                out[key] = d[grp]
        return out
    if name == "intent":
        from eval.intent import compute
        r = await compute()
        return {"accuracy": r["accuracy"]}
    if name == "condense":
        from eval.condense import compute
        r = await compute()
        return {"accuracy": r["accuracy"], "flaky": r["flaky"]}
    if name == "refusal":
        from eval.refusal import compute
        r = await compute()
        return {"accuracy": r["accuracy"],
                "false_answer": r["false_answer"], "false_refusal": r["false_refusal"]}
    if name == "cache":
        from eval.cache_eval import compute
        r = await compute()
        return {"accuracy": r["accuracy"],
                "false_hit": r["false_hit"], "false_miss": r["false_miss"]}
    if name == "other":
        from eval.other_eval import compute
        r = await compute()
        return {"accuracy": r["accuracy"], "flaky": r["flaky"]}
    raise ValueError(name)


def main() -> None:
    ap = argparse.ArgumentParser(description="통합 평가 러너")
    ap.add_argument("--retrieval", action="store_true", help="검색축")
    ap.add_argument("--ragas", action="store_true", help="생성축(RAGAS)")
    ap.add_argument("--intent", action="store_true", help="인텐트")
    ap.add_argument("--condense", action="store_true", help="질의 재작성")
    ap.add_argument("--refusal", action="store_true", help="거절 정확성")
    ap.add_argument("--cache", action="store_true", help="캐시 히트 정확성")
    ap.add_argument("--other", action="store_true", help="OTHER 경계 준수")
    ap.add_argument("--smoke", type=int, default=None, help="RAGAS 샘플 수(0=전체). 미지정 시 SMOKE 환경변수")
    ap.add_argument("--label", type=str, default="", help="이 실행의 라벨 (무엇을 바꿨나 — 이력 추적용)")
    ap.add_argument("--history", action="store_true", help="측정 없이 과거 이력을 시계열 표로 출력")
    args = ap.parse_args()

    # 이력 조회 모드 — 측정 안 하고 시계열만 보여주고 종료
    if args.history:
        _history_view()
        return

    # 아무 플래그도 없으면 전 축
    run_all = not (args.retrieval or args.ragas or args.intent or args.condense or args.refusal or args.cache or args.other)
    do = {
        "retrieval": run_all or args.retrieval,
        "ragas": run_all or args.ragas,
        "intent": run_all or args.intent,
        "condense": run_all or args.condense,
        "refusal": run_all or args.refusal,
        "cache": run_all or args.cache,
        "other": run_all or args.other,
    }

    summary: dict = {"stamp": datetime.now().strftime("%Y-%m-%d %H:%M"), "label": args.label}

    # 검색축·인텐트·condense·거절 (async) — 순차 실행 (DB/TEI/LLM 부하 분산)
    if do["intent"]:
        print("▶ 인텐트 채점 중…")
        summary["intent"] = asyncio.run(_run_async("intent"))
    if do["condense"]:
        print("▶ 질의재작성(condense) 채점 중…")
        summary["condense"] = asyncio.run(_run_async("condense"))
    if do["refusal"]:
        print("▶ 거절 정확성 채점 중…")
        summary["refusal"] = asyncio.run(_run_async("refusal"))
    if do["cache"]:
        print("▶ 캐시 히트 정확성 채점 중…")
        summary["cache"] = asyncio.run(_run_async("cache"))
    if do["other"]:
        print("▶ OTHER 경계 준수 채점 중…")
        summary["other"] = asyncio.run(_run_async("other"))
    if do["retrieval"]:
        print("▶ 검색축 채점 중…")
        summary["retrieval"] = asyncio.run(_run_async("retrieval"))
    # 생성축 (sync, 느림·키 필요) — 마지막
    if do["ragas"]:
        _warn_stale_generation()
        print("▶ 생성축(RAGAS) 채점 중… (느림)")
        from eval.ragas_eval import compute as ragas_compute
        r = ragas_compute(smoke=args.smoke)
        summary["ragas"] = {"faithfulness": r["faithfulness"],
                            "answer_relevancy": r["answer_relevancy"], "n": r["n"]}

    prev = _prev_summary()   # 이번 저장 전에 읽어야 '직전'이 맞음
    HISTORY.mkdir(parents=True, exist_ok=True)
    # 초 단위까지 — 같은 분 재실행 시 파일명 충돌로 history가 덮어써지는 것 방지
    fname = HISTORY / f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    fname.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")

    _report(summary, prev)
    print(f"\n→ 요약 저장: {fname}")


if __name__ == "__main__":
    main()
