"""성능 수치 리포트 (#97 ③) — Phoenix 스팬에서 단계별 지연 추출. 읽기 전용·GPU 무부하.

무엇을 뽑나 (발표 성능 1장의 재료):
- 턴 완료 시간 p50/p95 — 루트 kms.query 스팬 duration (kms.latency_ms와 같은 턴을
  스팬 시각으로 잰 값. DB Message.latency_ms와 달리 스팬은 단계 분해와 같은 표본이라
  합이 맞는다).
- 단계별 지연 p50/p95 — classify_and_guard / condense / retrieve / rerank / generate
  자식 스팬 duration. "지연이 어디서 나는가"를 한 표로.
- 턴 모양별 분리 — 자식 스팬 구성(children set)이 곧 경로다: generate 없는 done 턴은
  캐시 히트·차단·OTHER 계열. kms.status·kms.intent 속성으로 나눠 센다.

**표본 주의(발표 인용 시 함께 적을 것)**: Phoenix의 스팬은 개발계 실사용+테스트 트래픽이고,
eval 풀런이 도는 동안의 턴은 GPU 경합으로 지연이 부풀어 있다. --since/--until로 그런
시간대를 잘라내고 쓰라. 첫 토큰(TTFT)은 스팬에 없다 — eval/perf_probe.py(실측 프로브)가
별도로 잰다.

의존: Phoenix HTTP(OTEL_ENDPOINT의 호스트)만. DB·LLM·TEI 불필요.
실행: python -m eval.perf_report [--since 2026-08-20] [--until 2026-08-29]
"""
import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import httpx

from config import settings

RESULT_DIR = Path("eval/results")
OUT_MD = RESULT_DIR / "perf_report.md"
OUT_RAW = RESULT_DIR / "perf_spans.jsonl"

STAGES = ["classify_and_guard", "condense", "retrieve", "rerank", "generate"]


def phoenix_base() -> str:
    """OTEL_ENDPOINT(…/v1/traces)에서 Phoenix API 베이스를 파생 — 설정 중복 방지."""
    ep = settings.otel_endpoint
    if not ep:
        raise SystemExit("OTEL_ENDPOINT 미설정 — Phoenix 없이 뽑을 스팬이 없다.")
    return ep.split("/v1/")[0]


def fetch_spans(base: str) -> list[dict]:
    """전 스팬 페이지네이션 수집. 프로젝트는 'default' 하나뿐(계측이 단일 서비스)."""
    with httpx.Client(timeout=30) as client:
        projects = client.get(f"{base}/v1/projects").json()["data"]
        pid = next(p["id"] for p in projects if p["name"] == "default")
        spans, cursor = [], None
        while True:
            params = {"limit": 100} | ({"cursor": cursor} if cursor else {})
            page = client.get(f"{base}/v1/projects/{pid}/spans", params=params).json()
            spans.extend(page["data"])
            cursor = page.get("next_cursor")
            if not cursor:
                return spans


def _dur_ms(s: dict) -> float:
    st = datetime.fromisoformat(s["start_time"])
    et = datetime.fromisoformat(s["end_time"])
    return (et - st).total_seconds() * 1000


def _pct(values: list[float], q: float) -> float:
    """단순 최근접 순위 백분위 — 표본 수백 건에 보간 정밀도는 과잉이다."""
    if not values:
        return float("nan")
    vs = sorted(values)
    return vs[min(len(vs) - 1, round(q * (len(vs) - 1)))]


def _fmt_row(name: str, values: list[float]) -> str:
    return (f"| {name} | {len(values)} | {_pct(values, 0.5):,.0f} | "
            f"{_pct(values, 0.95):,.0f} | {max(values):,.0f} |")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", help="이 날짜(포함, ISO) 이전 스팬 제외 — 부하 오염 시간대 컷")
    ap.add_argument("--until", help="이 날짜(미포함, ISO) 이후 스팬 제외")
    args = ap.parse_args()

    spans = fetch_spans(phoenix_base())
    if args.since:
        spans = [s for s in spans if s["start_time"] >= args.since]
    if args.until:
        spans = [s for s in spans if s["start_time"] < args.until]

    by_trace: dict[str, list[dict]] = defaultdict(list)
    for s in spans:
        by_trace[s["context"]["trace_id"]].append(s)

    # 경로 분리 (#21 참고 반영): retrieve 자식 유무가 곧 경로다 — KNOWLEDGE(검색+생성)와
    # OTHER(검색 스킵, 인사·회상류)는 지연 분포가 달라서 섞으면 "지식 질문 기준 X초"를 못 말한다.
    kn_total: list[float] = []               # KNOWLEDGE 완주 턴 총 시간
    ot_total: list[float] = []               # OTHER 완주 턴 총 시간
    stage_ms: dict[str, list[float]] = defaultdict(list)      # KNOWLEDGE 턴의 단계 분해
    ot_generate: list[float] = []            # OTHER 턴의 생성 시간 (참고용)
    shapes = Counter()                       # (status, intent, 자식 구성) 분포 — 경로 파악용
    nogen_total: dict[str, list[float]] = defaultdict(list)   # 비생성 done 턴 총 시간 (모양별)

    rows_raw = []
    for spans_t in by_trace.values():
        root = next((s for s in spans_t if s["parent_id"] is None and s["name"] == "kms.query"), None)
        if root is None:
            continue
        children = sorted({s["name"] for s in spans_t if s["parent_id"] is not None})
        attrs = root.get("attributes", {})
        status, intent = attrs.get("kms.status"), attrs.get("kms.intent")
        shape = f"status={status} intent={intent} children={'+'.join(children) or '-'}"
        shapes[shape] += 1
        total = _dur_ms(root)
        rows_raw.append({"trace_id": root["context"]["trace_id"], "start": root["start_time"],
                         "total_ms": total, "status": status, "intent": intent,
                         "children": children,
                         "stages": {s["name"]: _dur_ms(s) for s in spans_t if s["name"] in STAGES}})
        if "generate" in children and status == "done":
            if "retrieve" in children:
                kn_total.append(total)
                for s in spans_t:
                    if s["name"] in STAGES:
                        stage_ms[s["name"]].append(_dur_ms(s))
            else:
                ot_total.append(total)
                ot_generate.extend(_dur_ms(s) for s in spans_t if s["name"] == "generate")
        elif status == "done":
            nogen_total[shape].append(total)

    RESULT_DIR.mkdir(exist_ok=True)
    OUT_RAW.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows_raw))

    lines = [f"# 성능 리포트 (Phoenix 스팬, {args.since or '전체'}~{args.until or '현재'})",
             f"\n트레이스 {len(by_trace)}건 / 완주 턴: KNOWLEDGE {len(kn_total)}·OTHER {len(ot_total)}\n",
             "## KNOWLEDGE 완주 턴 (검색+생성) — 단계별 지연 (ms)\n",
             "| 구간 | n | p50 | p95 | max |", "|---|---|---|---|---|",
             _fmt_row("**턴 전체**", kn_total)]
    lines += [_fmt_row(st, stage_ms[st]) for st in STAGES if stage_ms[st]]
    lines += ["\n## OTHER 완주 턴 (검색 스킵 — 인사·회상류) — (ms)\n",
              "| 구간 | n | p50 | p95 | max |", "|---|---|---|---|---|"]
    if ot_total:
        lines += [_fmt_row("턴 전체", ot_total), _fmt_row("generate", ot_generate)]
    lines += ["\n## 비생성 done 턴 (캐시 히트·차단·거절 계열) — 총 시간 (ms)\n",
              "| 모양 | n | p50 | p95 | max |", "|---|---|---|---|---|"]
    lines += [_fmt_row(shape, vs) for shape, vs in sorted(nogen_total.items())]
    lines += ["\n## 턴 모양 분포 (경로 검산용)\n"]
    lines += [f"- {n:>4}× {shape}" for shape, n in shapes.most_common()]
    lines += ["\n주의: TTFT(첫 토큰)는 이 표에 없다 — eval/perf_probe.py 실측으로 별도 확보.",
              "표본은 개발계 트래픽이다 — eval 풀런 시간대 포함 여부를 --since/--until로 통제했는지 확인."]

    OUT_MD.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\n→ {OUT_MD} · 원자료 {OUT_RAW}")


if __name__ == "__main__":
    main()
