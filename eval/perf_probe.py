"""TTFT(첫 토큰) 실측 프로브 (#97 ③) — SSE 클라이언트로 서버를 직접 때려 잰다.

Phoenix 스팬으로 못 재는 것만 잰다: 첫 토큰 지연(요청→첫 delta)과 캐시 히트 응답.
generate 스팬은 스트리밍 루프 전체를 감싸서 첫 delta 시각이 어디에도 안 남는다 —
그래서 이 축만은 스팬 추출(eval/perf_report.py)이 아니라 실측이다.

측정 설계:
- 문항은 골드셋에서 타입 균등 표본(N=30 기본) — 재현 가능(무작위 없음, _smoke_sample).
- 각 문항을 **새 대화로 2회 연속** 질의: 1회차 = 콜드(전체 파이프라인) TTFT/완료,
  2회차 = 같은 질문 재질의 → 답변 캐시 히트 경로의 응답 시간. 단 1회차가 근거없음
  거절이면 캐시 미적재라 2회차도 콜드다 — done의 citations 유무로 구분해 따로 센다.
- 직렬 1콜씩(동시성 1) — 이 축은 "한 상담사의 체감"이다. 동시 부하는 #101의 몫.

**GPU가 한가할 때 돌릴 것** — eval 풀런과 겹치면 수치가 부풀어 발표에 못 쓴다.
의존: 로컬 백엔드(uvicorn) + vLLM + TEI + DB. 실행: python -m eval.perf_probe
  [--base-url http://localhost:8000] [--tenant summers] [--n 30]
"""
import argparse
import asyncio
import json
import time
from pathlib import Path

import httpx

from eval.generation import GEN_TYPES, GOLD, _smoke_sample, row_tenant

RESULT_DIR = Path("eval/results")
OUT = RESULT_DIR / "perf_probe.jsonl"


def _pct(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    vs = sorted(values)
    return vs[min(len(vs) - 1, round(q * (len(vs) - 1)))]


async def probe_turn(client: httpx.AsyncClient, base: str, tenant: str, query: str) -> dict:
    """새 대화 1턴 SSE — 요청→첫 delta, 요청→done + 스트리밍 품질 지표 3종 (#21 규칙 차용).

    - answer_chars: 답변 길이 — 지연은 길이에 비례하므로 이것 없이는 p95를 해석할 수 없다
      ("보고 공통 규칙: 답변 토큰 수 동시 기록"의 문자 단위 근사).
    - gen_cps: 요청당 생성 속도(자/초) — 첫 delta→마지막 delta 구간. 스트리밍 체감 한 숫자.
    - max_gap_ms: delta 간 최대 간격(ITL stall) — "중간에 멈칫하는가".
    """
    t0 = time.monotonic()
    ttft_ms = None
    last_delta_t = None
    max_gap_ms = 0.0
    answer_chars = 0
    done: dict = {}
    async with client.stream(
        "POST", f"{base}/kms/query",
        headers={"X-Tenant-Id": tenant},
        json={"query": query},
    ) as resp:
        resp.raise_for_status()
        event = None
        async for line in resp.aiter_lines():
            if line.startswith("event:"):
                event = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                if event == "delta":
                    now = time.monotonic()
                    if ttft_ms is None:
                        ttft_ms = (now - t0) * 1000
                    else:
                        max_gap_ms = max(max_gap_ms, (now - last_delta_t) * 1000)
                    last_delta_t = now
                    answer_chars += len(json.loads(line.split(":", 1)[1]).get("text", ""))
                elif event == "done":
                    done = json.loads(line.split(":", 1)[1])
    stream_s = (last_delta_t - t0) * 1000 / 1000 - (ttft_ms or 0) / 1000 if last_delta_t else 0
    return {"ttft_ms": ttft_ms, "total_ms": (time.monotonic() - t0) * 1000,
            "answer_chars": answer_chars,
            "gen_cps": answer_chars / stream_s if stream_s > 0.1 else None,   # 한 delta짜리 답은 속도 무의미
            "max_gap_ms": max_gap_ms or None,
            "citations": done.get("citations", []), "finish": done.get("finish_reason")}


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--sleep", type=float, default=1.0, help="턴 간 간격(s) — limiter·GPU 여유")
    args = ap.parse_args()

    gold = [json.loads(l) for l in GOLD.read_text().splitlines() if l.strip()]
    sample = _smoke_sample([g for g in gold if g["type"] in GEN_TYPES], args.n)

    rows = []
    async with httpx.AsyncClient(timeout=120) as client:
        for i, g in enumerate(sample, 1):
            tenant = row_tenant(g)
            cold = await probe_turn(client, args.base_url, tenant, g["query"])
            await asyncio.sleep(args.sleep)
            warm = await probe_turn(client, args.base_url, tenant, g["query"])
            await asyncio.sleep(args.sleep)
            # 캐시 히트 판별: 콜드가 인용을 남겼을 때만 2회차가 히트 후보다 (모듈 docstring).
            # 서버가 히트 여부를 직접 노출하지 않으므로 "웜이 TTFT 없이(스트리밍 없이)
            # 즉시 done"인지도 함께 기록해 사람이 검산할 수 있게 남긴다.
            rows.append({"id": g["id"], "type": g["type"], "tenant": tenant,
                         "cold": cold, "warm": warm,
                         "cacheable": bool(cold["citations"])})
            print(f"[{i}/{len(sample)}] {g['id']}: cold ttft={cold['ttft_ms'] and f'{cold['ttft_ms']:.0f}ms'}"
                  f" total={cold['total_ms']:.0f}ms / warm total={warm['total_ms']:.0f}ms")

    RESULT_DIR.mkdir(exist_ok=True)
    OUT.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows))

    cold_ttft = [r["cold"]["ttft_ms"] for r in rows if r["cold"]["ttft_ms"] is not None]
    cold_total = [r["cold"]["total_ms"] for r in rows]
    warm_hit = [r["warm"]["total_ms"] for r in rows if r["cacheable"]]
    chars = [r["cold"]["answer_chars"] for r in rows]
    cps = [r["cold"]["gen_cps"] for r in rows if r["cold"]["gen_cps"]]
    gaps = [r["cold"]["max_gap_ms"] for r in rows if r["cold"]["max_gap_ms"]]
    print(f"\n콜드 TTFT   p50 {_pct(cold_ttft, .5):,.0f}ms · p95 {_pct(cold_ttft, .95):,.0f}ms (n={len(cold_ttft)})")
    print(f"콜드 완료   p50 {_pct(cold_total, .5):,.0f}ms · p95 {_pct(cold_total, .95):,.0f}ms (n={len(cold_total)})")
    print(f"캐시 후보 웜 p50 {_pct(warm_hit, .5):,.0f}ms · p95 {_pct(warm_hit, .95):,.0f}ms (n={len(warm_hit)})")
    print(f"답변 길이   p50 {_pct(chars, .5):,.0f}자 (지연 해석용 — 길이 비례)")
    print(f"생성 속도   p50 {_pct(cps, .5):,.0f}자/초 · p5 {_pct(cps, .05):,.0f}자/초 (n={len(cps)})")
    print(f"delta 최대 간격 p50 {_pct(gaps, .5):,.0f}ms · p95 {_pct(gaps, .95):,.0f}ms (stall 감시)")
    print(f"→ {OUT}")


if __name__ == "__main__":
    asyncio.run(main())
