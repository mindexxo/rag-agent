"""동시 부하 실측 (#101) — 계단식 N으로 지연 곡선·병목·앱 방어를 잰다.

#97 perf_probe가 단일 요청 체감(TTFT)을 쟀다면, 여기는 **동시 N명일 때** 지연이
어떻게 무너지는지다. 파일럿 규모 산정·8B 전환 판단의 근거.

두 모드가 서로 다른 한계를 잰다 (limiter 상한이 테넌트당 10이라 갈린다, config):
- single : 한 테넌트에 N 동시 → N>10이면 limiter가 429로 막는다. **앱 레벨 방어 검증**.
- spread : 6테넌트에 고르게 분산 → limiter를 우회해 vLLM/TEI 실제 처리 한계를 본다.
           선행 실측(#101 코멘트)의 "KV 상한 ~32, 60에서 처리량 반토막"을 앱 경유로 재확인.

각 요청은 SSE로 실서버(/kms/query)를 때린다 — prepare→검색→생성 전 구간이 부하에 든다.
질문은 골드셋에서 **요청마다 유니크**로 뽑아 답변 캐시 히트를 배제한다(히트하면 GPU를
안 쓰므로 부하가 안 걸린다). 부족하면 접미 태그로 변형해 유니크를 보장한다.

**GPU가 한가할 때 돌릴 것** — 다른 eval·실트래픽과 겹치면 곡선이 오염된다.
의존: 로컬 백엔드(uvicorn) + vLLM + TEI + DB + Redis(limiter). 실행:
  python -m eval.load_test --base-url http://localhost:8877 --mode spread --steps 1,4,8,16,24,32
"""
import argparse
import asyncio
import json
import time
from pathlib import Path

import httpx

from eval.generation import GEN_TYPES, GOLD, row_tenant

RESULT_DIR = Path("eval/results")
OUT = RESULT_DIR / "load_test.jsonl"
V2_TENANTS = ["summers", "homeplus", "adererror", "aromanica", "goodpeople", "harim"]


def _pct(vals: list[float], q: float) -> float:
    if not vals:
        return float("nan")
    s = sorted(vals)
    return s[min(len(s) - 1, round(q * (len(s) - 1)))]


async def _one_request(client: httpx.AsyncClient, base: str, tenant: str, query: str) -> dict:
    """SSE 1건 — (상태, TTFT, 완료시간) 수집. 429·타임아웃·오류를 상태로 구분한다."""
    t0 = time.monotonic()
    ttft = None
    try:
        async with client.stream("POST", f"{base}/kms/query",
                                 headers={"X-Tenant-Id": tenant},
                                 json={"query": query}) as resp:
            if resp.status_code == 429:
                return {"status": "429_limited", "ttft_ms": None,
                        "total_ms": (time.monotonic() - t0) * 1000}
            if resp.status_code != 200:
                return {"status": f"http_{resp.status_code}", "ttft_ms": None,
                        "total_ms": (time.monotonic() - t0) * 1000}
            event = None
            async for line in resp.aiter_lines():
                if line.startswith("event:"):
                    event = line.split(":", 1)[1].strip()
                elif line.startswith("data:") and event == "delta" and ttft is None:
                    ttft = (time.monotonic() - t0) * 1000
            return {"status": "ok", "ttft_ms": ttft, "total_ms": (time.monotonic() - t0) * 1000}
    except (httpx.TimeoutException, httpx.HTTPError) as exc:
        return {"status": f"error_{type(exc).__name__}", "ttft_ms": None,
                "total_ms": (time.monotonic() - t0) * 1000}


def _assign(mode: str, n: int, idx: int) -> str:
    """요청 idx의 테넌트 — single은 전부 summers, spread는 6테넌트 순환."""
    return "summers" if mode == "single" else V2_TENANTS[idx % len(V2_TENANTS)]


async def _step(client, base, mode: str, n: int, queries: list[str]) -> dict:
    """동시 N 요청을 한 번에 발사하고 지연 분포를 집계한다 (한 계단)."""
    t0 = time.monotonic()
    results = await asyncio.gather(*(
        _one_request(client, base, _assign(mode, n, i), queries[i]) for i in range(n)))
    wall = time.monotonic() - t0
    ok = [r for r in results if r["status"] == "ok"]
    limited = sum(1 for r in results if r["status"] == "429_limited")
    errors = sum(1 for r in results if r["status"].startswith(("error", "http")))
    ttft = [r["ttft_ms"] for r in ok if r["ttft_ms"] is not None]
    total = [r["total_ms"] for r in ok]
    return {"mode": mode, "n": n, "wall_s": round(wall, 2),
            "ok": len(ok), "limited_429": limited, "errors": errors,
            "throughput_rps": round(len(ok) / wall, 2) if wall else 0.0,
            "ttft_p50": _pct(ttft, .5), "ttft_p95": _pct(ttft, .95),
            "total_p50": _pct(total, .5), "total_p95": _pct(total, .95)}


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8877")
    ap.add_argument("--mode", choices=["single", "spread"], default="spread")
    ap.add_argument("--steps", default="1,4,8,16,24,32",
                    help="쉼표로 구분한 동시 N 계단")
    ap.add_argument("--rounds", type=int, default=1, help="계단당 반복(분포 안정화)")
    args = ap.parse_args()
    steps = [int(s) for s in args.steps.split(",")]

    gold = [json.loads(l) for l in GOLD.read_text().splitlines() if l.strip()
            if json.loads(l)["type"] in GEN_TYPES]
    # 유니크 질문 풀 — 필요 수만큼, 부족하면 접미 태그로 변형(캐시 히트 배제)
    need = sum(steps) * args.rounds   # 계단마다 유니크를 누적 소비하므로 합계 (max 아님)
    pool = [g["query"] for g in gold]
    queries = [(pool[i % len(pool)] if i < len(pool)
                else f"{pool[i % len(pool)]} (문의 {i})") for i in range(need)]

    RESULT_DIR.mkdir(exist_ok=True)
    rows = []
    # 넉넉한 타임아웃 — 과부하 구간의 대기까지 관찰(끊지 않는다). vLLM read 300초와 정합.
    async with httpx.AsyncClient(timeout=httpx.Timeout(320.0, connect=5.0)) as client:
        qi = 0
        print(f"[부하 실측] mode={args.mode} steps={steps} rounds={args.rounds}\n")
        print(f"{'N':>4}{'ok':>5}{'429':>5}{'err':>5}{'RPS':>8}"
              f"{'TTFTp50':>9}{'TTFTp95':>9}{'완료p50':>9}{'완료p95':>9}")
        for n in steps:
            for _ in range(args.rounds):
                step = await _step(client, args.base_url, args.mode, n, queries[qi:qi + n])
                qi += n
                rows.append(step)
                print(f"{n:>4}{step['ok']:>5}{step['limited_429']:>5}{step['errors']:>5}"
                      f"{step['throughput_rps']:>8.2f}"
                      f"{step['ttft_p50']:>9.0f}{step['ttft_p95']:>9.0f}"
                      f"{step['total_p50']:>9.0f}{step['total_p95']:>9.0f}")
                await asyncio.sleep(3)   # 계단 사이 GPU KV 반납 여유

    OUT.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    print(f"\n→ {OUT}")
    print("주의: single 모드의 429는 앱 방어(테넌트 상한 10)가 작동한 것 — 오류가 아니다.")


if __name__ == "__main__":
    asyncio.run(main())
