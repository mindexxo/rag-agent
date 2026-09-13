"""캐시 재사용 판정기 단독 측정 — 프롬프트를 바꿀 때 이것부터 돌린다 (#153).

eval/cache_eval.py와의 분업:
- cache_eval = 전체 파이프라인(floor→docset→판정기). DB·TEI·vLLM 전부 필요, 느리다.
  **판정 아닌 관문이 막은 미스까지 섞이므로** 프롬프트 변경의 효과만 보기엔 잡음이 있다.
- 이것 = 판정기만. **vLLM만 있으면 된다**(DB·TEI 불필요). 프롬프트를 고치고 몇 분 안에
  "거절→승인 뒤집힘이 위험 방향으로 났는가"를 본다.

두 축을 잰다:
1. 40쌍 회귀 (eval/cache_set_v1.jsonl) — 이 셋은 자족 단일턴 질의라 원문==재작성문으로 넣는다.
   운영에서도 단일턴은 condense를 건너뛰므로 이게 실제 형태다.
2. 방어 케이스 (_GUARD) — **원문이 같은데 재작성문이 다른** 유형. 40쌍에 없는 축이고,
   원문 추가(#153)가 위험한 방향으로 작동하는지를 보는 자리다. 멀티턴 실로그가 쌓이면
   여기에 실제 쌍을 옮겨 넣어라.

실행: python -m eval.cache_judge_probe [반복횟수]   (기본 3회 — 판정 흔들림을 보려면 3 이상)

**"원문 없음"은 도입 전 프롬프트가 아니다** — 같은 새 템플릿에 원문 자리만 '(없음)'으로 채운
것이다. 이 구분이 실측에서 결정적이었다(아래).

실측 (2026-09-13, Qwen3-14B, 5회 반복). 도입 전 템플릿(원문 줄이 아예 없는 2슬롯)까지 넣어
셋을 갈랐다 — "원문 내용"과 "템플릿 구조" 중 무엇이 판정을 바꾸는지 분리하기 위해서다:

| | 40쌍 기대값 전회 일치 | 위험 방향 | 흔들림 | 실사례 26496 |
|---|---|---|---|---|
| (a) 도입 전 템플릿(2슬롯) | 39/40 | 0 | 0 | 거절 |
| (b) 새 템플릿 · 원문 '(없음)' | 40/40 | 0 | 0 | **거절** |
| (c) 새 템플릿 · 원문 채움 | 40/40 | 0 | 0 | **승인** |

읽는 법 — 두 효과가 별개다:
- **원문 내용**이 실사례 26496을 구제한다. (b)가 거절이라는 것이 증거다. 템플릿만 바꿔서는
  안 되고, 원문을 실제로 저장·전달해야 한다. 이것이 #153의 본체다.
- **템플릿 구조**(원문 슬롯 + 질문 묶음 사이 빈 줄)가 40쌍 중 1건을 부수적으로 고친다
  (cache_para_deep_aromanica_refill_rate — (a)에서 유일하게 틀리던 쌍이 (b)에서 이미 승인).
  원문 내용과 무관한 효과이므로 #153의 근거로 쓰지 마라.
"""
import asyncio
import json
import sys
from pathlib import Path

from rag.clients import shared_llm
from rag.llm_schemas import ReuseJudgment, acomplete_validated
from rag.prompts import build_cache_reuse_judge_messages

GOLD = Path(__file__).resolve().parent / "cache_set_v1.jsonl"
CONCURRENCY = 6          # vLLM 안전 운영선(동시 24, #101)보다 훨씬 아래 — 다른 작업과 겹쳐도 안전

# 원문이 같은데 재작성문이 다른 쌍 — 원문 추가가 오승인을 부르는지 보는 방어 케이스.
# (라벨, 캐시원문, 캐시재작성문, 신규원문, 신규재작성문, 기대값)
_GUARD = [
    ("생략형 — 연차 3일 vs 출장 3일",
     "3일이요", "연차휴가를 3일 연속으로 사용할 수 있나요?",
     "3일이요", "출장 기간이 3일인 경우 출장여비는 얼마인가요?", False),
    ("부정형 — 불량 인정 vs 인정 안 됨",
     "프린트 크랙", "세탁 후 프린트 크랙이 불량으로 인정되는 경우는?",
     "프린트 크랙", "세탁 후 프린트 크랙이 불량으로 인정 안 되는 경우는?", False),
    ("포괄 대 상세 — 환불 규정 vs 개봉 화장품",
     "환불", "환불 규정은 어떻게 되나요?",
     "환불", "개봉한 화장품도 환불이 되나요?", False),
    # 실사례: 개발계 대화 26496 — 같은 입력이 이력 차이로 다르게 재작성됐다. 이 이슈의 출발점.
    ("실사례 26496 — 같은 입력, 재작성만 갈림",
     "인사 규정", "인사 규정은 어떻게 되나요?",
     "인사 규정", "인사 규정의 주요 내용은 무엇인가요?", True),
    # 도입 전 캐시 행: 원문 컬럼이 NULL이다. 현행과 같은 판정이어야 한다(동작 무변경 보장).
    ("도입 전 행 — 캐시 쪽 원문 없음",
     None, "인사 규정은 어떻게 되나요?",
     "인사 규정", "인사 규정의 주요 내용은 무엇인가요?", False),
]

_sem = asyncio.Semaphore(CONCURRENCY)


async def _judge(cs: str, ns: str, co: str | None, no: str | None) -> bool | None:
    """판정 1회. None은 호출·스키마 실패 — 운영은 이것도 거절로 수렴시키지만
    여기선 '거절'과 구분해 보여준다(프롬프트가 스키마를 깨뜨렸는지 알아야 하므로)."""
    async with _sem:
        try:
            j = await acomplete_validated(
                shared_llm, build_cache_reuse_judge_messages(cs, ns, co, no), ReuseJudgment)
            return j.same_answer
        except Exception:
            return None


def _fmt(v: list[bool | None]) -> str:
    return "".join("T" if x else ("F" if x is False else "?") for x in v)


async def _regression(n: int) -> dict:
    """40쌍 — 원문 없이(현행 재현) vs 원문 포함. 같은 실행 안에서 재야 모델 상태가 같다."""
    pairs = [json.loads(l) for l in GOLD.read_text().splitlines() if l.strip()]

    async def one(p):
        bare = await asyncio.gather(*[_judge(p["q1"], p["q2"], None, None) for _ in range(n)])
        orig = await asyncio.gather(*[_judge(p["q1"], p["q2"], p["q1"], p["q2"]) for _ in range(n)])
        return p, bare, orig

    rows = await asyncio.gather(*[one(p) for p in pairs])

    risky, flips = [], []
    print(f'{"id":<44} {"kind":<12} {"기대":<6} {"원문없음":<7} {"원문포함"}')
    print("-" * 88)
    for p, bare, orig in rows:
        mark = ""
        if not any(bare) and any(orig):
            flips.append(p["id"])
            mark = "  ← F→T" + (" ★위험" if p["should_hit"] is False else "")
            if p["should_hit"] is False:
                risky.append(p)
        elif any(bare) and not any(orig):
            flips.append(p["id"]); mark = "  ← T→F"
        elif len(set(orig)) > 1:
            mark = "  ← 흔들림"
        print(f'{p["id"]:<44} {p["kind"]:<12} {str(p["should_hit"]):<6} '
              f'{_fmt(bare):<7} {_fmt(orig)}{mark}')

    agree = lambda key: sum(1 for p, b, o in rows
                            if all(x is p["should_hit"] for x in (b if key == "bare" else o)))
    waver = lambda key: sum(1 for p, b, o in rows if len(set(b if key == "bare" else o)) > 1)
    return {"n": len(rows), "risky": len(risky), "flips": flips,
            "agree_bare": agree("bare"), "agree_orig": agree("orig"),
            "waver_bare": waver("bare"), "waver_orig": waver("orig")}


async def _guards(n: int) -> int:
    """방어 케이스 — 원문 포함 형태로만 잰다(현행엔 원문 자리가 없다)."""
    print(f'\n{"방어 케이스":<40} {"기대":<6} {"원문포함"}')
    print("-" * 64)
    bad = 0
    for label, co, cs, no, ns, exp in _GUARD:
        v = await asyncio.gather(*[_judge(cs, ns, co, no) for _ in range(n)])
        ok = all(x is exp for x in v)
        bad += not ok
        print(f'{label:<40} {str(exp):<6} {_fmt(v)}{"" if ok else "  ★기대 불일치"}')
    return bad


async def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    r = await _regression(n)
    bad = await _guards(n)
    print(f'\n[40쌍 회귀 · {n}회 반복]')
    print(f'  기대값 {n}/{n} 일치 : 원문없음 {r["agree_bare"]}/{r["n"]} → 원문포함 {r["agree_orig"]}/{r["n"]}')
    print(f'  흔들림           : 원문없음 {r["waver_bare"]} · 원문포함 {r["waver_orig"]}')
    print(f'  판정 뒤집힘       : {len(r["flips"])}건 {r["flips"] or ""}')
    print(f'  **위험 방향**     : {r["risky"]}건  (거절해야 하는데 승인 — 0이 아니면 올리지 마라)')
    print(f'[방어 케이스] 기대 불일치 {bad}건')


if __name__ == "__main__":
    asyncio.run(main())
