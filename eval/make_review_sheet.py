"""gold 라벨 수동 검수용 시트 생성 (읽기 전용 — gold/corpus 원본 불변).

gold_set_v2.jsonl을 사람이 읽기 좋은 markdown으로 테넌트별로 뽑는다.
각 케이스에 type/질문/기대문서/기대청크/기대포인트를 정리하고, 검수 체크 4항목을 헤더에 안내.
출력: eval/review_sheets/{tenant}_review.md  (기존 파일 안 건드림, 새 디렉터리)

대조용 corpus 원문: sample_docs/corpus_v2/_src/{tenant}/

실행: python -m eval.make_review_sheet
"""
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
GOLD = ROOT / "gold_set_v2.jsonl"
OUT_DIR = ROOT / "review_sheets"

V2_TENANTS = {"summers", "homeplus", "adererror", "aromanica", "goodpeople", "harim"}

# type 출력 순서 (검수 우선순위: 거절/트랩 먼저 = 지점3, 그다음 검색류)
TYPE_ORDER = [
    "no_evidence", "trap", "single_fact", "paraphrase", "rare_lexical",
    "multi_doc", "multi_turn", "smalltalk", "safety", "prompt_injection", "pii",
]

CHECK_GUIDE = """> **검수 4항목** (각 케이스마다):
> 1. `type`이 맞나 — 특히 `no_evidence`인데 실은 문서에 근거(긍정/부정) 있어 답변 가능한 것
> 2. `기대문서`가 실제 정답 문서인가 (엉뚱한 문서 아닌가)
> 3. `기대청크`가 너무 좁게 못박았나 (문서는 맞는데 특정 청크 1개만 정답이라 옆 청크 찾으면 오답)
> 4. `기대포인트`가 실제 문서 내용과 일치하나
>
> 대조: `sample_docs/corpus_v2/_src/{tenant}/` 의 원문과 나란히 보기.
> 의심되면 케이스 앞에 `[?]` 표시하며 읽으세요.
"""


def _tenant(case_id: str) -> str:
    p = case_id.split("_")[0]
    return p if p in V2_TENANTS else "demo"


def _fmt_case(c: dict) -> str:
    lines = [f"### `{c['id']}`"]
    if c.get("conversation"):
        lines.append("**이전 대화(멀티턴):**")
        for m in c["conversation"]:
            lines.append(f"- {m['role']}: {m['content']}")
    lines.append(f"**Q:** {c['query']}")
    lines.append(f"- has_evidence: `{c.get('has_evidence')}`")
    docs = c.get("expected_docs") or []
    lines.append(f"- 기대문서: {', '.join(f'`{d}`' for d in docs) if docs else '(없음)'}")
    chunks = c.get("expected_chunks") or []
    if chunks:
        lines.append("- 기대청크:")
        for ch in chunks:
            lines.append(f"    - `{ch.get('filename','')}` → \"{ch.get('snippet','')}\"")
    pts = c.get("expected_points") or []
    if pts:
        lines.append(f"- 기대포인트: {', '.join(f'`{p}`' for p in pts)}")
    if c.get("safety_tags"):
        lines.append(f"- safety_tags: {c['safety_tags']}")
    return "\n".join(lines)


def main() -> None:
    cases = [json.loads(l) for l in GOLD.read_text().splitlines() if l.strip()]
    by_tenant = defaultdict(list)
    for c in cases:
        by_tenant[_tenant(c["id"])].append(c)

    OUT_DIR.mkdir(exist_ok=True)
    for tenant, tcases in sorted(by_tenant.items()):
        by_type = defaultdict(list)
        for c in tcases:
            by_type[c["type"]].append(c)

        out = [f"# gold 검수 시트 — {tenant}  ({len(tcases)}문항)", ""]
        out.append(CHECK_GUIDE.replace("{tenant}", tenant))
        out.append("")
        # type별 (우선순위 순, 목록에 없는 type은 뒤에)
        types = [t for t in TYPE_ORDER if t in by_type] + [t for t in by_type if t not in TYPE_ORDER]
        for t in types:
            group = by_type[t]
            out.append(f"\n## {t}  ({len(group)}문항)\n")
            for c in group:
                out.append(_fmt_case(c))
                out.append("")
        path = OUT_DIR / f"{tenant}_review.md"
        path.write_text("\n".join(out) + "\n")
        print(f"  {path}  ({len(tcases)}문항)")

    print(f"\n검수 시트 {len(by_tenant)}개 생성 → {OUT_DIR}")
    print("대조 원문: sample_docs/corpus_v2/_src/{tenant}/")


if __name__ == "__main__":
    main()
