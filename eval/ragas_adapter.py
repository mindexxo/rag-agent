"""RAGAS 입력 어댑터.

저장된 생성 결과(eval/results/generation_*.jsonl) + gold(gold_set_v2.jsonl)를
RAGAS EvaluationDataset으로 변환. judge 불필요 — 순수 데이터 변환/검증용.

필드 매핑:
- user_input        ← gold.query (id로 조인; 생성 결과엔 query가 없음)
- response          ← 생성 결과.answer
- retrieved_contexts← 생성 결과.retrieved_contexts (chunk 텍스트 리스트)
- reference         ← gold.expected_points 결합 (context 계열 metric용, 약한 placeholder)

faithfulness / answer_relevancy는 reference 없이도 동작.
context_precision/recall은 reference 필요 → 현재 expected_points 기반(정답 문장 부재, gap2).

실행(스모크): python -m eval.ragas_adapter
"""
import json
from pathlib import Path

from ragas import EvaluationDataset

GOLD = Path("eval/gold_set_v2.jsonl")
RESULT_DIR = Path("eval/results")


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def build_dataset(mode: str = "retrieved") -> EvaluationDataset:
    """저장된 {mode} 생성 결과 → RAGAS EvaluationDataset."""
    gold_by_id = {g["id"]: g for g in _load_jsonl(GOLD)}
    rows = _load_jsonl(RESULT_DIR / f"generation_{mode}.jsonl")

    samples = []
    for r in rows:
        g = gold_by_id.get(r["id"])
        if g is None:                       # gold에 없는 결과 → 스킵
            continue
        samples.append({
            # multi_turn은 condense 재작성 질문을 사용 — 원 후속 질문("그건 언제까지?")은
            # 맥락이 없어 answer_relevancy가 부당하게 깎임. 재작성 질문이 답변의 공정한 기준.
            "user_input": r.get("standalone_query") or g["query"],
            "response": r["answer"],
            "retrieved_contexts": r.get("retrieved_contexts", []),
            "reference": " ".join(g.get("expected_points", [])),
        })
    return EvaluationDataset.from_list(samples)


if __name__ == "__main__":
    for mode in ("retrieved", "oracle"):
        ds = build_dataset(mode)
        print(f"=== {mode}: {len(ds)} samples ===")
        s = ds[0]
        print("  user_input        :", s.user_input[:50])
        print("  response          :", s.response[:50])
        print("  retrieved_contexts:", len(s.retrieved_contexts), "개")
        print("  reference         :", repr(s.reference))
