"""FAQ 항목 인용 정확도 축 (#278) — "FAQ 여러 개 중 **맞는 항목**을 인용하나".

#278 전에는 FAQ 인용이 'FAQ' 하나로 접혀(rag/citation_labels.sources_from_chunks) 어느 항목을 인용했는지
어디에도 남지 않았다 — 그래서 골드셋(eval/gold_set_v2.jsonl)에 FAQ 문항이 0건이고 잴 수 있는 것도
"FAQ를 인용했나"뿐이었다. 이 축은 항목 단위로 잰다: 합성 FAQ 30개(eval/faq_set.jsonl, 주제가 가까운 6묶음×5 —
옆 항목과 헷갈려야 시험이 된다)를 전용 테넌트에 넣고, 항목당 질문 2개(대표 질문 그대로·구어체 바꿔 쓰기)로
운영과 같은 배관(검색→리랭크→근거 준비→생성→꼬리 해석)을 태운 뒤 인용된 faq_id가 기대 항목인지 본다.

FAQ 내용은 문서와 **다르게** 만들었다(사용자 결정 7) — 겹치면 모델이 문서를 인용해 버리는데 그건 틀린 게 아니라
채점이 흐려진다. 전용 테넌트(FAQ_EVAL_TENANT)에는 문서가 없다 — 경쟁 상대는 다른 FAQ 항목이다.

지표(행마다 기록, 요약은 평균):
- faq_cited: 인용 중 FAQ가 하나라도 있나 — #278 전 코드에서도 잴 수 있는 유일한 값(전후 비교용)
- correct:   기대 faq_id가 인용에 있나 — #278 후 코드에서만 값이 있다(전 코드는 인용 객체에 faq_id가 없어 None)
- retrieved / rank: 기대 항목이 근거(prepare_evidence 결과)에 들어왔나, 몇 번째인가 — 생성과 검색 책임을 가른다
- n_candidates: 인용 후보 수 — #278 전엔 FAQ가 1로 접혀 항상 1, 후엔 근거에 든 항목 수

사용:
  python -m eval.faq_citation load              # 세트를 테넌트에 적재(기존 FAQ 삭제 후) + 색인(TEI 필요)
  python -m eval.faq_citation measure --label after   # 생성·채점 → eval/results/faq_citation_<label>.jsonl
'전' 측정은 #278 전 커밋의 워크트리에 이 파일과 세트를 복사해 같은 명령으로 돌린다(load는 한 번이면 된다 —
청크 텍스트 조립 규약 rag/search/faq_indexing.build_faq_chunk_text가 같다). 결과를 인용할 때는 커밋을 함께 적는다(AGENTS.md).
"""
import argparse
import asyncio
import json
from collections import defaultdict
from pathlib import Path

from sqlalchemy import delete, select

from database import AsyncSessionLocal
from eval.axis_common import commit
from eval.generation import CONCURRENCY, RESULT_DIR, generate, retrieved_context
from rag.citation_labels import sources_from_chunks
from rag.citation_tail import TailSplitter, resolve_citations
from rag.llm import LlmClient, create_llm
from rag.models import Faq
from rag.search import os_index

FAQ_EVAL_TENANT = "faq_eval"          # 전용 테넌트 — 다른 축의 측정 테넌트에 FAQ를 섞지 않는다(eval/axis_common.load_chunks 전제)
SET = Path("eval/faq_set.jsonl")


def load_set() -> list[dict]:
    return [json.loads(line) for line in SET.read_text().splitlines() if line.strip()]


async def load(session) -> None:
    """세트를 테넌트에 적재 — 기존 FAQ(행·색인)를 지우고 다시 넣는다. 색인은 워커를 거치지 않고 바로(os_index.index_faq_chunks)."""
    old_ids = (await session.execute(select(Faq.id).where(Faq.tenant_id == FAQ_EVAL_TENANT))).scalars().all()
    if old_ids:
        await os_index.drop_faqs_now(old_ids)
        await session.execute(delete(Faq).where(Faq.tenant_id == FAQ_EVAL_TENANT))
    rows = load_set()
    faqs = [Faq(tenant_id=FAQ_EVAL_TENANT, question=r["question"], variants=r["variants"], answer=r["answer"]) for r in rows]
    session.add_all(faqs)
    await session.flush()
    for f in faqs:
        await os_index.index_faq_chunks(session, f.id)
    await session.commit()
    print(f"{FAQ_EVAL_TENANT}: FAQ {len(faqs)}건 적재·색인 (기존 {len(old_ids)}건 삭제)")


def _cited(answer: str, chunks) -> list:
    splitter = TailSplitter()
    splitter.feed(answer)
    splitter.finish()
    return resolve_citations(splitter.tail_raw, sources_from_chunks(chunks), [])


async def measure(session, llm: LlmClient, label: str) -> list[dict]:
    rows = load_set()
    by_question = {q: i for q, i in (await session.execute(
        select(Faq.question, Faq.id).where(Faq.tenant_id == FAQ_EVAL_TENANT))).all()}
    missing = [r["key"] for r in rows if r["question"] not in by_question]
    if missing:
        raise SystemExit(f"테넌트에 없는 FAQ {len(missing)}건 — 먼저 load: {missing[:5]}")

    work = []
    for r in rows:
        for q in r["queries"]:
            chunks = await retrieved_context(session, FAQ_EVAL_TENANT, q["text"])   # 세션 직렬
            work.append((r, q, chunks))

    sem = asyncio.Semaphore(CONCURRENCY)

    async def _gen(q, chunks):
        async with sem:
            return await generate(llm, q["text"], chunks)

    answers = await asyncio.gather(*(_gen(q, c) for _, q, c in work))

    out = []
    for (r, q, chunks), answer in zip(work, answers):
        expected = by_question[r["question"]]
        cited = _cited(answer, chunks)
        cited_faq_ids = [getattr(c, "faq_id", None) for c in cited]
        per_item = any(getattr(c, "faq_id", None) is not None for c in sources_from_chunks(chunks))   # #278 후 코드인가
        ranks = [i + 1 for i, c in enumerate(chunks) if c.faq_id == expected]
        out.append({
            "key": r["key"], "cluster": r["cluster"], "kind": q["kind"], "query": q["text"],
            "expected_faq_id": expected, "expected_question": r["question"],
            "answer": answer,
            "cited": [c.filename for c in cited],
            "cited_faq_ids": [i for i in cited_faq_ids if i is not None],
            "faq_cited": any(i is not None or c.filename == "FAQ" for i, c in zip(cited_faq_ids, cited)),
            "correct": (expected in cited_faq_ids) if per_item else None,
            "retrieved": bool(ranks), "rank": ranks[0] if ranks else None,
            "n_candidates": len(sources_from_chunks(chunks)),
            "n_cited": len(cited),
            "label": label, "commit": commit(),
        })
    return out


def summarize(rows: list[dict]) -> None:
    def _rate(rs, key):
        vals = [r[key] for r in rs if r[key] is not None]
        return f"{sum(vals) / len(vals):.3f}" if vals else "-"
    print(f"\n== FAQ 항목 인용 ({rows[0]['label']}, {rows[0]['commit']}, {len(rows)}문항) ==")
    print(f"{'구분':<10}{'n':>4}  {'faq_cited':>9}  {'correct':>8}  {'retrieved':>9}  {'top1':>6}  {'cand':>5}  {'cited/행':>8}")
    groups = [("전체", rows)] + [(k, [r for r in rows if r["kind"] == k]) for k in ("direct", "para")] \
        + [(c, [r for r in rows if r["cluster"] == c]) for c in dict.fromkeys(r["cluster"] for r in rows)]
    for name, rs in groups:
        top1 = sum(1 for r in rs if r["rank"] == 1) / len(rs)
        cand = sum(r["n_candidates"] for r in rs) / len(rs)
        ncited = sum(r["n_cited"] for r in rs) / len(rs)
        print(f"{name:<10}{len(rs):>4}  {_rate(rs, 'faq_cited'):>9}  {_rate(rs, 'correct'):>8}  "
              f"{_rate(rs, 'retrieved'):>9}  {top1:>6.3f}  {cand:>5.1f}  {ncited:>8.2f}")


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["load", "measure"])
    ap.add_argument("--label", default="after")
    args = ap.parse_args()
    async with AsyncSessionLocal() as session:
        if args.cmd == "load":
            await load(session)
            return
        rows = await measure(session, create_llm(), args.label)
    RESULT_DIR.mkdir(exist_ok=True)
    out = RESULT_DIR / f"faq_citation_{args.label}.jsonl"
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows))
    summarize(rows)
    print(f"→ {out}")


if __name__ == "__main__":
    asyncio.run(main())
