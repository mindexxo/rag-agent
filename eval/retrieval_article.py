"""조 번호 질문 검색축 (#238) — "취업규칙 제48조 알려줘"에서 그 조가 오는가, 옆 조항·다른 규정이 섞이는가.

다른 팀 세미나 지적("유사도 검색으로 조 번호를 찾으면 옆 조항이 섞인다")의 실측용. 검색 코드는 건드리지 않는다
— 운영 경로(`retrieve_candidates` → `prepare_evidence`)를 그대로 태워 잰다.

질문셋 파일이 없다 — 실행할 때마다 색인에서 만든다(#238 결정 4). `eval/gold_private`에 두면
`eval.generation.load_gold`가 다른 축에 병합해 버리고, 재색인하면 고정 정답이 낡는다.

- 정답: 그 문서에서 색인 사전(`identifiers` — 제목 경로 + 묶음 헤딩, rag/identifiers)에 그 조가 있는 조각 전부(결정 3·20).
  하나라도 걸리면 적중. 본문 속 "제48조에 따라" 같은 참조는 사전에 없어 걸리지 않는다. 조가 여럿 든 조각은 "묶음"으로 따로 센다.
  **정답 정의와 식별자 채널이 같은 함수를 쓰므로 이 축은 자기검증 성격이다** — 독립 검증은 실제 채팅 질문(이슈 #238 코멘트의 23건)으로 한다.
- 질문: 조 하나에 하나, 문체 ①정직 ②러프 ③채팅체(오타)를 번갈아(결정 8). 오타는 문서명에만 — 조 번호가
  바뀌면 정답이 바뀐다. 시드 고정이라 매번 같은 질문이 나온다.
- 문서명 없는 "제N조" 질문은 원리적으로 모호해 검색 점수에서 뺀다(결정 2). `--answers`가 여러 문서에 겹치는
  번호 10개로 운영과 같은 검색→생성 답을 받아 파일로 떨군다 — 판정(단정했나/되묻거나 나눠 보였나)은 사람이 한다.
  `--answers`는 근거 상위 5개에 정답이 없는 문항의 실제 답변도 떨군다 — 검색 섞임이 답변 섞임으로 이어지는지가 진짜 위험이다.
- 합격 기준은 **정답이 근거 상위 5개(생성 모델이 받는 것)에 들었나**(`in_evidence`)다. Hit@1은 참고로만 남긴다
  — 1위가 아니어도 5개 안이면 생성이 골라 답했다(화면 재현 3건, 2026-10-15).

테넌트: inticube(사내 실문서) + article_test(공개 법령 DOCX — 가지 조문·DOCX 보강용, 결정 7).
필요: OpenSearch + TEI(임베딩·리랭커). `--answers`는 vLLM도.

사용: python -m eval.retrieval_article            # 검색 채점
     python -m eval.retrieval_article --answers  # + 문서명 없는 질문 10건·근거에 정답 없는 문항 답변
"""
import argparse
import asyncio
import json
import random
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from config import settings
from eval.axis_common import chunk_head, commit, load_chunks   # 색인 읽기·지문 — 키워드 축(#248)과 공용
from eval.retrieval import score_one
from rag.identifiers import chunk_identifiers
from rag.os_client import close_client
from rag.retriever import prepare_evidence, retrieve_candidates

TENANTS = ("inticube", "article_test")
RESULT_DIR = Path("eval/results")
SEED = 238
TOP_K = 5             # LLM이 받는 근거 수 — eval.generation.TOP_K와 같다
CONCURRENCY = 6       # TEI 공유 장비 — retrieval_mt와 같은 안전선
BARE_N = 10

# 파일명 → 상담원이 부를 이름 (결정 5). 회사명은 상담원이 붙이지 않는다.
_COMPANY_WORDS = ("주식회사", "(주)", "한솔인티큐브", "인티큐브")
_NAME_DROP = (
    re.compile(r"\[[^\]]*\]"),             # [인사 _ 규정] · [인티큐브]
    re.compile(r"\(\d+\)"),                # 다운로드 사본 표식 (1)
    re.compile(r"\s*Ver\s*[\d.]+", re.I),  # Ver1.8
    re.compile(r"_\d{6,8}"),               # _20260701
    re.compile(r"_[\d.]+$"),               # 끝 버전 _8.0
    re.compile(r"^\d+_"),                  # 앞 번호 4_
)


def clean_doc_name(filename: str) -> str:
    name = Path(filename).stem
    for rx in _NAME_DROP:
        name = rx.sub(" ", name)
    for w in _COMPANY_WORDS:
        name = name.replace(w, " ")
    return re.sub(r"[\s_]+", " ", name).strip()


def article_keys(heading_path: list[str], headings: list[str]) -> list[str]:
    """조각이 담은 조 — 색인 사전과 같은 함수(rag.identifiers)로 뽑고 '장' 토큰은 뺀다. 키는 '48조'·'76조의2' 꼴."""
    return [k for k in chunk_identifiers(heading_path, headings) if "조" in k]


def build_gold(chunks: list[dict]) -> tuple[dict, dict]:
    """조각 목록 → ((document_id, 조) → 정답 chunk_id 집합, chunk_id → 담은 조 목록)."""
    gold: dict[tuple[int, str], set[int]] = defaultdict(set)
    keys_of: dict[int, list[str]] = {}
    for c in chunks:
        keys = article_keys(c["heading_path"], c.get("headings") or [])
        keys_of[c["chunk_id"]] = keys
        for k in keys:
            gold[(c["document_id"], k)].add(c["chunk_id"])
    return dict(gold), keys_of


def _article_num(key: str) -> tuple[int, int]:
    """'76조의2' → (76, 2) — 정렬·이웃 판정용."""
    n = re.findall(r"\d+", key)
    return int(n[0]), int(n[1]) if len(n) > 1 else 0


# ----- 질문 생성 (결정 8) -------------------------------------------------

_CHAT_ENDINGS = ("머임", "어케돼요", "내용좀요", "뭐라고 나와있음?", "알려줘용", "확인좀")
_TYPO_TAILS = (0, 1, 4, 21)   # 받침 없음·ㄱ·ㄴ·ㅇ — 채팅 오타에서 흔한 받침 실수


def _typo(name: str, rng: random.Random) -> str:
    """한글 음절 하나의 받침을 바꾼다('규칙' → '귝칙'). 한글이 없으면 그대로."""
    idx = [i for i, ch in enumerate(name) if "가" <= ch <= "힣"]
    if not idx:
        return name
    i = rng.choice(idx)
    code = ord(name[i]) - 0xAC00
    tail = code % 28
    new_tail = rng.choice([t for t in _TYPO_TAILS if t != tail])
    return name[:i] + chr(0xAC00 + code - tail + new_tail) + name[i + 1:]


def make_query(doc: str, key: str, style: int, rng: random.Random) -> str:
    """style 0=정직 1=러프 2=채팅체. 조 번호(key, '48조' 꼴)는 어떤 문체에서도 글자가 바뀌지 않는다."""
    if style == 0:
        return f"{doc} 제{key}의 내용을 알려주세요"
    if style == 1:
        return f"{doc} {key} 뭐야"          # 러프·채팅체는 '제'를 뺀다
    name = _typo(doc, rng) if rng.random() < 0.5 else doc
    sep = "" if rng.random() < 0.5 else " "
    return f"{name}{sep}{key} {rng.choice(_CHAT_ENDINGS)}"


def build_items(chunks: list[dict], gold: dict, tenant: str) -> list[dict]:
    """(문서, 조)마다 질문 1개. 순서를 고정해 문체가 결정적으로 번갈아 붙는다."""
    meta = {c["document_id"]: c["filename"] for c in chunks}
    rng = random.Random(SEED)
    pairs = sorted(gold, key=lambda p: (meta[p[0]], _article_num(p[1])))
    items = []
    for i, (doc_id, key) in enumerate(pairs):
        filename = meta[doc_id]
        style = i % 3
        items.append({
            "tenant": tenant, "document_id": doc_id, "filename": filename, "article": key,
            "style": style, "query": make_query(clean_doc_name(filename), key, style, rng),
            "gold_ids": sorted(gold[(doc_id, key)]),
            "branch": "의" in key, "format": Path(filename).suffix.lower().lstrip("."),
        })
    return items


# ----- 채점 ---------------------------------------------------------------

def classify_mix(item: dict, top: list, keys_of: dict) -> dict:
    """상위 TOP_K 중 정답 밖 조각을 같은 문서(이웃 조 포함 여부)/다른 문서로 센다."""
    gold, (n, sub) = set(item["gold_ids"]), _article_num(item["article"])
    same = adjacent = other = 0
    for ch in top:
        if ch.chunk_id in gold:
            continue
        if ch.document_id == item["document_id"]:
            same += 1
            nums = {_article_num(k)[0] for k in keys_of.get(ch.chunk_id, [])}
            adjacent += bool(nums & {n - 1, n + 1} or (sub and n in nums))
        else:
            other += 1
    return {"mix_same_doc": same, "mix_adjacent": adjacent, "mix_other_doc": other}


async def score_items(items: list[dict], keys_of: dict) -> list[dict]:
    sem = asyncio.Semaphore(CONCURRENCY)

    async def _one(it: dict) -> dict:
        async with sem:
            # session은 쓰지 않는다(retrieve_candidates docstring) — 그래서 gather에 태울 수 있다.
            cands = await retrieve_candidates(None, it["tenant"], it["query"], top_n=20)
        got = [c.chunk_id for c in cands.chunks]
        top = prepare_evidence(cands.chunks, TOP_K)
        packed = any(len(keys_of.get(g, [])) > 1 for g in it["gold_ids"])
        # 합격 기준: 생성 모델이 실제로 받는 근거(top)에 정답이 들었나. recall_at_5(리랭크 순 상위 5)와
        # 거의 같지만 prepare_evidence의 접기·자르기 뒤라 운영과 정확히 같은 집합이다.
        in_evidence = float(any(c.chunk_id in set(it["gold_ids"]) for c in top))
        return {**it, **score_one(got, it["gold_ids"]), **classify_mix(it, top, keys_of),
                "in_evidence": in_evidence, "gold_packed": packed, "got_ids": got,
                "top": [{"chunk_id": c.chunk_id, "filename": c.filename,
                         "articles": keys_of.get(c.chunk_id, []), "head": c.text[:80]} for c in top]}

    return await asyncio.gather(*(_one(it) for it in items))


METRICS = ("in_evidence", "hit_at_1", "recall_at_5", "recall_at_20", "mrr")   # 합격 기준은 in_evidence


def summarize(rows: list[dict], by: str | None = None) -> dict:
    groups: dict = defaultdict(list)
    for r in rows:
        groups[r[by] if by else "all"].append(r)
    out = {}
    for g, rs in sorted(groups.items(), key=lambda kv: str(kv[0])):
        n = len(rs)
        out[str(g)] = {"n": n, **{m: round(sum(r[m] for r in rs) / n, 3) for m in METRICS},
                       "any_mix_same_doc": round(sum(r["mix_same_doc"] > 0 for r in rs) / n, 3),
                       "any_mix_adjacent": round(sum(r["mix_adjacent"] > 0 for r in rs) / n, 3),
                       "any_mix_other_doc": round(sum(r["mix_other_doc"] > 0 for r in rs) / n, 3)}
    return out


# ----- 사람이 읽을 파일 ----------------------------------------------------

def write_failures(path: Path, rows: list[dict], text_of: dict) -> None:
    lines = ["# 조 번호 질문 — 근거 상위 5개에 정답이 없는 사례", ""]
    for r in rows:
        if r["in_evidence"]:
            continue
        lines += [f"## [{r['tenant']}] {r['query']}  (정답 {r['filename']} {r['article']}, R@5={int(r['recall_at_5'])})",
                  f"- 정답 조각: " + " / ".join(chunk_head(text_of[g]) for g in r["gold_ids"][:2])]
        lines += [f"- {i}위 [{t['filename']}] {t['articles'][:4]} {chunk_head(t['head'], 80)}"
                  for i, t in enumerate(r["top"], 1)]
        lines.append("")
    path.write_text("\n".join(lines))


def write_gold_sample(path: Path, items: list[dict], text_of: dict, n: int = 20) -> None:
    """정답 표시가 맞는지 사람이 원문과 대조할 표본 (결정 3)."""
    rng = random.Random(SEED)
    lines = ["# 정답 조각 표본 — 원문의 그 조가 맞는지 대조", ""]
    for it in rng.sample(items, min(n, len(items))):
        lines += [f"## {it['filename']} {it['article']}  (조각 {len(it['gold_ids'])}개)"]
        lines += [f"- [{g}] {chunk_head(text_of[g], 300)}" for g in it["gold_ids"]]
        lines.append("")
    path.write_text("\n".join(lines))


# ----- 문서명 없는 질문 (결정 2·9) -------------------------------------------

_BARE_FORMS = ("제{key} 내용 알려줘", "{key} 뭐야", "제{key} 몇이야")


def pick_bare(gold: dict) -> list[str]:
    """여러 문서에 겹치는 조 번호에서 BARE_N개를 고르게 뽑는다."""
    docs_of: dict[str, set] = defaultdict(set)
    for doc_id, key in gold:
        docs_of[key].add(doc_id)
    shared = sorted((k for k, d in docs_of.items() if len(d) > 1), key=_article_num)
    step = max(1, len(shared) // BARE_N)
    return shared[::step][:BARE_N]


async def _answer(llm, tenant: str, q: str):
    """운영과 같은 검색 → 생성. (근거 상위 TOP_K, 답변)."""
    from eval.generation import generate   # 무거운 import(임베딩·심판) — --answers일 때만
    cands = await retrieve_candidates(None, tenant, q, top_n=20)
    chunks = prepare_evidence(cands.chunks, TOP_K)
    return chunks, await generate(llm, q, chunks)


def _evidence_line(chunks, keys_of: dict) -> str:
    return " / ".join(f"{c.filename} {keys_of.get(c.chunk_id, [])[:4]}" for c in chunks)


async def run_bare(llm, tenant: str, gold: dict, keys_of: dict, meta: dict, path: Path) -> None:
    lines = ["# 문서명 없는 조 번호 질문 — 단정했나 / 되묻거나 규정별로 나눠 보였나 (사람 판정)", ""]
    for i, key in enumerate(pick_bare(gold)):
        q = _BARE_FORMS[i % len(_BARE_FORMS)].format(key=key)
        chunks, answer = await _answer(llm, tenant, q)
        owners = sorted(meta[d] for d, k in gold if k == key)
        lines += [f"## {q}", f"- 이 번호를 가진 문서 {len(owners)}개: {', '.join(owners)}",
                  f"- 근거 상위 {TOP_K}: {_evidence_line(chunks, keys_of)}",
                  "", answer, "", "판정: ☐ 단정  ☐ 되물음  ☐ 규정별 제시  ☐ 기타", ""]
    path.write_text("\n".join(lines))


async def run_failed(llm, rows: list[dict], keys_of: dict, path: Path) -> None:
    """근거 상위 5개에 정답이 없는 문항의 실제 답변 — 다른 조·다른 규정 내용을 섞어 말하는가(사람 판정)."""
    lines = ["# 근거에 정답 없는 문항의 답변 — 거절했나 / 다른 조·규정 내용으로 답했나 (사람 판정)", ""]
    for r in rows:
        if r["in_evidence"]:
            continue
        chunks, answer = await _answer(llm, r["tenant"], r["query"])
        lines += [f"## [{r['tenant']}] {r['query']}  (정답 {r['filename']} {r['article']})",
                  f"- 근거 상위 {TOP_K}: {_evidence_line(chunks, keys_of)}",
                  "", answer, "", "판정: ☐ 정답 조만  ☐ 다른 조 섞임  ☐ 다른 규정 섞임  ☐ 다른 조로 답함  ☐ 거절", ""]
    path.write_text("\n".join(lines))


# ----- 진입점 -------------------------------------------------------------

async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", action="store_true",
                    help="문서명 없는 질문 10건 + 근거에 정답 없는 문항의 답변 생성(vLLM 필요)")
    args = ap.parse_args()

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    rows, items_all, text_of, keys_all = [], [], {}, {}
    llm = None
    if args.answers:
        from rag.llm import create_llm
        llm = create_llm()
    for tenant in TENANTS:
        chunks = await load_chunks(tenant)
        gold, keys_of = build_gold(chunks)
        text_of.update({c["chunk_id"]: c["text"] for c in chunks})
        keys_all.update(keys_of)
        items = build_items(chunks, gold, tenant)
        print(f"{tenant}: 조각 {len(chunks)} · 문서+조 {len(items)}")
        items_all += items
        rows += await score_items(items, keys_of)
        if llm and tenant == "inticube":
            meta = {c["document_id"]: c["filename"] for c in chunks}
            await run_bare(llm, tenant, gold, keys_of, meta, RESULT_DIR / f"article_{stamp}_bare.md")

    summary = {by or "all": summarize(rows, by)
               for by in (None, "tenant", "style", "format", "branch", "gold_packed")}
    fingerprint = {"commit": commit(), "rerank": settings.rerank_enabled,
                   "hybrid_lexical": settings.hybrid_lexical_enabled,
                   "hybrid_inject": settings.hybrid_lexical_inject, "seed": SEED, "at": stamp}
    out = RESULT_DIR / f"article_{stamp}.json"
    out.write_text(json.dumps({"fingerprint": fingerprint, "summary": summary, "rows": rows},
                              ensure_ascii=False, indent=1))
    write_failures(RESULT_DIR / f"article_{stamp}_failures.md", rows, text_of)
    write_gold_sample(RESULT_DIR / f"article_{stamp}_gold_sample.md", items_all, text_of)
    if llm:
        await run_failed(llm, rows, keys_all, RESULT_DIR / f"article_{stamp}_failed_answers.md")
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    print(f"→ {out}")


async def _run() -> None:
    try:
        await main()
    finally:
        await close_client()


if __name__ == "__main__":
    asyncio.run(_run())
