"""키워드형 질문 검색축 (#248) — 상담원이 문서 속 코드·약어·전문용어를 **짧게** 치는 질의("AR-2003", "HACCP 알려줘")에서
그 토큰이 나오는 조각이 근거로 오는가.

왜 따로 있나: 기존 골드의 키워드 성격 유형(rare_lexical — 정본 60 + 사내 실문서 골드 11, 2026-10-18)은 "상품코드 AR-2003 원료의 계열이 뭐예요?"처럼 **완결된 문장**이다.
상담원식 짧은 질의는 0건이라, 하이브리드의 키워드 쪽이 실제로 일하는지를 그 골드로는 볼 수 없었다(#128 "주입 상방 0/450"도
문장형 골드의 결과). 이 축이 그 빈칸을 잰다.

질문셋 파일이 없다 — 실행할 때마다 색인에서 만든다(eval/axis_common docstring). 시드 고정이라 색인이 같으면 같은 질문이 나온다.
- 토큰(#248 범위 결정 코멘트의 결정 3): ① 코드(라틴+숫자, 'AR-2003'·'HF-101'·'SB-1000') ② 라틴 대문자 약어('HACCP'·'CTI') ③ 한국어 명사 — 색인과 같은
  Nori 분석기의 품사로 명사 형태소를 고르고, 띄어쓰기 없이 붙은 명사 형태소는 이어 붙인다('시향'+'일지' → '시향일지').
  셋 다 테넌트 안에서 **조각 3개 이하**에 나오는 것만(GOLD_MAX_DF) — 흔한 단어는 키워드 질의의 대상이 아니다.
- 정답(결정 4): 그 토큰이 든 조각 전부, 공백·대소문자 무시(norm_text). 하나라도 걸리면 적중. 조각 본문(text)만 본다 —
  파일명·제목 머리말에만 있는 토큰은 정답이 비어 질문이 성립하지 않는다.
- 질문(결정 3·설계 6): 토큰당 1문항, 문체 3종을 번갈아 — ① 토큰만 ② 토큰+뜻을 바꾸지 않는 꼬리 ③ "토큰 뭐야".
  꼬리에 속성 단어를 붙이지 않는다 — 정답 조각의 단어가 질문에 새어 들어가면 "토큰 하나로 찾나"를 못 잰다.
- 상한(설계 7): 테넌트당 CAP_PER_TENANT, 종류별 균등 배분(모자라면 다른 종류로 채움).
- 합격 기준(설계 5): 생성 모델이 실제로 받는 근거 상위 5개의 **글**에 그 토큰이 있나(in_evidence, evidence_has) — 표 시트
  접기 때문에 조각 id가 아니라 글로 본다.

알려진 한계: Nori가 명사가 아닌 품사로 잘못 자르는 용어(부사+동사로 분석되는 업무 용어가 실재한다)는 명사로 안 잡혀 빠진다. 토큰을 뽑은 규칙과 BM25의 매칭 원리가
닮아 이 축은 **BM25에 유리한 쪽으로 기운 상한 추정**이다 — 문장형 골드(retrieval_v2)와 나란히 읽는다.
실문서 토큰·원문은 결과(eval/results, gitignore)에만 쓴다 — 저장소가 공개다.

필요: OpenSearch + TEI(임베딩·리랭커).
사용: python -m eval.retrieval_keyword [--tenants a,b] [--cap 60] [--spot 30]
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
from eval.axis_common import chunk_head, commit, evidence_has, load_chunks, norm_text
from eval.generation import TOP_K, V2_TENANTS
from eval.retrieval import score_one
from rag.search.os_client import client, close_client
from rag.search.retriever import prepare_evidence, retrieve_candidates

TENANTS = tuple(sorted(V2_TENANTS))   # 모의 6 + inticube — 문장형 골드(retrieval_v2)와 같은 테넌트
RESULT_DIR = Path("eval/results")
SEED = 248
CONCURRENCY = 6           # TEI 공유 장비 — retrieval_article과 같은 안전선
ANALYZE_CONCURRENCY = 8   # _analyze는 OpenSearch만 쓴다
CAP_PER_TENANT = 60
GOLD_MAX_DF = 3
KINDS = ("code", "acronym", "noun")
TAILS = ("관련 내용", "알려줘", "어떻게 돼?")   # 뜻을 바꾸지 않는 꼬리(설계 6)
METRICS = ("in_evidence", "hit_at_1", "recall_at_5", "recall_at_20", "mrr")   # 합격 기준은 in_evidence

# 코드 — 라틴 문자와 숫자가 함께 있는 한 덩어리. 하이픈으로 이어진 문자 마디는 통째로 본다 — 'SUM-MB-012'에서 'MB-012'만
# 떼어 내면 문서번호의 일부가 코드로 둔갑한다(첫 검수 실측). 순수 숫자·날짜는 문자가 없어 안 걸린다.
# 버전 표기('Ver1.8'·'v1.2')는 코드가 아니다 — 숫자 뒤에 '.숫자'가 이어지면 뺀다(리뷰 지적).
_CODE_RE = re.compile(r"(?<![A-Za-z0-9-])[A-Za-z]{1,6}(?:-[A-Za-z]{1,6})*-?\d{1,6}(?:-[A-Za-z0-9]{1,6})*(?![A-Za-z0-9]|\.\d)")
# 약어 — 라틴 대문자 2~6자. 앞뒤가 라틴 문자·숫자·하이픈이면 더 긴 단어·코드의 일부라 뺀다('HP-OPS-015'의 'OPS',
# 'SB1000'의 'SB' — 리뷰 지적).
_ACRONYM_RE = re.compile(r"(?<![A-Za-z0-9-])[A-Z]{2,6}(?![A-Za-z0-9-])")
# 파일 형식명은 문서 안내문에 흔하지만 상담원이 묻는 대상이 아니다(코퍼스 실측 2026-10-18).
_ACRONYM_STOP = frozenset({"PDF", "DOC", "DOCX", "XLS", "XLSX", "PPT", "PPTX", "HWP", "TXT", "CSV", "JPG", "PNG"})
_NOUN_POS = ("NNG", "NNP")
_HANGUL_RE = re.compile(r"[가-힣]")


def extract_codes(text: str) -> set[str]:
    """코드 토큰(_CODE_RE). 하이픈을 빼고 3자 미만이면 버린다."""
    return {m for m in _CODE_RE.findall(text) if len(m.replace('-', '')) >= 3}   # 'D-1'·'N-2'(일정 표기)는 코드가 아니다


def extract_acronyms(text: str) -> set[str]:
    """약어 토큰(_ACRONYM_RE) — 파일 형식명 제외."""
    return {m for m in _ACRONYM_RE.findall(text) if m not in _ACRONYM_STOP}


def noun_terms(tokens: list[dict]) -> set[str]:
    """Nori 토크나이저 출력(explain의 tokenizer 단계) → **복합 명사** 용어. 품사가 NNG/NNP인 형태소를 고르고, 원문에서 띄어쓰기
    없이 바로 붙은 명사 형태소는 하나로 잇는다('시향'+'일지'). Nori가 복합어로 낸 토큰(posType COMPOUND)은 그 자체를 쓰고
    하위 형태소는 건너뛴다. **명사 형태소 2개 이상이 이어졌거나 Nori 복합어인 것만** 남긴다 — 첫 검수에서 형태소 하나짜리
    명사는 10개 중 8개가 일반어('합산'·'전체'·'한도')였고, 복합어는 대부분 업무 용어('유산기부'·'체험등급')였다(2026-10-18).
    두 글자 이상 한글을 가진 것만 — 짧으면 부분문자열 정답이 지나치게 넓어진다."""
    out: set[str] = set()
    run: list[dict] = []
    skip_until = -1

    def flush() -> None:
        term = "".join(t["token"] for t in run)
        if (len(run) >= 2 or (run and run[0].get("posType") == "COMPOUND")) and len(_HANGUL_RE.findall(term)) >= 2:
            out.add(term)
        run.clear()

    for t in tokens:
        if t["start_offset"] < skip_until:          # 복합어의 하위 형태소
            continue
        is_noun = str(t.get("leftPOS", "")).startswith(_NOUN_POS)
        if t.get("posType") == "COMPOUND":
            skip_until = t["end_offset"]
        if is_noun and (not run or run[-1]["end_offset"] == t["start_offset"]):
            run.append(t)
            continue
        flush()
        if is_noun:
            run.append(t)
    flush()
    return out


def gold_of(token: str, norm_texts: dict[int, str]) -> list[int]:
    """정답 = 토큰이 든 조각 전부(결정 4)."""
    n = norm_text(token)
    return sorted(cid for cid, t in norm_texts.items() if n in t)


def select_candidates(cands: dict[str, str], norm_texts: dict[int, str], other_text: str = "",
                      max_df: int = GOLD_MAX_DF) -> dict[str, tuple[str, list[int]]]:
    """{토큰: 종류} → {토큰: (종류, 정답 조각)} — 정답이 1~max_df개인 것만. 정규화가 같은 토큰('CTI'·'cti')은 하나만.

    명사는 다른 테넌트 글(other_text, 정규화해 이은 것)에 한 번이라도 나오면 뺀다 — 여러 회사 문서에 공통으로 나오는 말은
    일반어이고, 그 회사 문서에만 나오는 말이 업무 용어다(첫 검수 실측, noun_terms docstring). 코드·약어에는 걸지 않는다 —
    'P001' 같은 코드는 다른 회사에 같은 글자가 있어도 그 회사 안에서는 고유한 이름이다."""
    out: dict[str, tuple[str, list[int]]] = {}
    seen: set[str] = set()
    for token in sorted(cands):
        key = norm_text(token)
        if key in seen or (cands[token] == "noun" and key in other_text):
            continue
        gold = gold_of(token, norm_texts)
        if 1 <= len(gold) <= max_df:
            out[token] = (cands[token], gold)
            seen.add(key)
    return out


def sample_tokens(selected: dict[str, tuple[str, list[int]]], cap: int,
                  rng: random.Random) -> list[tuple[str, str, list[int]]]:
    """종류별 균등 배분(cap // 종류 수), 모자란 종류의 몫은 남은 후보에서 채운다(설계 7). 결과는 (종류, 토큰) 순 정렬."""
    by_kind: dict[str, list[str]] = defaultdict(list)
    for token, (kind, _) in sorted(selected.items()):
        by_kind[kind].append(token)
    picked: list[str] = []
    rest: list[str] = []
    quota = cap // len(KINDS)
    for kind in KINDS:
        pool = by_kind.get(kind, [])
        rng.shuffle(pool)
        picked += pool[:quota]
        rest += pool[quota:]
    rng.shuffle(rest)
    picked += rest[:max(0, cap - len(picked))]
    return sorted(((selected[t][0], t, selected[t][1]) for t in picked), key=lambda x: (x[0], x[1]))


def make_query(token: str, i: int) -> str:
    """문체 3종을 번갈아(i % 3). 토큰 글자는 어느 문체에서도 바꾸지 않는다."""
    style = i % 3
    if style == 0:
        return token
    if style == 1:
        return f"{token} {TAILS[(i // 3) % len(TAILS)]}"
    return f"{token} 뭐야"


def build_items(tenant: str, picked: list[tuple[str, str, list[int]]]) -> list[dict]:
    """(종류, 토큰, 정답) → 문항. qid는 테넌트 안에서 순번이라 테넌트를 섞어도 유일하다."""
    return [{"set": "keyword", "tenant": tenant, "qid": f"kw_{tenant}_{i:03d}", "type": f"keyword_{kind}",
             "kind": kind, "style": i % 3, "token": token, "query": make_query(token, i), "gold_ids": gold}
            for i, (kind, token, gold) in enumerate(picked)]


# ----- 색인 읽기 ----------------------------------------------------------

async def _analyze_nouns(texts: list[str]) -> list[set[str]]:
    sem = asyncio.Semaphore(ANALYZE_CONCURRENCY)

    async def _one(text: str) -> set[str]:
        if not _HANGUL_RE.search(text):
            return set()
        async with sem:
            r = await client().indices.analyze(index=settings.opensearch_index, body={
                "analyzer": "nori_ko", "text": text, "explain": True, "attributes": ["leftPOS", "posType"]})
        return noun_terms(r["detail"]["tokenizer"]["tokens"])

    return await asyncio.gather(*(_one(t) for t in texts))


async def generate_items(tenants: tuple[str, ...] = TENANTS,
                         cap: int = CAP_PER_TENANT) -> tuple[list[dict], dict[int, str]]:
    """질문 생성 → (문항, {chunk_id: 원문}). 원문은 검수 덤프용. 명사의 "다른 테넌트에 없음" 판정 때문에 테넌트들을 함께
    읽는다 — 그래서 tenants를 바꾸면 명사 후보도 바뀐다(기본값 = 측정 테넌트 전부, 시드 고정)."""
    chunks_of = {t: await load_chunks(t) for t in tenants}
    norm_of = {t: {c["chunk_id"]: norm_text(c["text"]) for c in cs} for t, cs in chunks_of.items()}
    items: list[dict] = []
    text_of: dict[int, str] = {}
    for tenant, chunks in chunks_of.items():
        text_of.update({c["chunk_id"]: c["text"] for c in chunks})
        # 조각 사이를 \x00으로 잇는다 — 정규화가 공백을 지우므로 그냥 이으면 조각 경계를 넘는 우연한 일치가 생긴다
        other_text = "\x00".join(t for u, n in norm_of.items() if u != tenant for t in n.values())
        cands: dict[str, str] = {}
        nouns = await _analyze_nouns([c["text"] for c in chunks])
        for c, ns in zip(chunks, nouns):
            for kind, toks in (("code", extract_codes(c["text"])), ("acronym", extract_acronyms(c["text"])), ("noun", ns)):
                for tok in toks:
                    cands.setdefault(tok, kind)   # 같은 글자가 두 종류에 걸리면 먼저 잡힌 쪽(코드 > 약어 > 명사)
        picked = sample_tokens(select_candidates(cands, norm_of[tenant], other_text), cap,
                               random.Random(f"{SEED}:{tenant}"))
        items += build_items(tenant, picked)
    return items, text_of


# ----- 채점 ---------------------------------------------------------------

async def score_items(items: list[dict]) -> list[dict]:
    sem = asyncio.Semaphore(CONCURRENCY)

    async def _one(it: dict) -> dict:
        async with sem:
            # session은 쓰지 않는다(retrieve_candidates docstring) — 그래서 gather에 태울 수 있다.
            cands = await retrieve_candidates(None, it["tenant"], it["query"], top_n=20)
        got = [c.chunk_id for c in cands.chunks]
        top = prepare_evidence(cands.chunks, TOP_K)
        return {**it, **score_one(got, it["gold_ids"]), "in_evidence": evidence_has(top, it["token"]),
                "got_ids": got, "top_ids": [c.chunk_id for c in top]}

    return await asyncio.gather(*(_one(it) for it in items))


def summarize(rows: list[dict], by: str | None = None) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[str(r[by]) if by else "all"].append(r)
    return {k: {"n": len(v), **{m: round(sum(r[m] for r in v) / len(v), 3) for m in METRICS}}
            for k, v in sorted(groups.items())}


def _snippet(text: str, token: str, width: int = 60) -> str:
    """원문에서 토큰 주변만 — 대소문자 무시, 공백이 끼어 원문 위치를 못 찾으면 조각 머리."""
    i = text.casefold().find(token.casefold())
    if i < 0:
        return chunk_head(text, width * 2)
    return chunk_head(text[max(0, i - width):i + len(token) + width], width * 3)


def write_spot_check(path: Path, items: list[dict], text_of: dict[int, str], n: int = 30) -> None:
    """검수 표본(결정 2) — 종류별로 고르게 뽑아 질문·정답 조각을 원문과 나란히. 원문이 들어가므로 eval/results에만 쓴다."""
    rng = random.Random(SEED)
    by_kind: dict[str, list[dict]] = defaultdict(list)
    for it in items:
        by_kind[it["kind"]].append(it)
    per = max(1, n // len(KINDS))
    sample = [it for k in KINDS for it in rng.sample(by_kind.get(k, []), min(per, len(by_kind.get(k, []))))]
    lines = ["# 키워드형 질문 검수 표본 — 토큰이 상담원이 물을 만한 말인가 / 정답 조각에 그 토큰이 실제로 있나", "",
             "판정: ☐ 적합  ☐ 토큰이 일반어(키워드 질의 대상 아님)  ☐ 토큰이 잘못 잘림  ☐ 정답 조각 이상", ""]
    for it in sample:
        lines += [f"## [{it['tenant']}] {it['query']}  ({it['kind']}, 정답 조각 {len(it['gold_ids'])}개)"]
        lines += [f"- [{g}] {_snippet(text_of[g], it['token'])}" for g in it["gold_ids"]]
        lines += ["", "판정: ☐ 적합  ☐ 일반어  ☐ 잘못 잘림  ☐ 정답 이상", ""]
    path.write_text("\n".join(lines))


# ----- 진입점 -------------------------------------------------------------

async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tenants", default=",".join(TENANTS))
    ap.add_argument("--cap", type=int, default=CAP_PER_TENANT)
    ap.add_argument("--spot", type=int, default=30)
    args = ap.parse_args()

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    items_all, text_all = await generate_items(tuple(args.tenants.split(",")), args.cap)
    for tenant in args.tenants.split(","):
        counts = {k: sum(it["kind"] == k for it in items_all if it["tenant"] == tenant) for k in KINDS}
        print(f"{tenant}: 문항 {sum(counts.values())} {counts}", flush=True)
    rows = await score_items(items_all)

    summary = {by or "all": summarize(rows, by) for by in (None, "tenant", "kind", "style")}
    fingerprint = {"commit": commit(), "rerank": settings.rerank_enabled,
                   "hybrid_lexical": settings.hybrid_lexical_enabled, "hybrid_inject": settings.hybrid_lexical_inject,
                   "ident_channel": settings.ident_channel_enabled, "seed": SEED, "cap": args.cap, "at": stamp}
    out = RESULT_DIR / f"keyword_{stamp}.json"
    out.write_text(json.dumps({"fingerprint": fingerprint, "summary": summary, "rows": rows},
                              ensure_ascii=False, indent=1))
    (RESULT_DIR / f"keyword_{stamp}_items.jsonl").write_text(
        "\n".join(json.dumps(it, ensure_ascii=False) for it in items_all))
    write_spot_check(RESULT_DIR / f"keyword_{stamp}_spot.md", items_all, text_all, args.spot)
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    print(f"→ {out}")


async def _run() -> None:
    try:
        await main()
    finally:
        await close_client()


if __name__ == "__main__":
    asyncio.run(_run())
