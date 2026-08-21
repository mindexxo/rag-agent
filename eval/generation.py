"""1.5.D 생성 평가 하네스 (oracle / retrieved 2모드, §5.3).

검색 결과를 고정하고 LLM만 평가:
- oracle:    gold 정답 청크를 직접 주입 → LLM 순수 성능 (검색 변수 제거)
- retrieved: 실제 검색기(hybrid + 리랭커) 결과 주입 → 시스템 전체

채점:
- deterministic(서버 무관): Expected Points Coverage, Citation Accuracy
- LLM-judge(서버 + 심판모델 필요): Answer Relevancy, Faithfulness → 현재 stub

LLM 서버 필요: .env의 VLLM_BASE_URL/VLLM_MODEL (Ollama qwen3:4b 또는 vLLM).
실행: python -m eval.generation
"""
import asyncio
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

from sqlalchemy import select

from config import settings
from database import AsyncSessionLocal
from rag.citation_labels import sources_from_chunks
from rag.citation_tail import TailSplitter, resolve_citations
from rag.conversation import build_prior_turns, condense_query
from rag.models import Chunk, Document
from rag.retriever import retrieve_candidates, RetrievedChunk
from rag.embeddings import embed_texts_sync
from rag.llm_schemas import is_schema_rejected
from rag.prompts import build_citation_constraint, build_knowledge_generation_prompt
from rag.llm import LlmClient
from eval.retrieval import resolve_gold

TENANT = "demo"                      # v1(단일 테넌트) 기본값 — v2 gold는 id 접두로 라우팅
V2_TENANTS = {"summers", "homeplus", "adererror", "aromanica", "goodpeople", "harim"}
GOLD = Path("eval/gold_set_v2.jsonl")
GOLD_VERSION = "v2"


def row_tenant(g: dict) -> str:
    """gold 문항의 테넌트 — id 접두(summers_sf001 등), v1 형식이면 demo."""
    prefix = g["id"].split("_")[0]
    return prefix if prefix in V2_TENANTS else TENANT
RESULT_DIR = Path("eval/results")
# 생성 평가 본체. multi_turn은 gold의 conversation을 운영 condense_query에 태워
# 독립 질문으로 재작성한 뒤 동일 경로로 평가 (E.1 운영 경로 그대로 측정)
#
# multi_turn은 **이 날짜 이전 결과와 비교 불가** (#48, 2026-08-19): 두 가지가 바뀌었다.
#   1) 이력 블록(prior_turns)을 넘기지 않던 버그 수정 — 운영은 넣는데 여기는 안 넣었고,
#      그래서 "재작성 오염 때문에 오답"이라는 오진이 나왔다(같은 오염 재작성본으로 재측정하니
#      이력 없음 1/10 · 이력 있음 8/10).
#   2) "질문:" 슬롯이 재작성본 → 원 질문 + 재작성본 병기로 바뀜.
# citation_accuracy v3(#56)와 같은 종류의 고지다 — 측정 정의가 바뀐 것이지 모델이 변한 게 아니다.
# single_fact/paraphrase/rare_lexical/multi_doc은 이력이 없고 두 질의가 같아 렌더 무변경.
GEN_TYPES = {"single_fact", "paraphrase", "rare_lexical", "multi_doc", "multi_turn"}
TOP_K = 5
USE_RERANK = False         # 추가 리랭크 여부. 주의: retrieve_candidates가 settings.rerank_enabled(현재 True)로
                           # 이미 내부 리랭크 1회 수행 → retrieved 모드는 이미 '리랭커 포함'(=시스템 전체).
                           # True로 켜면 이중 리랭크. 도입 확정이라 내부 1회로 충분 → False 유지.
SMOKE = int(os.getenv("SMOKE", "0")) or None   # 스모크셋 크기 (0/미설정=전체) — 검증용 부분 실행 (#18)


# ===== 컨텍스트 구성 =================================================

async def oracle_context(session, chunk_ids: list[int]) -> list[RetrievedChunk]:
    """gold 정답 청크 id들을 RetrievedChunk로 (LLM 컨텍스트 주입용)."""
    if not chunk_ids:
        return []
    rows = await session.execute(
        select(Chunk, Document.filename, Document.version)
        .join(Document, Chunk.document_id == Document.id)
        .where(Chunk.id.in_(chunk_ids))
    )
    return [
        RetrievedChunk(chunk_id=c.id, document_id=c.document_id, text=c.text,
                       heading_path=c.heading_path, page=c.page, filename=fn, version=v)
        for c, fn, v in rows.all()
    ]


async def retrieved_context(session, tenant: str, query: str) -> list[RetrievedChunk]:
    """실제 검색기 결과 top-K (옵션: 리랭커 재정렬)."""
    cands = await retrieve_candidates(session, tenant, query, top_n=20)
    chunks = cands.chunks
    if USE_RERANK and chunks:
        from rag.reranker import rerank
        chunks = await rerank(query, chunks)
    return chunks[:TOP_K]


# ===== 생성 ==========================================================

async def generate(llm: LlmClient, original_query: str, chunks: list[RetrievedChunk],
                   standalone_query: str | None = None,
                   prior_turns: list[dict] | None = None) -> str:
    """RAG 프롬프트로 답변 1건 생성 (LLM 서버 필요).

    운영(rag/service.py)과 같은 조립점을 쓴다 — 인자를 한쪽에서만 빼먹는 종류의 어긋남을
    구조적으로 막기 위해서다(#48: 여기서 prior_turns가 빠져 있었고 아무 데도 안 걸렸다).

    출처 꼬리 제약도 운영과 같이 붙인다 (#61) — 이전엔 하네스만 제약 없이 생성했다.
    첨부는 []다 — gold set에 첨부 문항이 0건이다.

    #65부터 이 제약이 **실제로 강제된다**(#56 도입 이후 처음). 그전 구현(정규식)은 vLLM에서
    무효였고, 그래서 붙이기 전후 위반율이 같았다(oracle 1.1%·retrieved 1.6%) — 경위와 사다리
    실측은 rag/prompts.build_citation_constraint docstring. 이 하네스가 그 강제 효과를 재는
    1차 지표다: 위반율이 0으로 떨어지는지가 검증 대상이다.

    fail-open: 서버가 제약을 거부하면(400/422) 제약 없이 1회 재시도한다.
    400/422로 좁힌 것은 rag/llm_schemas.py의 선례를 따른 것이다 — 운영 답변 생성 경로
    (rag/service.py의 astream)는 except Exception으로 넓게 잡지만, 그쪽은 첫 토큰을 당기는
    시점이라 재시도 비용이 0에 가깝다. acomplete는 블로킹이라 넓게 잡으면 타임아웃 대기가
    배가된다. 호출 형태가 같은 쪽(acomplete_validated)의 판단을 따른다.
    주의: response_format을 모르는 구버전 서버가 어떤 상태코드로 거부할지는 **미검증**이다
    (그런 서버가 없다). 400/422 밖이면 이 필터를 통과하지 못해 예외가 그대로 전파된다 —
    조용히 삼키는 것보다 시끄럽게 실패하는 쪽이 안전한 기본값이라 넓히지 않았다.
    """
    prompt = build_knowledge_generation_prompt(
        original_query, chunks, standalone_query=standalone_query, prior_turns=prior_turns)
    constraint = build_citation_constraint(sources_from_chunks(chunks), [])
    try:
        return await llm.acomplete(prompt, extra_body=constraint)
    except Exception as exc:
        if not is_schema_rejected(exc):
            raise
        print('  [warn] 꼬리 제약 거부 — 제약 없이 재시도 (fail-open, #61·#65)')
        return await llm.acomplete(prompt)


# ===== 채점 — deterministic (서버 무관) ==============================

EPCOV_SIM_THRESHOLD = 0.55   # 기대포인트 ↔ 답변문장 코사인 임계 (oracle 상한 보정으로 결정)


def _split_sentences(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"[.!?\n]+", text) if p.strip()]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def expected_points_coverage(answer: str, points: list[str]) -> float | None:
    """기대 포인트가 답변에 담긴 비율. 부분문자열 → 임베딩 2단 매칭.

    - 부분문자열: "30분"·"12개월" 같은 짧은 수치 포인트용. 임베딩 코사인은
      짧은 포인트 ↔ 긴 문장 대조에서 임계(0.55)를 못 넘어 과소평가한다
      (답변 길이 제약 해제 후 실측 — 정답인데 EPCov=0인 아티팩트 다수).
    - 임베딩: 서술형 포인트("본인 인증 후 재설정 링크")용. 부분문자열은 이걸 놓친다.
    둘 중 하나라도 걸리면 커버 — 각 방식의 사각을 상호 보완.
    """
    if not points:
        return None
    norm_answer = re.sub(r"\s+", "", answer)
    hits = [_contains_point(norm_answer, re.sub(r"\s+", "", p)) for p in points]

    remaining_idx = [i for i, h in enumerate(hits) if not h]
    if remaining_idx:
        sents = _split_sentences(answer)
        if sents:
            remaining = [points[i] for i in remaining_idx]
            embs = embed_texts_sync(remaining + sents)
            p_vecs = [e.dense for e in embs[:len(remaining)]]
            s_vecs = [e.dense for e in embs[len(remaining):]]
            for i, pv in zip(remaining_idx, p_vecs):     # index 직접 — 중복 포인트도 각자 갱신
                if max(_cosine(pv, sv) for sv in s_vecs) >= EPCOV_SIM_THRESHOLD:
                    hits[i] = True
    return sum(hits) / len(points)


def _contains_point(norm_answer: str, norm_point: str) -> bool:
    """숫자 경계를 지키는 부분문자열 매칭 — '30분'이 '130분'/'305분'에 오탐되는 것 방지.
    콤마는 양쪽에서 제거 — xlsx 숫자셀('38000')과 모델 표기('38,000원')의 표기 차이 흡수."""
    norm_answer = norm_answer.replace(',', '')
    norm_point = norm_point.replace(',', '')
    if not norm_point:
        return False
    for m in re.finditer(re.escape(norm_point), norm_answer):
        before = norm_answer[m.start() - 1] if m.start() > 0 else ''
        after = norm_answer[m.end()] if m.end() < len(norm_answer) else ''
        if norm_point[0].isdigit() and before.isdigit():
            continue
        if norm_point[-1].isdigit() and after.isdigit():
            continue
        return True
    return False


def _citation_match(core: str, stem: str) -> bool:
    """인용 토큰 코어 ↔ 기대 파일명 stem 매칭.

    허용: 정확 일치 / 언더스코어 마지막 조각 일치(kms_03_배송지연대응 ↔ 배송지연대응) /
          4자 이상일 때의 포함 관계 (짧은 토큰의 우연 매칭 오탐 방지).
    """
    if not core or not stem:
        return False
    if core == stem:
        return True
    if core == stem.split('_')[-1] or stem == core.split('_')[-1]:
        return True
    shorter, longer = sorted((core, stem), key=len)
    return len(shorter) >= 4 and shorter in longer


def citation_accuracy(answer: str, expected_docs: list[str], chunks: list[RetrievedChunk]) -> float:
    r"""기대 문서 중 답변이 실제로 인용한 비율 (0..1).

    v5 (#65, **v4 이전 결과와 비교 불가**): 꼬리 제약이 **처음으로 실제 강제된다**
    (정규식 → structural_tag, #56 도입 이후 최초). 채점 코드는 무변경 — 여전히 운영 배관
    (TailSplitter → resolve_citations)을 그대로 쓰고, resolve_citations의 `\d+` 추출이
    대괄호·쉼표·여백을 무시하므로 꼬리 형식이 `««1,3»»`에서 `««[1,3]»»`로 바뀐 것도
    파싱에 영향이 없다. 바뀌는 것은 **입력 분포**다: 유효 범위 밖 번호가 원천 차단되므로
    "꼬리가 범위 밖 번호뿐이라 인용 0건이 되던" 위반(v4 시점 oracle 1.1%·retrieved 1.6%)이
    사라진다. 수치가 오르면 그 강제 효과이고, 무변동이면 대다수가 이미 준수했다는 뜻이다 —
    어느 쪽도 채점 정의 변경이 아니다.

    v4 (#61, **v3 이전 결과와 비교 불가**): 두 가지가 운영과 일치하게 바뀌었다.
      1) 꼬리 분리를 운영 배관(TailSplitter)으로 교체 — 이전엔 rsplit/split으로 직접
         잘랐고, 그건 운영의 오탐 복구를 갖지 않았다. 원리적으로 갈리는 케이스:
         `««1»» 추가 설명` → 옛 파싱 '1' / 운영 None(END 뒤 텍스트 = 오탐),
         `««1,2 그리고 3개 항목이…`(END 없음) → 옛 파싱이 **본문에서 숫자를 뽑는다** / 운영 None.
         실데이터 450문항 전수 대조에서는 불일치 0이었지만(모델이 얌전했다는 뜻),
         채점 정의가 운영과 달랐다는 사실은 그대로다.
      2) generate()가 운영과 같은 꼬리 제약을 붙인다 — 생성 조건 변경(위 generate 참조).
    채점 정의가 바뀐 것이지 모델이 좋아지거나 나빠진 게 아니다.

    v3 (#56, **v2 이전 결과와 비교 불가**): 인용이 본문 인라인에서 답변 끝 출처 꼬리
    (TAIL_START…TAIL_END, 번호 목록)로 이동 — 운영과 같은 배관(sources_from_chunks →
    resolve_citations)으로 번호를 파일명으로 되돌려 채점한다. 채점기가 배관을 따로 들면
    운영과 다른 번호 해석이 생길 수 있다(단일 정의점 원칙). #56 이전 아카이브
    (report_generation_v1 등)의 Cite 수치와 이 채점기의 수치를 나란히 놓지 말 것 —
    채점 정의가 바뀐 것이지 모델이 좋아지거나 나빠진 게 아니다. 꼬리가 없으면
    (형식 미준수) 인용 0으로 집계 — 그게 사실이다.
    eval/rescore_v2.py는 v2 라벨 꼬리 전용이라 이 시그니처로는 동작하지 않는다(의도적).

    v2 개정 (기존 0/1 any-match와 비교 불가):
    - any-match는 multi_doc에서 문서 하나만 인용해도 만점 → 커버리지 비율로 교체.
    - FAQ 출처는 'FAQ'로 인용됨 — 기대에 'FAQ'를 넣으면 정확 매칭.
    - 확장자 스트립을 5개 형식 전체로 확장 (corpus v2 대응).
    """
    if not expected_docs:
        return 0.0
    # 꼬리 분리는 운영 배관 그대로 (#61) — 완성된 문자열을 한 청크로 흘려 넣는다.
    # feed/finish의 계약상 부분 스트리밍과 결과가 같다: 보류(_hold)는 finish가 비워내고,
    # 오탐 복구 판정은 누적 버퍼 기준이라 청크 경계와 무관하다.
    splitter = TailSplitter()
    splitter.feed(answer)
    splitter.finish()
    cited = resolve_citations(splitter.tail_raw, sources_from_chunks(chunks), [])
    # 기대 stem과 같은 규칙으로 확장자 제거 — '공지.txt' vs '공지'는 4자 미만이라 포함 매칭도 못 탄다
    cores = [re.sub(r"\.(pdf|docx|xlsx|txt|md)$", "", c.filename) for c in cited]
    # 원소가 리스트면 대체 출처 그룹 — 같은 정보가 여러 문서에 있을 때 하나만 인용해도 인정
    groups = [[d] if isinstance(d, str) else d for d in expected_docs]
    covered = 0
    for group in groups:
        stems = [re.sub(r"\.(pdf|docx|xlsx|txt|md)$", "", d) for d in group]
        if any(_citation_match(core, stem) for stem in stems for core in cores):
            covered += 1
    return covered / len(groups)


# ===== 채점 — LLM-judge (서버 + 심판모델, 현재 stub) =================

async def judge_relevancy(query: str, answer: str) -> float | None:
    """Answer Relevancy (0/0.5/1). TODO: 심판 LLM 프롬프트로 채점."""
    return None


async def judge_faithfulness(answer: str, chunks: list[RetrievedChunk]) -> float | None:
    """Faithfulness/Groundedness (0/0.5/1). TODO: 심판 LLM으로 답변 주장의 근거 일치 채점."""
    return None


# ===== 메인 ==========================================================

CONCURRENCY = 6   # worker15 공유 장비 — RAGAS max_workers와 동일 안전선 (#18)


async def run_mode(session, llm, mode: str, gold_rows, resolved):
    """oracle / retrieved 한 모드로 생성 + 채점 → row 리스트.

    LLM 콜(condense·generate)은 병렬(vLLM 연속 배칭, #18), 컨텍스트 조회·채점은 직렬 —
    AsyncSession은 동시 실행 불가라 gather에 태우지 않는다.
    """
    gen_rows = [g for g in gold_rows if g["type"] in GEN_TYPES]
    sem = asyncio.Semaphore(CONCURRENCY)

    # ① condense 선병렬 (multi_turn만 — DB 무관).
    # multi_turn: 이전 대화를 운영 condense에 태워 독립 질문으로 재작성.
    # **검색**은 재작성 질의를 쓰고, **생성**은 원 질문 + 재작성 참고 병기 — 운영 경로와 동일 (#48).
    # 이력 블록(prior_turns)도 운영과 같은 함수·같은 예산으로 만들어 함께 싣는다.
    async def _standalone(g):
        if g["type"] != "multi_turn":
            return g["id"], (None, [])
        history = [SimpleNamespace(**m) for m in (g.get("conversation") or [])]
        async with sem:
            standalone = await condense_query(llm, g["query"], history)
        return g["id"], (standalone, build_prior_turns(history, settings.history_budget_tokens))

    # {문항 id: (재작성 질의 | None, 이력 턴)} — multi_turn만 값이 채워진다
    turn_ctx = dict(await asyncio.gather(*(_standalone(g) for g in gen_rows)))

    # ② 컨텍스트 조회 — 세션 직렬. oracle에서 resolve 실패는 스킵(기존 동작)
    work: list[tuple[dict, list]] = []
    for g in gen_rows:
        query = turn_ctx[g["id"]][0] or g["query"]     # 검색은 재작성 질의로 (없으면 원문)
        if mode == "oracle":
            chunks = await oracle_context(session, resolved.chunk_ids.get(g["id"]) or [])
            if not chunks:                       # 정답 청크 resolve 실패 → oracle 스킵
                continue
        else:
            chunks = await retrieved_context(session, row_tenant(g), query)
        work.append((g, chunks))

    # ③ 생성 병렬 — 가장 무거운 구간이라 병렬화 효과 최대
    async def _generate(g, chunks):
        standalone, prior_turns = turn_ctx[g["id"]]
        async with sem:
            return await generate(llm, g["query"], chunks,
                                  standalone_query=standalone, prior_turns=prior_turns)

    answers = await asyncio.gather(*(_generate(g, c) for g, c in work))

    # ④ 채점·행 구성 — 직렬 (EPCov의 embed_texts_sync 포함)
    rows = []
    for (g, chunks), answer in zip(work, answers):
        standalone = turn_ctx[g["id"]][0]

        rows.append({
            "id": g["id"], "type": g["type"], "mode": mode,
            "standalone_query": standalone,      # multi_turn만 값 있음 — condense 품질 분석용
            "answer": answer,
            # RAGAS retrieved_contexts용 — 생성기가 본 것과 동일하게 [파일명 vN] 헤더 포함
            # (헤더 없이 text만 저장하면 답변의 인용 문장이 judge에게 근거없음 판정됨)
            "retrieved_contexts": [
                f"[{c.filename} v{c.version}] 섹션: "
                f"{' > '.join(c.heading_path) if c.heading_path else ''} / 페이지: {c.page or '-'}\n{c.text}"
                for c in chunks
            ],
            "scores": {
                "expected_points_coverage": expected_points_coverage(answer, g.get("expected_points", [])),
                "citation_accuracy": citation_accuracy(answer, g.get("expected_docs", []), chunks),
                "answer_relevancy": await judge_relevancy(g["query"], answer),
                "faithfulness": await judge_faithfulness(answer, chunks),
            },
            "gold_version": GOLD_VERSION,
        })
    return rows


async def main():
    gold = [json.loads(l) for l in GOLD.read_text().splitlines() if l.strip()]
    gen_gold = [g for g in gold if g["type"] in GEN_TYPES]
    if SMOKE:
        gen_gold = gen_gold[:SMOKE]
    RESULT_DIR.mkdir(exist_ok=True)

    llm = LlmClient()
    async with AsyncSessionLocal() as session:
        # 테넌트별 resolve 후 병합 (chunk_ids는 문항 id 키라 충돌 없음)
        from eval.retrieval import Resolved
        resolved = Resolved()
        by_tenant = defaultdict(list)
        for g in gold:
            by_tenant[row_tenant(g)].append(g)
        for tenant, rows_t in by_tenant.items():
            r_t = await resolve_gold(session, tenant, rows_t)
            resolved.chunk_ids.update(r_t.chunk_ids)
            resolved.stale.extend(f"[{tenant}] {m}" for m in r_t.stale)
        if resolved.stale:
            print(f"⚠ 라벨 노후 {len(resolved.stale)}건")
            for m in resolved.stale[:10]:
                print("  ", m)

        all_rows = {}
        for mode in ("oracle", "retrieved"):
            print(f"\n=== mode: {mode} ({len(gen_gold)}문항) ===")
            rows = await run_mode(session, llm, mode, gen_gold, resolved)
            if not rows:   # 채점 0 = resolve 전멸 (빈/잘못된 DB) — 재료 파일 덮어쓰기 방지 (#18 실사고 가드)
                raise SystemExit(f"{mode}: 생성 0행 — DATABASE_URL이 코퍼스 있는 DB인지 확인. 결과 파일 미변경.")
            out = RESULT_DIR / f"generation_{mode}.jsonl"
            out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows))
            summarize(rows)
            print(f"→ {out} ({len(rows)} rows)")
            all_rows[mode] = rows


def summarize(rows):
    """모드별 deterministic 평균 (judge는 채점 붙으면 추가)."""
    by_type = defaultdict(list)
    for r in rows:
        by_type[r["type"]].append(r["scores"])
    print(f"{'type':<14}{'n':>4}{'EPCov':>8}{'Cite':>7}")
    for t, ss in by_type.items():
        n = len(ss)
        cov = [s["expected_points_coverage"] for s in ss if s["expected_points_coverage"] is not None]
        cite = [s["citation_accuracy"] for s in ss]
        cov_avg = sum(cov) / len(cov) if cov else 0.0
        print(f"{t:<14}{n:>4}{cov_avg:>8.2f}{sum(cite)/n:>7.2f}")


if __name__ == "__main__":
    asyncio.run(main())
