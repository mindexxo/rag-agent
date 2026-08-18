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

from database import AsyncSessionLocal
from rag.citation_labels import TAIL_END, TAIL_START, sources_from_chunks
from rag.citation_tail import resolve_citations
from rag.conversation import condense_query
from rag.models import Chunk, Document
from rag.retriever import retrieve_candidates, RetrievedChunk
from rag.embeddings import embed_texts_sync
from rag.prompts import SYSTEM_PROMPT, build_chat_prompt, build_user_message
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

async def generate(llm: LlmClient, query: str, chunks: list[RetrievedChunk]) -> str:
    """RAG 프롬프트로 답변 1건 생성 (LLM 서버 필요)."""
    messages = build_chat_prompt(SYSTEM_PROMPT, build_user_message(query, chunks))
    return await llm.acomplete(messages)


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
    """기대 문서 중 답변이 실제로 인용한 비율 (0..1).

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
    tail = answer.rsplit(TAIL_START, 1)[1] if TAIL_START in answer else None
    if tail is not None:
        tail = tail.split(TAIL_END, 1)[0]
    cited = resolve_citations(tail, sources_from_chunks(chunks), [])
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
    # 검색·생성 모두 재작성된 질의를 쓴다 — 운영 /kms/query 경로와 동일.
    async def _standalone(g):
        if g["type"] != "multi_turn":
            return g["id"], None
        history = [SimpleNamespace(**m) for m in g.get("conversation", [])]
        async with sem:
            return g["id"], await condense_query(llm, g["query"], history)

    standalone_map = dict(await asyncio.gather(*(_standalone(g) for g in gen_rows)))

    # ② 컨텍스트 조회 — 세션 직렬. oracle에서 resolve 실패는 스킵(기존 동작)
    work: list[tuple[dict, list]] = []
    for g in gen_rows:
        query = standalone_map[g["id"]] or g["query"]
        if mode == "oracle":
            chunks = await oracle_context(session, resolved.chunk_ids.get(g["id"]) or [])
            if not chunks:                       # 정답 청크 resolve 실패 → oracle 스킵
                continue
        else:
            chunks = await retrieved_context(session, row_tenant(g), query)
        work.append((g, chunks))

    # ③ 생성 병렬 — 가장 무거운 구간이라 병렬화 효과 최대
    async def _generate(g, chunks):
        async with sem:
            return await generate(llm, standalone_map[g["id"]] or g["query"], chunks)

    answers = await asyncio.gather(*(_generate(g, c) for g, c in work))

    # ④ 채점·행 구성 — 직렬 (EPCov의 embed_texts_sync 포함)
    rows = []
    for (g, chunks), answer in zip(work, answers):
        standalone = standalone_map[g["id"]]

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
