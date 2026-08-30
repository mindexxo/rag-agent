"""롱컨텍스트 통짜 비교군 (#96 (b)) — 파이프라인 없이 "문서 전부 + 질문"을 Claude에.

무엇을 재나: "검색이니 리랭크니 없이 프론티어 모델에 코퍼스를 통째로 넣으면 되지 않나?"
라는 질문에 대한 실측 답. (a) 생성기 교체(GENERATOR=claude python -m eval.generation)가
파이프라인을 고정하고 모델 격차를 재는 것과 반대로, 여기는 **파이프라인 전체를 제거**해
파이프라인의 가치를 잰다. 같은 골드셋·같은 채점(EPCov)이라 두 결과는 나란히 놓을 수 있다.

파이프라인 부재의 구체적 의미 — 전부 의도된 조건이다:
- 검색·리랭크 없음: 테넌트 문서 전문(색인과 같은 원료 — 청크를 chunk_index로 이어붙인
  추출 텍스트) + FAQ 전부를 프롬프트에 싣는다.
- condense 없음: multi_turn은 원 대화를 그대로 싣는다 — "파이프라인 없음"의 정직한 형태.
- 출처 꼬리 없음: 청크 번호가 존재하지 않으므로 ««[n]»» 채점을 못 쓴다. 대신
  '출처: [파일명, ...]' 형식을 프롬프트로 유도해 파싱한다(형식 강제 불가 — structural_tag는
  vLLM 배관이다. 미준수는 인용 0으로 집계 — (a)와 같은 정직 원칙, 결과 각주에 명시).

의존: DB(코퍼스 로드) + TEI(EPCov 임베딩) + claude CLI 로그인. vLLM 불필요.
실행: python -m eval.longcontext_claude   (SMOKE=n 지원 — 타입별 균등 표본)
"""
import asyncio
import json
import os
import re
from collections import defaultdict
from pathlib import Path

from sqlalchemy import select

from database import AsyncSessionLocal
from eval.claude_client import ClaudeCliClient
from eval.generation import (GEN_TYPES, GOLD, GOLD_VERSION, RESULT_DIR,
                             _citation_match, _smoke_sample,
                             expected_points_coverage, row_tenant, summarize)
from rag.models import Chunk, Document

SMOKE = int(os.getenv("SMOKE", "0")) or None
CONCURRENCY = 4          # claude -p 서브프로세스 병렬 상한 — vLLM 배칭과 달리 콜=프로세스
OUT = RESULT_DIR / "generation_longcontext_claude.jsonl"

# 코퍼스 크기 경고선(문자). 한국어는 대략 1.5~2자/토큰 — 30만 자면 15~20만 토큰으로
# 모델 컨텍스트에 육박한다. 넘어도 실행은 한다(모델 에러가 그 자체로 측정 결과다).
CORPUS_CHAR_WARN = 300_000

SYSTEM_PROMPT = """당신은 고객센터 상담사를 돕는 지식 어시스턴트입니다.

규칙:
1. 아래에 제공되는 문서들의 내용만 근거로 답합니다. 문서에 없는 내용은 지어내지 말고
   "제공된 문서에서 확인할 수 없습니다"라고 답합니다.
2. 답변 마지막 줄에 실제로 근거로 쓴 문서의 파일명을 다음 형식으로 적습니다.
   출처: [파일명1, 파일명2]
   근거로 쓴 문서가 없으면: 출처: []
3. 상담사가 고객에게 바로 전달할 수 있게 정확한 수치·조건을 그대로 답합니다."""


# ===== 코퍼스 로드 =====================================================

async def load_corpus(session, tenant: str) -> str:
    """테넌트의 검색 대상 전체를 하나의 텍스트로 — 색인 파이프라인과 같은 원료.

    문서: ready·is_active·is_searchable (운영 검색 필터와 동일 조건) 청크를
    chunk_index 순으로 이어붙임. FAQ: faq_id 청크 전부를 'FAQ' 블록으로
    (운영 인용 라벨이 'FAQ'인 것과 짝 — eval.generation.citation_accuracy v2 참조).
    """
    doc_rows = (await session.execute(
        select(Document.id, Document.filename)
        .where(Document.tenant_id == tenant, Document.status == "ready",
               Document.is_active, Document.is_searchable)
        .order_by(Document.filename)
    )).all()
    doc_ids = {d_id: fn for d_id, fn in doc_rows}

    parts: list[str] = []
    if doc_ids:
        chunk_rows = (await session.execute(
            select(Chunk.document_id, Chunk.text)
            .where(Chunk.document_id.in_(doc_ids))
            .order_by(Chunk.document_id, Chunk.chunk_index)
        )).all()
        by_doc: dict[int, list[str]] = defaultdict(list)
        for d_id, text in chunk_rows:
            by_doc[d_id].append(text)
        for d_id, fn in doc_ids.items():
            parts.append(f"=== 문서: {fn} ===\n" + "\n".join(by_doc.get(d_id, [])))

    faq_rows = (await session.execute(
        select(Chunk.text)
        .where(Chunk.tenant_id == tenant, Chunk.faq_id.is_not(None))
        .order_by(Chunk.faq_id)
    )).scalars().all()
    if faq_rows:
        parts.append("=== 문서: FAQ ===\n" + "\n\n".join(faq_rows))
    return "\n\n".join(parts)


# ===== 프롬프트·채점 ===================================================

def build_user_message(corpus: str, g: dict) -> str:
    """문서 전문 + (원 대화 그대로) + 질문. condense·검색이 없는 것이 이 축의 정의다."""
    blocks = [f"[지식 문서 전체]\n\n{corpus}"]
    conv = g.get("conversation") or []
    if conv:
        lines = [f"{'상담사' if m['role'] == 'user' else '어시스턴트'}: {m['content']}"
                 for m in conv]
        blocks.append("[이전 대화]\n" + "\n".join(lines))
    blocks.append(f"[질문]\n{g['query']}")
    return "\n\n".join(blocks)


_SOURCE_LINE = re.compile(r"출처\s*[:：]\s*\[(.*?)\]")


def parse_cited_files(answer: str) -> list[str]:
    """답변의 '출처: [...]' 줄에서 파일명 목록을 뽑는다 — 마지막 출현 기준(본문 인용 오탐 회피).

    형식 미준수(줄 자체가 없음)는 빈 목록 = 인용 0 집계. 부분 복구는 하지 않는다 —
    citation_tail과 같은 원칙(그럴듯한 복구는 실패를 지표에서 숨긴다).
    """
    matches = _SOURCE_LINE.findall(answer)
    if not matches:
        return []
    return [t.strip().strip("'\"") for t in matches[-1].split(",") if t.strip()]


def filename_citation_accuracy(cited: list[str], expected_docs: list[str]) -> float:
    """기대 문서 중 파일명 인용이 커버한 비율 — citation_accuracy의 그룹·매칭 규칙을
    파일명 직접 비교로 옮긴 것(청크 번호 배관이 없는 이 축 전용)."""
    if not expected_docs:
        return 0.0
    cores = [re.sub(r"\.(pdf|docx|xlsx|txt|md)$", "", c) for c in cited]
    groups = [[d] if isinstance(d, str) else d for d in expected_docs]
    covered = 0
    for group in groups:
        stems = [re.sub(r"\.(pdf|docx|xlsx|txt|md)$", "", d) for d in group]
        if any(_citation_match(core, stem) for stem in stems for core in cores):
            covered += 1
    return covered / len(groups)


# ===== 메인 ============================================================

async def main():
    gold = [json.loads(l) for l in GOLD.read_text().splitlines() if l.strip()]
    gen_gold = [g for g in gold if g["type"] in GEN_TYPES]
    if SMOKE:
        gen_gold = _smoke_sample(gen_gold, SMOKE)
    RESULT_DIR.mkdir(exist_ok=True)

    client = ClaudeCliClient(model=os.getenv("CLAUDE_MODEL", "sonnet"))
    sem = asyncio.Semaphore(CONCURRENCY)

    # 테넌트별 코퍼스 선로드 (DB 직렬) — 문항마다 다시 읽지 않는다
    corpora: dict[str, str] = {}
    async with AsyncSessionLocal() as session:
        for tenant in sorted({row_tenant(g) for g in gen_gold}):
            corpus = await load_corpus(session, tenant)
            if not corpus:
                raise SystemExit(f"{tenant}: 코퍼스 0자 — DATABASE_URL이 코퍼스 있는 DB인지 확인.")
            corpora[tenant] = corpus
            flag = " ⚠ 컨텍스트 초과 가능" if len(corpus) > CORPUS_CHAR_WARN else ""
            print(f"[corpus] {tenant}: {len(corpus):,}자{flag}")

    async def _run(g):
        async with sem:
            answer = await client.acomplete([
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_message(corpora[row_tenant(g)], g)},
            ])
        cited = parse_cited_files(answer)
        return {
            "id": g["id"], "type": g["type"], "mode": "longcontext",
            "answer": answer,
            "cited_files": cited,
            "scores": {
                # EPCov는 TEI 임베딩을 쓰는 동기 함수 — 직렬화를 위해 락 밖이 아닌 여기서
                # 호출하지 않고 아래 채점 루프에서 한다. (answer만 여기서 만든다)
                "expected_points_coverage": None,   # 채점 루프에서 채움
                "citation_accuracy": filename_citation_accuracy(cited, g.get("expected_docs", [])),
            },
            "generator": f"claude:{client.model}",
            "gold_version": GOLD_VERSION,
        }

    print(f"\n=== longcontext ({len(gen_gold)}문항, model={client.model}) ===")
    rows = await asyncio.gather(*(_run(g) for g in gen_gold))

    # 채점 — 직렬 (EPCov의 embed_texts_sync가 동기 TEI 콜이라 gather에 안 태운다)
    by_id = {g["id"]: g for g in gen_gold}
    for r in rows:
        r["scores"]["expected_points_coverage"] = expected_points_coverage(
            r["answer"], by_id[r["id"]].get("expected_points", []))

    OUT.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows))
    summarize(rows)
    print(f"→ {OUT} ({len(rows)} rows)")


if __name__ == "__main__":
    asyncio.run(main())
