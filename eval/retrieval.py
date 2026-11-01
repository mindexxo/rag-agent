"""검색 채점 공용부 — 골드의 안정 키(파일명·스니펫)를 현재 문서·청크 id로 바꾸고, 한 문항의 Recall@k·Hit@1·MRR을 계산한다."""

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from config import settings
from eval.rows import ExpectedChunk, GoldRow
from rag.models import Document
from rag.search import os_client


@dataclass
class Resolved:
    """gold의 안정 키를 현재 id(문서=PG id, 청크=색인 chunk_id)로 변환한 결과 묶음."""

    doc_ids: dict[str, int] = field(default_factory=dict)
    chunk_ids: dict[str, list[int]] = field(default_factory=dict)  # 스니펫당 **첫 매치** 1개 — oracle 주입용(순서 보존)
    # 스니펫을 포함한 청크 **전부** — 검색축 채점용. 같은 문장이 여러 쪽에 중복된 문서에서 다른 중복 청크가 1위면
    # chunk_ids만으론 miss로 찍히고, 어느 쪽이 1위인지는 재인제스트마다 뒤집힌다.
    # oracle엔 쓰지 않는다 — 같은 문장 청크를 두 번 주입하면 토큰 낭비이고 생성 oracle 기준선이 흔들린다.
    scoring_ids: dict[str, set[int]] = field(default_factory=dict)
    stale: list[str] = field(default_factory=list)


def _norm_snippet(text: str) -> str:
    return re.sub(r"[\s|]+", "", text)  # 검증기와 동일 정규화 (공백+파이프) — PDF/DOCX 추출은 공백 위치가 원본과 다르다


def match_expected_chunks(
    chunk_rows: list[tuple[int, list[str], str]], expected_chunks: list[ExpectedChunk]
) -> tuple[list[int], set[int], list[str]]:
    """expected_chunks → (주 정답 id 목록, 채점용 id 집합, 못 찾은 스니펫 목록). DB 없는 순수 함수 — 테스트는 튜플만 먹인다.

    주 정답 = 스니펫당 하나: heading_path가 일치하는 매치가 있으면 그것, 없으면 첫 매치(현행 규칙 그대로, oracle용).
    채점 집합 = 스니펫을 포함한 청크 전부. snippet 포함이 정답 판정의 최종 검증자라는 원칙은 같고,
    "포함한 청크가 여럿이면 전부 근거"라는 당연한 귀결을 채점에만 적용한다.
    """
    primary: list[int] = []
    all_ids: set[int] = set()
    missing: list[str] = []
    for ec in expected_chunks:
        snip = _norm_snippet(ec["snippet"])
        hits = [(cid, hp) for cid, hp, text in chunk_rows if snip in _norm_snippet(text)]
        if not hits:
            missing.append(ec["snippet"][:20])
            continue
        want_hp = ec.get("heading_path")  # v2 스키마엔 없음 (PDF/DOCX 헤딩 미보존)
        primary.append(next((cid for cid, hp in hits if want_hp and hp == want_hp), hits[0][0]))
        all_ids.update(cid for cid, _ in hits)
    return primary, all_ids, missing


async def _indexed_chunks(document_id: int) -> list[tuple[int, list[str], str]]:
    """문서의 색인 청크 (chunk_id, heading_path, text). 청크는 OpenSearch에만 있다."""
    resp = await os_client.client().search(
        index=settings.opensearch_index,
        body={
            "size": 10000,  # 문서 하나의 청크 수 상한으로 충분 (실문서 최대 86)
            "_source": ["chunk_id", "heading_path", "text"],
            "query": {"term": {"document_id": document_id}},
        },
    )
    return [
        (int(h["_source"]["chunk_id"]), list(h["_source"].get("heading_path") or []), h["_source"].get("text") or "")
        for h in resp["hits"]["hits"]
    ]


# 안정 키 -> 현재 id
async def resolve_gold(session: AsyncSession, tenant_id: str, gold_rows: Sequence[GoldRow]) -> Resolved:
    """gold set의 안정 키(filename/heading_path/snippet)를 현재 id로 resolve.

    gold는 id가 아닌 내용 기준 안정 키로 정답을 적어둔다 (재인제스천마다 id가
    바뀌므로 — §4.1). 하지만 Recall/Hit/MRR 계산은 검색 결과의 id와 비교해야
    하므로, 평가 본 계산 전에 안정 키 -> 현재 id 변환이 선행돼야 한다.
    문서 id는 PG(documents), 청크 id는 검색 색인(chunk_os_id)에서 온다 — 검색기가 돌려주는
    `RetrievedChunk.chunk_id`가 색인 id라 채점의 양변이 같은 공간이다.

    resolve 실패(문서 개정·청킹 변경으로 라벨이 낡음)는 조용히 0점 처리하지 않고
    stale에 모아 경보로 올린다 — "검색 실패"와 "라벨 노후"를 구분하기 위함.
    """
    r = Resolved()

    # 1) filename -> active/ready doc id  (한 번에 조회)
    # expected_docs 원소는 str 또는 [str,...] (대체 출처 그룹) — 평탄화해 조회
    filenames = {fn for row in gold_rows for d in row["expected_docs"] for fn in ([d] if isinstance(d, str) else d)}
    doc_rows = (
        await session.execute(
            select(Document.filename, Document.id)
            .where(Document.tenant_id == tenant_id)
            .where(Document.is_active.is_(True))
            .where(Document.status == "ready")
            .where(Document.filename.in_(filenames))
        )
    ).all()
    r.doc_ids = {fn: did for fn, did in doc_rows}

    # 2) expected_chunks -> chunk id. 매칭 규칙은 match_expected_chunks가 정의점 — 문서별로 색인 청크를 한 번 읽어 넘긴다.
    chunk_cache: dict[int, list[tuple[int, list[str], str]]] = {}
    for row in gold_rows:
        primary: list[int] = []
        scoring: set[int] = set()
        for ec in row.get("expected_chunks", []):
            doc_id = r.doc_ids.get(ec["filename"])
            if doc_id is None:
                r.stale.append(f'{row["id"]}: doc 못찾음 {ec["filename"]}')
                continue
            if doc_id not in chunk_cache:
                chunk_cache[doc_id] = await _indexed_chunks(doc_id)
            p, a, missing = match_expected_chunks(chunk_cache[doc_id], [ec])
            primary.extend(p)
            scoring.update(a)
            r.stale.extend(f'{row["id"]}: chunk 못찾음 {m}' for m in missing)
        r.chunk_ids[row["id"]] = primary
        r.scoring_ids[row["id"]] = scoring
    return r


# 한 문항 검색 채점
def score_one(got_ids: list[int], gold_ids: list[int], ks: Iterable[int] = (5, 20)) -> dict | None:
    """검색 결과 순위(got)와 정답 id(gold)로 Recall@k·Hit@1·MRR 계산.

    got_ids: 검색기가 가져온 id, RRF 순 (1등이 [0]).
    gold_ids: resolve된 정답 id. 비어 있으면 채점 불가 → None (집계 제외).
    """
    if not gold_ids:
        return None
    gold = set(gold_ids)

    # Recall@k: 정답이 상위 k개 안에 하나라도 있으면 1, 없으면 0
    scores = {f"recall_at_{k}": float(bool(gold & set(got_ids[:k]))) for k in ks}

    # Hit@1: 1등(got_ids[0])이 정답인가
    scores["hit_at_1"] = float(bool(got_ids and got_ids[0] in gold))

    # MRR: 정답이 처음 등장하는 순위의 역수 (1등=1.0, 3등=0.33, 못 찾으면 0)
    rank = next((i for i, cid in enumerate(got_ids, 1) if cid in gold), None)
    scores["mrr"] = 1.0 / rank if rank else 0.0

    return scores
