"""PG → OpenSearch 색인 + 배선 검증 게이트 (#139).

매핑·질의 조립의 정의점은 **`rag/opensearch.py`** 다(운영 백엔드와 같은 것을 쓴다 —
eval이 재는 것과 운영이 도는 것이 갈라지면 눈금이 운영을 예언하지 못한다).
이 파일은 그 인덱스를 PG에서 채우고, 채운 결과를 값으로 검증한다.

**임베딩을 재계산하지 않는다.** `chunks.dense`를 그대로 복사한다 — TEI는 호출마다
비결정적이라(실측 1.4e-4) 재계산하면 두 엔진이 다른 벡터를 보게 되고, 그 순간 "엔진 차이"가
아니라 "벡터 차이"를 재게 된다.

색인 대상은 PG의 `_searchable_condition()`을 통과한 청크만이다. 그래서 이 인덱스는
**스냅샷**이고, 업로드·삭제·검색토글 뒤에는 재색인해야 한다 — 그 동기화 부재가 운영 전환의
선결 조건이다(config.py의 search_backend 주석).

## 게이트 — 통과 전에는 어떤 품질 수치도 읽지 않는다

AGENTS.md 측정 규율("바뀐 결과물을 값으로 검증한 뒤 측정한다" — 조립이 틀린 채로 측정한
사고가 있었다). 게이트 3(kNN 동형성)은 어블레이션에서 재므로 여기선 1·2·4·5만 본다.

  1. 건수 대조 — OS 문서 수 = PG searchable 청크 수, 테넌트별로도 일치
  2. 벡터 무손실 — 표본을 되읽어 PG 원본과 element-wise 일치 (pgvector float4 → JSON 왕복)
  4. 토큰 값 대조 — `_analyze`가 `rag.lexical.bigrams()`와 동일 토큰을 내는지
  5. 테넌트 격리 — 테넌트 A 질의에 B 청크가 0건 (kNN·BM25 두 경로 각각)

실행:
  python -m eval._os_index --recreate     # 인덱스 재생성 후 전량 색인 (매핑 변경 시 필수)
  python -m eval._os_index                # 기존 인덱스에 upsert (_id=chunk_id라 중복 없음)

의존: DB + OpenSearch(docker-compose.opensearch.yml). TEI·LLM 불필요.
산출: eval/results/os_index_gates.md — 사람이 읽는 게이트 실측값.
"""
import argparse
import asyncio
import json
import random
from datetime import datetime
from pathlib import Path

from sqlalchemy import func, select

from config import settings
from database import AsyncSessionLocal
from rag import opensearch as B
from rag.lexical import bigrams
from rag.models import Chunk, Document, Faq, Folder
from rag.retriever import _searchable_condition

OUT = Path(__file__).resolve().parent / "results" / "os_index_gates.md"
INDEX = settings.opensearch_index
BULK = 200          # 청크가 수백~수천 규모라 넉넉하다. 실문서 확대 시 재조정.
SAMPLE_N = 5        # 게이트 2·4 표본 수


def _vec(dense) -> list[float]:
    """pgvector 컬럼 → JSON 직렬화 가능한 float 리스트.

    pgvector는 numpy float32 배열을 돌려주는데 json이 float32를 못 싼다. float32 값은
    float64로 정확히 표현되고(가수 비트가 부분집합) OpenSearch의 knn_vector도 4바이트
    float이라, 이 왕복은 무손실이어야 한다 — 게이트 2가 그것을 element-wise로 확인한다.
    """
    return [float(x) for x in dense]


async def _rows(session):
    stmt = (
        select(Chunk, Document.filename, Document.version)
        .outerjoin(Document, Chunk.document_id == Document.id)
        .outerjoin(Folder, Document.folder_id == Folder.id)
        .outerjoin(Faq, Chunk.faq_id == Faq.id)
        .where(_searchable_condition())
    )
    return (await session.execute(stmt)).all()


async def _bulk(os_client, docs: list[dict]) -> int:
    """전량 색인용 _bulk. 정의점(rag.opensearch.bulk_index)과 달리 refresh를 끄고 마지막에
    한 번만 refresh한다 — 배치는 건별 가시성이 필요 없고, wait_for를 배치마다 걸면 느리다.

    **_bulk은 원자적이지 않다** — 건별로 성공/실패한다. 부분 실패를 삼키면 색인이 조용히
    비므로 첫 오류에서 멈춘다(게이트 1이 뒤에서 다시 잡지만, 원인 메시지는 여기서만 나온다).
    """
    lines = []
    for d in docs:
        lines.append(json.dumps({"index": {"_index": INDEX, "_id": str(d["chunk_id"])}}))
        lines.append(json.dumps(d, ensure_ascii=False))
    resp = await os_client.bulk(body="\n".join(lines) + "\n", refresh=False)
    if resp.get("errors"):
        first = next(i["index"] for i in resp["items"] if i["index"].get("error"))
        raise SystemExit(f"bulk 실패: {first['error']}")
    return len(docs)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recreate", action="store_true",
                    help="인덱스를 삭제하고 다시 만든다(매핑 변경 시 필수 — 매핑은 사후 변경 불가)")
    ap.add_argument("--repair", action="store_true",
                    help="재동기화만 — PG↔OS id 집합을 대조해 누락분 색인·잉여분 삭제 "
                         "(정합 3층. 이중 쓰기가 놓친 것을 줍는다)")
    args = ap.parse_args()

    if args.repair:
        # 전량 색인 없이 차이만 메운다 — 크론으로 돌릴 수 있게 짧게 끝난다.
        os_client = B.client()
        async with AsyncSessionLocal() as session:
            r = await B.reconcile(session)
        print(f"재동기화: PG {r['pg']} · OS {r['os']} → "
              f"누락 색인 {r['indexed']}건 · 잉여 삭제 {r['deleted']}건")
        await os_client.close()
        return

    os_client = B.client()
    if args.recreate and await os_client.indices.exists(INDEX):
        await os_client.indices.delete(INDEX)
        print(f"인덱스 삭제: {INDEX}")
    if not await os_client.indices.exists(INDEX):
        await os_client.indices.create(INDEX, body=B.MAPPING)
        print(f"인덱스 생성: {INDEX}")
    await B.ensure_pipeline(os_client)
    print(f"검색 파이프라인 등록: {B.PIPELINE}")

    async with AsyncSessionLocal() as session:
        rows = await _rows(session)
        docs = [B.build_doc(c, fn, ver) for c, fn, ver in rows]
        n = 0
        for i in range(0, len(docs), BULK):
            n += await _bulk(os_client, docs[i:i + BULK])
            print(f"  색인 {n}/{len(docs)}", flush=True)
        await os_client.indices.refresh(INDEX)

        info = await os_client.info()
        report = [f"# OpenSearch 색인 게이트 (#139)\n",
                  f"- 측정: {datetime.now():%Y-%m-%d %H:%M} · 인덱스 `{INDEX}` "
                  f"· {settings.opensearch_url}",
                  f"- OpenSearch {info['version']['number']} "
                  f"(Lucene {info['version']['lucene_version']})\n"]

        # ── 게이트 1: 건수 대조 (테넌트별) ─────────────────────────────────
        pg_counts = dict((await session.execute(
            select(Chunk.tenant_id, func.count())
            .outerjoin(Document, Chunk.document_id == Document.id)
            .outerjoin(Folder, Document.folder_id == Folder.id)
            .outerjoin(Faq, Chunk.faq_id == Faq.id)
            .where(_searchable_condition())
            .group_by(Chunk.tenant_id)
        )).all())
        agg = await os_client.search(index=INDEX, body={
            "size": 0,
            "aggs": {"t": {"terms": {"field": "tenant_id", "size": 500}}},
        })
        os_counts = {b["key"]: b["doc_count"] for b in agg["aggregations"]["t"]["buckets"]}
        mism = {t: (pg_counts.get(t, 0), os_counts.get(t, 0))
                for t in set(pg_counts) | set(os_counts)
                if pg_counts.get(t, 0) != os_counts.get(t, 0)}
        g1 = not mism
        report.append(f"## 게이트 1 — 건수 대조: {'통과' if g1 else '실패'}\n")
        report.append(f"- PG searchable 합계 {sum(pg_counts.values())} / OS 문서 "
                      f"{sum(os_counts.values())} · 테넌트 {len(pg_counts)}개")
        if mism:
            report.append(f"- **불일치**: {mism}")
        else:
            report.append("- 테넌트별 전부 일치: "
                          + ", ".join(f"{t} {c}" for t, c in sorted(pg_counts.items())
                                      if c > 1))

        # ── 게이트 2: 벡터 무손실 ─────────────────────────────────────────
        ids = random.Random(139).sample([d["chunk_id"] for d in docs],
                                        min(SAMPLE_N, len(docs)))
        pg_vec = {c.id: _vec(c.dense) for c in (await session.execute(
            select(Chunk).where(Chunk.id.in_(ids)))).scalars()}
        bad_vec = []
        for cid in ids:
            got = (await os_client.get(index=INDEX, id=str(cid),
                                       _source=["dense"]))["_source"]["dense"]
            mine = pg_vec[cid]
            if len(got) != len(mine) or any(a != b for a, b in zip(got, mine)):
                worst = max(abs(a - b) for a, b in zip(got, mine)) if len(got) == len(mine) else None
                bad_vec.append((cid, len(mine), len(got), worst))
        g2 = not bad_vec
        report.append(f"\n## 게이트 2 — 벡터 무손실: {'통과' if g2 else '실패'}\n")
        report.append(f"- 표본 {len(ids)}건 · 차원 {len(pg_vec[ids[0]])} · element-wise 완전일치 "
                      f"{len(ids) - len(bad_vec)}/{len(ids)}")
        if bad_vec:
            report.append(f"- **불일치**: {bad_vec} (chunk_id, PG차원, OS차원, 최대오차)")

        # ── 게이트 4: 토큰 값 대조 + Nori 예시 ────────────────────────────
        samples = random.Random(140).sample(docs, min(SAMPLE_N, len(docs)))
        bad_tok = []
        for d in samples:
            mine = bigrams(d[B.NORI_FIELD])
            theirs = await B.analyze(os_client, "pretokenized", d[B.LEX_BIGRAM_FIELD])
            if mine != theirs:
                bad_tok.append(d["chunk_id"])
        g4 = not bad_tok
        report.append(f"\n## 게이트 4 — bigram 토큰 값 대조: {'통과' if g4 else '실패'}\n")
        report.append(f"- 표본 {len(samples)}청크 · `rag.lexical.bigrams()` 출력과 "
                      f"OpenSearch `pretokenized` analyzer 결과가 완전일치 "
                      f"{len(samples) - len(bad_tok)}/{len(samples)}")
        if bad_tok:
            report.append(f"- **불일치 chunk_id**: {bad_tok}")
        report.append("\nNori 토큰 예시 — 일치가 목적이 아니라 **사람이 읽고 판단할 재료**다"
                      "(고유명·복합어가 어떻게 갈리는지가 #138에서 볼 것의 예고편):\n")
        for probe in ("아더에러", "아더에러 반품 기간", "브랜드 아더에러 교환 규정",
                      "배송지연 대응", "홈플러스 온라인몰", "리필스테이션 이용기준"):
            report.append(f"- `{probe}` → `{await B.analyze(os_client, 'nori_ko', probe)}`")

        # ── 게이트 5: 테넌트 격리 (kNN·BM25 두 경로) ──────────────────────
        tenants = [t for t, c in sorted(pg_counts.items(), key=lambda x: -x[1]) if c > 10][:2]
        leaks = []
        if len(tenants) >= 2:
            a, b = tenants[0], tenants[1]
            probe_doc = next(d for d in docs if d["tenant_id"] == b)
            vec = probe_doc["dense"]                      # B의 청크 벡터로 A를 검색 —
            q_text = probe_doc[B.NORI_FIELD][:200]        # 격리가 없으면 B가 상위에 뜬다
            own = {d["chunk_id"] for d in docs if d["tenant_id"] == a}
            for label, got in (
                ("kNN", await B.knn_ids(os_client, a, vec, 30)),
                ("BM25(nori)", await B.bm25_ids(os_client, a, B.NORI_FIELD, q_text, 30)),
                ("BM25(bigram)", await B.bm25_ids(os_client, a, B.LEX_BIGRAM_FIELD,
                                                  " ".join(bigrams(q_text)), 30)),
            ):
                foreign = [i for i in got if i not in own]
                if foreign:
                    leaks.append((label, len(foreign), foreign[:5]))
        g5 = not leaks and len(tenants) >= 2
        report.append(f"\n## 게이트 5 — 테넌트 격리: "
                      f"{'통과' if g5 else ('실패' if leaks else '검증 불가(테넌트 부족)')}\n")
        report.append(f"- 테넌트 `{tenants[0] if tenants else '?'}` 검색에 다른 테넌트 청크가 "
                      f"섞이는지 — `{tenants[1] if len(tenants) > 1 else '?'}` 청크의 벡터·본문을 질의로 사용")
        report.append("- PG는 RLS 없이 WHERE절이 유일한 방어선이고(rag/models.py:14-26), "
                      "OpenSearch에서는 그 보장을 `bool.filter`로 처음부터 다시 세운다")
        if leaks:
            report.append(f"- **유출**: {leaks}")
        elif g5:
            report.append("- kNN·BM25(nori)·BM25(bigram) 세 경로 모두 유출 0건")

        gates = {"1 건수": g1, "2 벡터": g2, "4 토큰": g4, "5 격리": g5}
        report.insert(3, "**종합: " + " · ".join(
            f"{k} {'✅' if v else '❌'}" for k, v in gates.items())
            + "** (게이트 3 kNN 동형성은 `_hybrid_ablation --os`에서 측정)\n")

        OUT.parent.mkdir(exist_ok=True)
        OUT.write_text("\n".join(report) + "\n")
        print("\n" + "\n".join(report[1:]))
        print(f"\n리포트: {OUT}")
        await os_client.close()
        if not all(gates.values()):
            raise SystemExit("게이트 실패 — 품질 측정으로 넘어가지 마라 (배선을 먼저 고칠 것)")


if __name__ == "__main__":
    asyncio.run(main())
