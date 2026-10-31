"""#235 측정 — 조건(임시 인덱스)마다 ⓐ dense 풀의 근사 vs 정확 ⓑ 최종 검색 지표 ⓒ 회귀 문항 덤프.

ⓐ 같은 질의 벡터로 운영 kNN 절(rag/search/os_search.knn_clause, 상위 30)과 정확 탐색(script_score + knn_score, 같은 필터)을 돌려
   상위 30 겹침·상위 1 일치·정답 조각의 양쪽 순위·지연을 적는다. 정확 점수는 1+cos, 근사는 (1+cos)/2 — 순위만 비교한다.
   실측(운영 인덱스, 10/23): 현 규모(테넌트 444조각)에서 30/30 일치 — 노이즈가 커지면 여기가 먼저 갈린다.
ⓑ eval/retrieval_v2.compute(tenants={측정 테넌트})를 임시 인덱스를 보게 한 채 돌린다(settings.opensearch_index만 바꿔 끼움 —
   #234 condition()의 단순판. 리랭커 입력은 조각 필드 그대로라 바꿀 게 없다).
ⓒ x0 대비 Hit@1이나 R@5가 떨어진 문항은 상위 5 조각 머리를 노이즈/실문서 표시와 함께 덤프 → 사람이 유형을 분류한다.
산출: OUT/measure.json(원자료) · report.md(요약표) · regressions.md(덤프).
"""
import json
import time
from contextlib import contextmanager
from datetime import datetime

from config import settings
from database import AsyncSessionLocal
from eval.axis_common import chunk_head, commit
from eval.generation import load_gold, row_tenant
from eval.retrieval import resolve_gold
from eval.retrieval_v2 import TYPES, compute
from eval.scale_noise import CONDITIONS, NOISE_DOC_ID_BASE, OUT, TENANT, index_name
from rag.search.embeddings import embed_texts
from rag.search.os_client import client
from rag.search.os_search import knn_clause, tenant_filter
from rag.search.retriever import retrieve_candidates

POOL = 30   # 운영 candidates_per_branch(rag/search/retriever.retrieve_candidates)


@contextmanager
def condition(cond: str):
    orig = settings.opensearch_index
    settings.opensearch_index = index_name(cond)
    try:
        yield
    finally:
        settings.opensearch_index = orig


def _exact(tenant: str, vector: list[float], n: int) -> dict:
    return {'size': n, '_source': ['chunk_id'],
            'query': {'script_score': {'query': tenant_filter(tenant),
                                       'script': {'source': 'knn_score', 'lang': 'knn',
                                                  'params': {'field': 'dense', 'query_value': vector, 'space_type': 'cosinesimil'}}}}}


async def _pool_compare(gold_rows: list[dict], gold_ids: dict[str, set[int]]) -> list[dict]:
    """ⓐ — 문항마다 근사·정확 상위 30."""
    out = []
    vecs = await embed_texts([g['query'] for g in gold_rows], kind="query")
    for g, v in zip(gold_rows, vecs):
        ids = gold_ids.get(g['id']) or set()
        if not ids:
            continue
        t = time.time()
        a = await client().search(index=settings.opensearch_index,
                                  body={'size': POOL, '_source': ['chunk_id'], 'query': knn_clause(TENANT, v.dense, POOL)})
        ta = time.time() - t
        t = time.time()
        e = await client().search(index=settings.opensearch_index, body=_exact(TENANT, v.dense, POOL))
        te = time.time() - t
        aa = [int(h['_source']['chunk_id']) for h in a['hits']['hits']]
        ee = [int(h['_source']['chunk_id']) for h in e['hits']['hits']]
        rank = lambda lst: next((i + 1 for i, c in enumerate(lst) if c in ids), None)   # noqa: E731
        out.append({'id': g['id'], 'overlap': len(set(aa) & set(ee)), 'top1_same': bool(aa and ee and aa[0] == ee[0]),
                    'gold_rank_approx': rank(aa), 'gold_rank_exact': rank(ee),
                    'noise_in_pool': sum(1 for c in aa if (c >> 20) >= NOISE_DOC_ID_BASE),
                    'ms_approx': round(ta * 1000), 'ms_exact': round(te * 1000)})
    return out


def _summary(rows: list[dict]) -> dict:
    n = len(rows) or 1
    return {'n': len(rows), **{k: round(sum(r['scores'][k] for r in rows) / n, 3) for k in ('hit_at_1', 'recall_at_5', 'recall_at_20', 'mrr')}}


def _pool_summary(rows: list[dict]) -> dict:
    n = len(rows) or 1
    return {'n': len(rows), 'overlap30': round(sum(r['overlap'] for r in rows) / n, 1),
            'top1_same': round(sum(r['top1_same'] for r in rows) / n, 3),
            'gold_in_approx30': round(sum(r['gold_rank_approx'] is not None for r in rows) / n, 3),
            'gold_in_exact30': round(sum(r['gold_rank_exact'] is not None for r in rows) / n, 3),
            'noise_in_pool': round(sum(r['noise_in_pool'] for r in rows) / n, 1),
            'ms_approx': round(sum(r['ms_approx'] for r in rows) / n), 'ms_exact': round(sum(r['ms_exact'] for r in rows) / n)}


async def _regressions(cond: str, rows: list[dict], base: dict[str, dict], gold: dict[str, dict], gold_ids: dict[str, set[int]]) -> list[str]:
    lines = []
    async with AsyncSessionLocal() as session:
        for r in rows:
            b = base.get(r['id'])
            if not b or (r['scores']['hit_at_1'] >= b['scores']['hit_at_1'] and r['scores']['recall_at_5'] >= b['scores']['recall_at_5']):
                continue
            cands = await retrieve_candidates(session, TENANT, gold[r['id']]['query'], top_n=5)
            lines.append(f"### {cond} · {r['id']} ({r['type']}) — Hit@1 {b['scores']['hit_at_1']}→{r['scores']['hit_at_1']} "
                         f"R@5 {b['scores']['recall_at_5']}→{r['scores']['recall_at_5']}\n질문: {gold[r['id']]['query']}\n")
            for i, c in enumerate(cands.chunks, 1):
                tag = '노이즈' if c.document_id and c.document_id >= NOISE_DOC_ID_BASE else ('**정답**' if c.chunk_id in gold_ids.get(r['id'], set()) else '실문서')
                lines.append(f"{i}. [{tag}] {c.filename} › {' > '.join(c.heading_path or [])} — {chunk_head(c.text, 100)}")
            lines.append('')
    return lines


def _signature(cond: str) -> str:
    """빌드 서명(indices.json의 built 시각·조각 수) — 재빌드 뒤 낡은 측정 결과를 다시 쓰지 않게(리뷰 지적 1)."""
    info = json.loads((OUT / 'indices.json').read_text(encoding='utf-8')).get(cond)
    if not info:
        raise SystemExit(f'{cond}: build 먼저')
    return f"{info['built']}|{info['total']}|{info['noise_chunks']}"


async def measure(conds: list[str]) -> None:
    bad = [c for c in conds if c not in CONDITIONS]
    if bad:
        raise SystemExit(f'모르는 조건 {bad} — 가능: {CONDITIONS}')
    gold_all = load_gold()
    gold_rows = [g for g in gold_all if row_tenant(g) == TENANT and g['type'] in TYPES]
    gold = {g['id']: g for g in gold_rows}
    path = OUT / 'measure.json'
    result = (json.loads(path.read_text(encoding='utf-8')) if path.exists() else
              {'commit': commit(), 'base_index': settings.opensearch_index, 'started': datetime.now().isoformat(timespec='seconds'),
               'signature': {}, 'pool': {}, 'retrieval': {}})
    result.setdefault('signature', {})      # 서명 도입 전 파일(10/23 x0 1차)도 읽는다 — 서명이 없으면 전부 재측정
    OUT.mkdir(parents=True, exist_ok=True)
    gold_ids_by_cond: dict[str, dict[str, set[int]]] = {}
    for cond in conds:
        sig = _signature(cond)
        if result['signature'].get(cond) != sig:          # 빌드가 바뀌었으면 그 조건은 처음부터
            result['pool'].pop(cond, None)
            result['retrieval'].pop(cond, None)
            result['signature'][cond] = sig
        with condition(cond):
            async with AsyncSessionLocal() as session:
                resolved = await resolve_gold(session, TENANT, gold_rows)
            gold_ids = {k: set(v) for k, v in resolved.scoring_ids.items()}
            gold_ids_by_cond[cond] = gold_ids
            if cond not in result['pool']:
                result['pool'][cond] = await _pool_compare(gold_rows, gold_ids)
                path.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding='utf-8')
            if cond not in result['retrieval']:
                result['retrieval'][cond] = (await compute(tenants={TENANT}))['rows']
                path.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding='utf-8')
            print(cond, _pool_summary(result['pool'][cond]), _summary(result['retrieval'][cond]), flush=True)
    _report(result)
    # 회귀 덤프는 저장된 **모든** 조건으로 다시 만든다 — 일부 조건만 다시 재도 앞선 조건의 덤프가 사라지지 않게(리뷰 지적 2).
    # 상위 5는 채점과 별도로 다시 검색한 것이라 TEI 흔들림(1.4e-4)만큼 순위가 다를 수 있다.
    reg_lines = []
    if 'x0' in result['retrieval']:
        base = {r['id']: r for r in result['retrieval']['x0']}
        for cond in CONDITIONS:
            if cond == 'x0' or cond not in result['retrieval']:
                continue
            with condition(cond):
                if cond not in gold_ids_by_cond:
                    async with AsyncSessionLocal() as session:
                        resolved = await resolve_gold(session, TENANT, gold_rows)
                    gold_ids_by_cond[cond] = {k: set(v) for k, v in resolved.scoring_ids.items()}
                reg_lines += await _regressions(cond, result['retrieval'][cond], base, gold, gold_ids_by_cond[cond])
    (OUT / 'regressions.md').write_text('# #235 회귀 문항 덤프(x0 대비)\n\n' + '\n'.join(reg_lines) + '\n', encoding='utf-8')
    print('→', path)


def _report(result: dict) -> None:
    info = json.loads((OUT / 'indices.json').read_text(encoding='utf-8')) if (OUT / 'indices.json').exists() else {}
    lines = [f"# #235 규모 곡선 — 커밋 {result['commit']}, 기준 인덱스 {result['base_index']}, {result['started']}", '',
             '| 조건 | 테넌트 조각 | 전체 조각 | 근사∩정확@30 | 상위1 일치 | 정답∈근사30 | 정답∈정확30 | 풀 내 노이즈 | kNN ms 근사/정확 |',
             '|---|---|---|---|---|---|---|---|---|']
    for cond in CONDITIONS:
        if cond in result['pool']:
            p, i = _pool_summary(result['pool'][cond]), info.get(cond, {})
            lines.append(f"| {cond} | {i.get('tenant_chunks', '?')} | {i.get('total', '?')} | {p['overlap30']}/30 | {p['top1_same']} | "
                         f"{p['gold_in_approx30']} | {p['gold_in_exact30']} | {p['noise_in_pool']} | {p['ms_approx']}/{p['ms_exact']} |")
    lines += ['', '| 조건 | n | Hit@1 | R@5 | R@20 | MRR |', '|---|---|---|---|---|---|']
    for cond in CONDITIONS:
        if cond in result['retrieval']:
            s = _summary(result['retrieval'][cond])
            lines.append(f"| {cond} | {s['n']} | {s['hit_at_1']} | {s['recall_at_5']} | {s['recall_at_20']} | {s['mrr']} |")
    (OUT / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))
