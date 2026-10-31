"""#235 임시 인덱스 — 운영 인덱스 복사 + 노이즈 조각(규모별) → forcemerge.

복사는 #234와 같이 search(match_all)+bulk로 `_id`·`_source`(dense 포함)를 그대로 옮긴다(재임베딩 없음). 노이즈 조각은 운영 색인
문서 조립 함수(rag/search/os_index.build_doc)에 ParsedChunk를 넘겨 만든다 — 필드 모양(헤딩·식별자·Nori·bigram)이 운영과 같다.
document_id는 NOISE_DOC_ID_BASE부터 문서마다 하나(PG 행 없음 — 검색 평가 경로는 노이즈에 PG를 안 쓴다, eval/retrieval.resolve_gold는
정답 문서만 PG로 푼다). x0는 노이즈 0(복사+forcemerge만) — 세그먼트 구조를 맞춘 기준선. idf는 x30과 같은 노이즈를 NOISE_TENANT로.
규모 n은 "측정 테넌트 조각 수 × 배수"를 문서 단위로 채운다(같은 시드로 섞어 x5 ⊂ x10 ⊂ x30).
산출: OUT/indices.json(조건별 조각 수·세그먼트).
"""
import json
import random
from collections import defaultdict
from datetime import datetime

from opensearchpy.helpers import async_bulk

from config import settings
from rag.search.os_client import MAPPING, client
from rag.search.os_index import ParsedChunk, build_doc, chunk_os_id
from eval.scale_noise import CHUNKS, CONDITIONS, FILTER, NOISE_DOC_ID_BASE, NOISE_TENANT, OUT, SCALES, TENANT, index_name


async def _all_docs(index: str) -> list[dict]:
    resp = await client().search(index=index, body={'size': 10000, 'track_total_hits': True, 'query': {'match_all': {}}})
    if resp['hits']['total']['value'] > 10000:
        raise RuntimeError('조각 10,000개 초과 — scroll 필요')
    return [{'_id': h['_id'], **h['_source']} for h in resp['hits']['hits']]


def _noise_docs(limit: int, tenant: str) -> tuple[list[dict], int]:
    """노이즈 조각을 문서 단위로 limit개 이상 채운다 → (OS 문서 목록, 문서 수). filter를 다시 돌려 kept가 줄어도 chunks.jsonl엔
    옛 문서가 남으므로 **kept 교집합**만 쓴다(리뷰 지적 6)."""
    kept = set(json.loads(FILTER.read_text(encoding='utf-8'))['kept'])
    by_doc = defaultdict(list)
    for l in CHUNKS.read_text(encoding='utf-8').splitlines():
        if l.strip():
            c = json.loads(l)
            if c['doc'] in kept:
                by_doc[c['doc']].append(c)
    order = sorted(by_doc)
    random.Random(235).shuffle(order)
    out, n_docs = [], 0
    for k, doc in enumerate(order):
        if len(out) >= limit:
            break
        document_id = NOISE_DOC_ID_BASE + k
        for c in by_doc[doc]:
            pc = ParsedChunk(id=chunk_os_id(document_id=document_id, chunk_index=c['chunk_index']), tenant_id=tenant,
                              document_id=document_id, faq_id=None, text=c['text'], heading_path=c['heading_path'],
                              page=c['page'], meta=c['meta'], dense=c['dense'])
            out.append({'_id': str(pc.id), **build_doc(pc, c['filename'], 1)})
        n_docs += 1
    if len(out) < limit:
        raise SystemExit(f'노이즈 조각 부족: {len(out)} < {limit} — generate/chunks를 더 돌릴 것')
    return out, n_docs


async def build(conds: list[str]) -> None:
    bad = [c for c in conds if c not in CONDITIONS]
    if bad:
        raise SystemExit(f'모르는 조건 {bad} — 가능: {CONDITIONS}')
    base = await _all_docs(settings.opensearch_index)
    # 배수의 기준 = 측정 테넌트의 **검색 가능한** 문서 조각(kNN 풀이 searchable 필터를 걸므로 같은 모집단, 리뷰 지적 10)
    n_tenant = sum(1 for d in base if d.get('tenant_id') == TENANT and d.get('document_id') is not None and d.get('searchable'))
    info = json.loads((OUT / 'indices.json').read_text(encoding='utf-8')) if (OUT / 'indices.json').exists() else {}
    print(f'운영 복사 {len(base)}건 · {TENANT} 검색 가능 문서 조각 {n_tenant}', flush=True)
    for cond in conds:
        name = index_name(cond)
        scale = SCALES['x30'] if cond == 'idf' else SCALES[cond]
        noise, n_docs = _noise_docs(n_tenant * scale, NOISE_TENANT if cond == 'idf' else TENANT) if scale else ([], 0)
        if await client().indices.exists(name):
            await client().indices.delete(name)
        await client().indices.create(name, body=MAPPING)
        actions = [{'_index': name, '_id': d['_id'], '_source': {k: v for k, v in d.items() if k != '_id'}} for d in base]
        actions += [{'_index': name, '_id': d['_id'], '_source': {k: v for k, v in d.items() if k != '_id'}} for d in noise]
        await async_bulk(client(), actions, chunk_size=500, refresh=True)
        await client().indices.forcemerge(index=name, max_num_segments=1)   # 세그먼트 수가 HNSW 탐색 조건을 바꾼다 — 조건 간 통일
        await client().indices.refresh(index=name)
        total = (await client().count(index=name))['count']
        seg = await client().cat.segments(index=name, format='json')
        info[cond] = {'index': name, 'base': len(base), 'noise_chunks': len(noise), 'noise_docs': n_docs, 'total': total,
                      'tenant_chunks': n_tenant + (len(noise) if cond != 'idf' else 0), 'segments': len(seg),
                      'built': datetime.now().isoformat(timespec='seconds')}   # measure가 이 서명으로 낡은 결과를 가린다
        print(f'  {name}: 전체 {total} · 노이즈 {len(noise)}조각/{n_docs}문서 · 세그먼트 {len(seg)}', flush=True)
        (OUT / 'indices.json').write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding='utf-8')


async def drop() -> None:
    for cond in CONDITIONS:
        name = index_name(cond)
        if await client().indices.exists(name):
            await client().indices.delete(name)
            print('삭제', name)
