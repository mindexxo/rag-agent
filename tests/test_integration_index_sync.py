"""색인 동기화 계약 (#139) — E2E 수동 케이스 중 코드로 덮을 수 있는 것을 테스트로 옮긴 것.

라우터는 대기열 행만 남기고 색인은 워커가 한다. 여기서는 워커 대신 `ingest`/`sync_faq`(그 문서·FAQ의
행만 drain)로 같은 경로를 태운다. 지연 창("drain 전엔 아직 보인다")은 제품 결정이므로 그대로 단언한다.

  1. 인용 메타(파일명·버전·페이지·폴더명·폴더 설명)는 엔진 _source에서 온다      (E2E #1)
  2. 문서 검색토글: META 부분 갱신 — 제외/복귀, 재색인 없음(indexed_at 불변)        (E2E #5)
  3. 색인 일시 실패: attempts·last_error 남기고 다음 회차에 성공                     (E2E #7)
  4. reconcile(문서): 색인에서 사라진 ready 문서는 재등재, PG에 없는 문서 청크는 삭제  (E2E #9)
  5. reconcile(FAQ): 같은 계약                                                        (배포 체크리스트 D)
"""
import json

import pytest
from sqlalchemy import select

from database import AsyncSessionLocal
from rag import os_index, os_reconcile, outbox
from rag.models import Document, SearchIndexOutbox
from rag.retriever import retrieve_candidates
from tests.conftest import faq_doc, indexed_chunk_texts, ingest, sync_faq

MD = '# 환불 정책\n\n## 1. 기간\n\n단순변심 반품은 14일 이내 신청한다.\n'.encode()


async def _upload(client, filename, content=MD):
    res = await client.post('/kms/documents', files={'file': (filename, content, 'text/markdown')})
    assert res.status_code == 200, res.text
    return res.json()


async def _doc(doc_id):
    async with AsyncSessionLocal() as s:
        return await s.get(Document, doc_id)


async def _rows(doc_id):
    async with AsyncSessionLocal() as s:
        rows = (await s.execute(select(SearchIndexOutbox).order_by(SearchIndexOutbox.id))).scalars().all()
    return [(r.op, r.status, r.attempts, r.last_error) for r in rows
            if r.payload.get('document_id') == doc_id or doc_id in (r.payload.get('document_ids') or [])]


async def _cited_ids(tenant_id, query='반품 기간') -> set[int]:
    async with AsyncSessionLocal() as s:
        cands = await retrieve_candidates(s, tenant_id, query, top_n=20)
    return {c.document_id for c in cands.chunks}


@pytest.mark.asyncio
async def test_인용_메타는_엔진_source에서_온다(client, tenant_id, fake_queue, blob_tmp, fake_embed):
    """PG를 되묻지 않으므로 RetrievedChunk의 모든 필드가 색인 문서에서 채워져야 한다 — 하나라도
    비면 인용 표시·리랭커 입력(폴더 설명)이 조용히 빈다."""
    folder = (await client.post('/kms/folders', json={'name': '정책', 'description': '환불·반품 규정 모음'})).json()
    body = await _upload(client, '환불정책.md')
    assert (await client.patch(f"/kms/documents/{body['document_id']}",
                               json={'folder_id': folder['id']})).status_code == 200
    await ingest(body['document_id'])           # 폴더 이동 META 행도 함께 처리된다

    async with AsyncSessionLocal() as s:
        cands = await retrieve_candidates(s, tenant_id, '반품 기간', top_n=20)
    mine = [c for c in cands.chunks if c.document_id == body['document_id']]
    assert mine, '방금 색인한 문서가 후보에 없다'
    c = mine[0]
    assert c.filename == '환불정책.md' and c.version == 1
    assert c.folder_name == '정책' and c.folder_description == '환불·반품 규정 모음'
    assert c.heading_path and c.page is None and c.text     # md: 헤딩 있음·페이지 없음


@pytest.mark.asyncio
async def test_문서_검색토글은_META_부분갱신으로_제외_복귀한다(client, tenant_id, fake_queue, blob_tmp, fake_embed):
    body = await _upload(client, '환불정책.md')
    await ingest(body['document_id'])
    doc_id = body['document_id']
    indexed_at = (await _doc(doc_id)).indexed_at
    assert doc_id in await _cited_ids(tenant_id)

    assert (await client.patch(f'/kms/documents/{doc_id}', json={'is_searchable': False})).status_code == 200
    assert doc_id in await _cited_ids(tenant_id)            # drain 전 — 가이드대로 아직 보인다
    r = await ingest(doc_id)                                # META_DOCUMENTS 행 하나
    assert r == {'done': 1, 'failed': 0}
    assert doc_id not in await _cited_ids(tenant_id)        # 제외
    assert (await _doc(doc_id)).indexed_at == indexed_at    # 재색인 없음 — 플래그만 바뀜
    assert [op for op, *_ in await _rows(doc_id)] == ['index_document', 'meta_documents']

    assert (await client.patch(f'/kms/documents/{doc_id}', json={'is_searchable': True})).status_code == 200
    await ingest(doc_id)
    assert doc_id in await _cited_ids(tenant_id)            # 복귀
    assert (await _doc(doc_id)).indexed_at == indexed_at


@pytest.mark.asyncio
async def test_색인_일시_실패는_attempts를_남기고_다음_회차에_성공한다(client, tenant_id, fake_queue, blob_tmp,
                                                                fake_embed, monkeypatch):
    """OpenSearch가 잠깐 죽은 상황 — 행은 pending으로 남고 attempts·last_error가 쌓이며, 문서는
    failed가 아니다. 복구되면 다음 회차가 같은 행을 끝낸다(at-least-once)."""
    body = await _upload(client, '환불정책.md')
    doc_id = body['document_id']

    import rag.documents as rd
    real = rd.os_index.index_parsed_document
    calls = {'n': 0}

    async def _flaky(**kw):
        calls['n'] += 1
        if calls['n'] == 1:
            raise ConnectionError('Cannot connect to host localhost:9200')
        return await real(**kw)

    monkeypatch.setattr(rd.os_index, 'index_parsed_document', _flaky)
    assert await ingest(doc_id) == {'done': 0, 'failed': 1}
    doc = await _doc(doc_id)
    assert doc.status == 'pending'                          # failed 아님 — MAX_ATTEMPTS(5) 전
    (op, status, attempts, err), = await _rows(doc_id)
    assert (op, status, attempts) == ('index_document', 'pending', 1)
    assert 'Cannot connect' in err

    assert await ingest(doc_id) == {'done': 1, 'failed': 0}  # "복구 후" 회차
    doc = await _doc(doc_id)
    assert doc.status == 'ready' and doc.is_active is True
    (op, status, attempts, _), = await _rows(doc_id)
    assert (status, attempts) == ('done', 1)                 # 시도 이력은 남는다
    assert await indexed_chunk_texts(doc_id)


@pytest.mark.asyncio
async def test_reconcile은_사라진_ready_문서를_재등재하고_유령_청크를_지운다(client, tenant_id, fake_queue,
                                                                     blob_tmp, fake_embed):
    body = await _upload(client, '환불정책.md')
    doc_id = body['document_id']
    await ingest(doc_id)
    # 누락 재현: 색인에서만 지운다(스냅샷 복구 뒤 유실 등). PG는 ready 그대로.
    await os_index._delete_by_terms('document_id', [doc_id])
    assert await indexed_chunk_texts(doc_id) == []
    # 잉여 재현: PG에 없는 document_id의 청크를 이 테넌트로 넣는다.
    ghost = os_index.build_doc(
        os_index._ParsedChunk(cid=os_index.chunk_os_id(document_id=999_999_001, chunk_index=0),
                                tenant_id=tenant_id, document_id=999_999_001, faq_id=None,
                                text='유령', heading_path=[], page=None, meta={}, dense=[0.01] * 1024),
        '유령.md', 1, searchable=True)
    await os_index.bulk_index([ghost])

    async with AsyncSessionLocal() as s:
        r = await os_reconcile.reconcile(s, tenant_id)        # 테넌트 스코프 — 남의 문서는 건드리지 않는다
    assert r['indexed'] == 1 and r['deleted'] == 1 and r['unit'] == 'document'
    assert (await _doc(doc_id)).status == 'pending'         # 재등재 = pending 되돌림 + INDEX 행
    assert await indexed_chunk_texts(999_999_001) == []

    assert await ingest(doc_id) == {'done': 1, 'failed': 0}  # 워커가 다시 색인
    assert (await _doc(doc_id)).status == 'ready'
    assert await indexed_chunk_texts(doc_id)
    async with AsyncSessionLocal() as s:
        assert (await os_reconcile.reconcile(s, tenant_id))['indexed'] == 0


@pytest.mark.asyncio
async def test_reconcile_faqs는_사라진_FAQ를_재등재하고_유령을_지운다(client, tenant_id, fake_embed):
    res = await client.post('/kms/faqs', json={'question': '환불 기간은?', 'variants': [], 'answer': '7일'})
    faq_id = res.json()['id']
    await sync_faq(faq_id)
    assert await faq_doc(faq_id)
    await os_index._delete_by_terms('faq_id', [faq_id])
    ghost = os_index.build_doc(
        os_index._ParsedChunk(cid=os_index.chunk_os_id(faq_id=999_999_002), tenant_id=tenant_id,
                                document_id=None, faq_id=999_999_002, text='Q: 유령 A: 유령',
                                heading_path=['유령'], page=None, meta={}, dense=[0.01] * 1024),
        None, None, searchable=True)
    await os_index.bulk_index([ghost])

    async with AsyncSessionLocal() as s:
        r = await os_reconcile.reconcile_faqs(s, tenant_id)
    assert r['indexed'] == 1 and r['deleted'] == 1 and r['unit'] == 'faq'
    assert await faq_doc(999_999_002) is None
    assert await sync_faq(faq_id) == {'done': 1, 'failed': 0}
    assert await faq_doc(faq_id)


@pytest.mark.asyncio
async def test_업로드_시_지정한_폴더가_색인_메타까지_간다(client, tenant_id, fake_queue, blob_tmp, fake_embed):
    """#165: 업로드 시점 folder_id는 INDEX_DOCUMENT 경로 하나로 색인까지 간다 — PATCH가 필요 없다.

    위 1번(인용 메타) 테스트는 업로드 후 PATCH로 옮기는 경로를 본다. 여기서 보는 건 그 전 단계,
    즉 처음부터 폴더를 달고 들어온 문서가 META 부분갱신 없이도 폴더 메타를 갖느냐다.
    """
    folder = (await client.post('/kms/folders',
                                json={'name': '정책', 'description': '환불·반품 규정 모음'})).json()
    res = await client.post('/kms/documents', files={
        'file': ('환불정책.md', MD, 'text/markdown'),
        'document-data': ('blob', json.dumps({'folder_id': folder['id']}), 'application/json'),
    })
    assert res.status_code == 200, res.text
    doc_id = res.json()['document_id']
    await ingest(doc_id)

    async with AsyncSessionLocal() as s:
        cands = await retrieve_candidates(s, tenant_id, '반품 기간', top_n=20)
    mine = [c for c in cands.chunks if c.document_id == doc_id]
    assert mine, '방금 색인한 문서가 후보에 없다'
    assert mine[0].folder_name == '정책'
    assert mine[0].folder_description == '환불·반품 규정 모음'
