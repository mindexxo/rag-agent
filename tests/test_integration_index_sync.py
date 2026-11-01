"""색인 동기화 계약(검색 메타) — 라우터는 대기열 행만 남기고 색인은 워커가 한다.

워커 대신 `ingest`(그 문서의 행만 drain)로 같은 경로를 태운다.
지연 창("drain 전엔 아직 보인다")은 제품 결정이므로 그대로 단언한다.

  1. 인용 메타(파일명·버전·페이지·폴더명·폴더 설명)는 엔진 _source에서 온다
  2. 문서 검색토글: META 부분 갱신 — 제외/복귀, 재색인 없음(indexed_at 불변)
  3. 업로드 시 지정한 폴더·검색제외 폴더·일괄 참조끄기도 같은 메타로 반영된다
"""

import json

import pytest

from database import AsyncSessionLocal
from rag.search.retriever import retrieve_candidates
from tests.conftest import ingest
from tests.helpers_documents import MD, get_doc, upload
from tests.helpers_index_sync import cited_ids, outbox_rows


@pytest.mark.asyncio
async def test_인용_메타는_엔진_source에서_온다(client, tenant_id, fake_queue, blob_tmp, fake_embed):
    """PG를 되묻지 않으므로 RetrievedChunk의 모든 필드가 색인 문서에서 채워져야 한다 — 하나라도
    비면 인용 표시·리랭커 입력(폴더 설명)이 조용히 빈다."""
    folder = (await client.post('/kms/folders', json={'name': '정책', 'description': '환불·반품 규정 모음'})).json()
    body = await upload(client, '환불정책.md', MD)
    assert (
        await client.patch(f"/kms/documents/{body['document_id']}", json={'folder_id': folder['id']})
    ).status_code == 200
    await ingest(body['document_id'])  # 폴더 이동 META 행도 함께 처리된다

    async with AsyncSessionLocal() as s:
        cands = await retrieve_candidates(s, tenant_id, '반품 기간', top_n=20)
    mine = [c for c in cands.chunks if c.document_id == body['document_id']]
    assert mine, '방금 색인한 문서가 후보에 없다'
    c = mine[0]
    assert c.filename == '환불정책.md' and c.version == 1
    assert c.folder_name == '정책' and c.folder_description == '환불·반품 규정 모음'
    assert c.heading_path and c.page is None and c.text  # md: 헤딩 있음·페이지 없음


@pytest.mark.asyncio
async def test_문서_검색토글은_META_부분갱신으로_제외_복귀한다(client, tenant_id, fake_queue, blob_tmp, fake_embed):
    body = await upload(client, '환불정책.md', MD)
    await ingest(body['document_id'])
    doc_id = body['document_id']
    indexed_at = (await get_doc(doc_id)).indexed_at
    assert doc_id in await cited_ids(tenant_id)

    assert (await client.patch(f'/kms/documents/{doc_id}', json={'is_searchable': False})).status_code == 200
    assert doc_id in await cited_ids(tenant_id)  # drain 전 — 가이드대로 아직 보인다
    r = await ingest(doc_id)  # META_DOCUMENTS 행 하나
    assert r == {'done': 1, 'failed': 0}
    assert doc_id not in await cited_ids(tenant_id)  # 제외
    assert (await get_doc(doc_id)).indexed_at == indexed_at  # 재색인 없음 — 플래그만 바뀜
    assert [op for op, *_ in await outbox_rows(doc_id)] == ['index_document', 'meta_documents']

    assert (await client.patch(f'/kms/documents/{doc_id}', json={'is_searchable': True})).status_code == 200
    await ingest(doc_id)
    assert doc_id in await cited_ids(tenant_id)  # 복귀
    assert (await get_doc(doc_id)).indexed_at == indexed_at


@pytest.mark.asyncio
async def test_업로드_시_지정한_폴더가_색인_메타까지_간다(client, tenant_id, fake_queue, blob_tmp, fake_embed):
    """업로드 시점 folder_id는 INDEX_DOCUMENT 경로 하나로 색인까지 간다 — PATCH가 필요 없다.

    위 1번(인용 메타) 테스트는 업로드 후 PATCH로 옮기는 경로를 본다. 여기서 보는 건 그 전 단계,
    즉 처음부터 폴더를 달고 들어온 문서가 META 부분갱신 없이도 폴더 메타를 갖느냐다.
    """
    folder = (await client.post('/kms/folders', json={'name': '정책', 'description': '환불·반품 규정 모음'})).json()
    res = await client.post(
        '/kms/documents',
        files={
            'file': ('환불정책.md', MD, 'text/markdown'),
            'document-data': ('blob', json.dumps({'folder_id': folder['id']}), 'application/json'),
        },
    )
    assert res.status_code == 200, res.text
    doc_id = res.json()['document_id']
    await ingest(doc_id)

    async with AsyncSessionLocal() as s:
        cands = await retrieve_candidates(s, tenant_id, '반품 기간', top_n=20)
    mine = [c for c in cands.chunks if c.document_id == doc_id]
    assert mine, '방금 색인한 문서가 후보에 없다'
    assert mine[0].folder_name == '정책'
    assert mine[0].folder_description == '환불·반품 규정 모음'


@pytest.mark.asyncio
async def test_검색제외_폴더로_업로드하면_검색에_안_잡힌다(client, tenant_id, fake_queue, blob_tmp, fake_embed):
    """폴더의 참조 off가 **신규 업로드에도** 먹는지.

    이미 색인된 문서는 폴더 토글이 META 부분갱신으로 반영된다(위 2번). 처음부터 off 폴더로
    들어온 문서는 그 경로를 타지 않고 INDEX_DOCUMENT로 색인되므로, 색인 시점의
    effective_searchable 계산에 폴더 상태가 들어가야 한다 — 빠지면 대외비 폴더에 올린 문서가
    검색에 그대로 노출된다.
    """
    folder = (await client.post('/kms/folders', json={'name': '대외비'})).json()
    assert (await client.patch(f"/kms/folders/{folder['id']}", json={'is_searchable': False})).status_code == 200

    res = await client.post(
        '/kms/documents',
        files={
            'file': ('환불정책.md', MD, 'text/markdown'),
            'document-data': ('blob', json.dumps({'folder_id': folder['id']}), 'application/json'),
        },
    )
    assert res.status_code == 200, res.text
    doc_id = res.json()['document_id']
    await ingest(doc_id)

    async with AsyncSessionLocal() as s:
        cands = await retrieve_candidates(s, tenant_id, '반품 기간', top_n=20)
    assert [c for c in cands.chunks if c.document_id == doc_id] == [], '참조 off 폴더 문서가 검색 후보에 떴다'


@pytest.mark.asyncio
async def test_일괄_참조끄기가_한_행으로_모든_문서를_제외한다(client, tenant_id, fake_queue, blob_tmp, fake_embed):
    """일괄 변경이 남기는 META 행은 **하나**인데, 그 한 행이 담긴 문서 전부를 갱신해야 한다.

    위 2번(단건 토글)은 문서 하나짜리 payload를 본다. 여기서 보는 건 payload의 document_ids가
    여러 개일 때다 — 워커가 최종값이 같은 문서끼리 묶어 한 번에 갱신하는 경로(sync_meta_documents_now)라
    한쪽만 반영되면 "체크해서 껐는데 일부가 계속 검색되는" 형태로 샌다.
    """
    a = await upload(client, '환불정책.md', MD)
    b = await upload(client, '배송정책.md', MD)
    await ingest(a['document_id'])
    await ingest(b['document_id'])
    assert {a['document_id'], b['document_id']} <= await cited_ids(tenant_id)

    res = await client.patch(
        '/kms/documents', json={'document_ids': [a['document_id'], b['document_id']], 'is_searchable': False}
    )
    assert res.status_code == 200, res.text

    # 한 행에 둘 다 담겨 있으므로 drain은 한 번이면 된다 (pending_row_ids가 document_ids 배열도 찾는다)
    assert await ingest(a['document_id']) == {'done': 1, 'failed': 0}
    ids = await cited_ids(tenant_id)
    assert a['document_id'] not in ids and b['document_id'] not in ids
