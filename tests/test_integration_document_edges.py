"""문서 API 경계 — 확장자·크기·파일명 판정, 다운로드·exists·목록·단건 수정 갈래, 이미지 경고.

HTTP로만 확인하고, 상태는 DB·대기열 행으로 본다. 색인은 워커 대신 `ingest`(그 문서의 대기열 행만 drain)로 태운다.
"""

import io

import httpx
import pytest
from docx import Document as DocxDocument
from reportlab.pdfgen import canvas
from sqlalchemy import select

from database import AsyncSessionLocal
from rag import ingestion
from rag.models import Document, SearchIndexOutbox
from rag.turn import cache
from schemas.kms import DOC_LIST_MAX_LIMIT
from tests.conftest import ingest
from tests.helpers_documents import MD, XLSX_TABLE, doc_data, get_doc, list_docs, make_folder, upload
from tests.helpers_xlsx import XLSX_MIME

DOC_MAX = 10 * 1024 * 1024
UNREADABLE = {'detail': '텍스트를 추출할 수 없는 파일입니다.'}


def _lenient_client(tenant: str) -> httpx.AsyncClient:
    """앱 예외를 테스트로 되던지지 않고 500 응답으로 받는 클라이언트."""
    from main import app

    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
        base_url='http://testserver',
        headers={'X-Tenant-Id': tenant},
    )


def _pdf_bytes(text: str) -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    c.drawString(72, 720, text)
    c.save()
    return buf.getvalue()


def _docx_bytes(text: str) -> bytes:
    buf = io.BytesIO()
    d = DocxDocument()
    d.add_paragraph(text)
    d.save(buf)
    return buf.getvalue()


async def _meta_rows(doc_id: int) -> int:
    async with AsyncSessionLocal() as s:
        rows = (await s.execute(select(SearchIndexOutbox).where(SearchIndexOutbox.op == 'meta_documents'))).scalars()
        return sum(1 for r in rows if doc_id in (r.payload.get('document_ids') or []))


async def _post(client, filename: str, content: bytes, mime: str = 'text/markdown'):
    return await client.post('/kms/documents', files={'file': (filename, content, mime)})


async def _doc_count(tenant: str) -> int:
    async with AsyncSessionLocal() as s:
        return len((await s.execute(select(Document).where(Document.tenant_id == tenant))).scalars().all())


# ── 업로드: 확장자·크기 ──


@pytest.mark.asyncio
async def test_미지원_확장자는_업로드와_같은_문구로_400(client, tenant_id, blob_tmp):
    """DOC-02·DOC-03 — 문구까지 고정한다. 확장자 없음·점으로 시작·끝 공백은 '확장자 없음' 또는 판정 불가로 400."""
    res = await _post(client, '악성.exe', b'x', 'application/octet-stream')
    assert res.status_code == 400
    assert res.json() == {'detail': '지원하지 않는 형식입니다: .exe. 지원: PDF/DOCX/XLSX/TXT/MD'}
    for name in ('README', '.md', 'a.md '):
        res = await _post(client, name, MD)
        assert res.status_code == 400, name
        assert res.json()['detail'].startswith('지원하지 않는 형식입니다'), name
    assert '확장자 없음' in (await _post(client, 'README', MD)).json()['detail']
    assert await _doc_count(tenant_id) == 0


@pytest.mark.asyncio
async def test_대문자_확장자도_색인돼_ready가_되고_파일명은_원문_그대로(
    client, tenant_id, fake_queue, blob_tmp, fake_embed
):
    """DOC-04."""
    doc = await upload(client, '환불정책.MD', MD)
    assert doc['filename'] == '환불정책.MD'
    assert await ingest(doc['document_id']) == {'done': 1, 'failed': 0}
    assert (await get_doc(doc['document_id'])).status == 'ready'


@pytest.mark.asyncio
async def test_업로드_크기는_10MB까지_받고_넘으면_흔적_없이_413(client, tenant_id, fake_queue, blob_tmp):
    """DOC-07 — 정확히 10MB는 통과, +1바이트는 본문 검사로 413. Content-Length 선차단은 test_upload_guard가 단위로 고정."""
    exact = b'a' * DOC_MAX
    assert (await _post(client, '큰문서.txt', exact, 'text/plain')).status_code == 200

    res = await _post(client, '넘는문서.txt', exact + b'a', 'text/plain')
    assert res.status_code == 413 and res.json() == {'detail': '파일이 10MB를 초과합니다.'}

    assert await _doc_count(tenant_id) == 1  # 통과한 하나만
    assert len(list(blob_tmp.rglob('*.txt'))) == 1


@pytest.mark.asyncio
async def test_빈_xlsx는_업로드에서_400이고_빈_txt는_색인에서_failed(
    client, tenant_id, fake_queue, blob_tmp, fake_embed
):
    """DOC-10 — md는 기존 테스트가 고정. xlsx는 사전검증에서, txt는 색인에서 끝난다."""
    res = await client.post(
        '/kms/documents',
        files={'file': ('빈표.xlsx', b'', XLSX_MIME), **doc_data(description='빈 표')},
    )
    assert res.status_code == 400
    doc = await upload(client, '빈파일.txt', b'', mime='text/plain')
    await ingest(doc['document_id'])
    row = await get_doc(doc['document_id'])
    assert row.status == 'failed' and '추출된 텍스트가 없습니다' in (row.status_reason or '')


@pytest.mark.asyncio
async def test_설명은_200자까지_두_수신_경로_모두_받는다(client, tenant_id, fake_queue, blob_tmp):
    """DOC-20 — 201자 거절은 기존 테스트가 고정. document-data 파트와 개별 필드 둘 다."""
    desc = 'ㄱ' * 200
    a = await client.post(
        '/kms/documents', files={'file': ('a.xlsx', XLSX_TABLE, XLSX_MIME), **doc_data(description=desc)}
    )
    b = await client.post(
        '/kms/documents', files={'file': ('b.xlsx', XLSX_TABLE, XLSX_MIME)}, data={'description': desc}
    )
    assert a.status_code == b.status_code == 200, (a.text, b.text)
    assert a.json()['description'] == b.json()['description'] == desc


@pytest.mark.asyncio
async def test_expect_version은_비정수_422_음수는_409(client, tenant_id, fake_queue, blob_tmp):
    """DOC-49."""
    res = await client.post(
        '/kms/documents', files={'file': ('a.md', MD, 'text/markdown')}, data={'expect_version': 'abc'}
    )
    assert res.status_code == 422
    res = await client.post(
        '/kms/documents', files={'file': ('a.md', MD, 'text/markdown')}, data={'expect_version': '-1'}
    )
    assert res.status_code == 409
    assert await _doc_count(tenant_id) == 0


# ── 다운로드·exists·목록 ──


@pytest.mark.asyncio
async def test_다운로드는_원본_바이트_blob이_없으면_410_없는_id는_404(client, tenant_id, fake_queue, blob_tmp):
    """DOC-106 — 삭제 뒤 200은 기존 테스트가 고정."""
    doc = await upload(client, '환불정책.md', MD)
    res = await client.get(f'/kms/documents/{doc["document_id"]}/download')
    assert res.status_code == 200 and res.content == MD

    for f in blob_tmp.rglob('*.md'):
        f.unlink()
    res = await client.get(f'/kms/documents/{doc["document_id"]}/download')
    assert res.status_code == 410 and res.json() == {'detail': 'blob file missing'}
    res = await client.get('/kms/documents/999999999/download')
    assert res.status_code == 404 and res.json() == {'detail': 'document not found'}


@pytest.mark.asyncio
async def test_exists는_filename이_없으면_422(client, tenant_id):
    """DOC-59 — 나머지 갈래는 test_exists_* 5건이 고정."""
    res = await client.get('/kms/documents/exists')
    assert res.status_code == 422
    assert ('query', 'filename') in [tuple(e['loc']) for e in res.json()['detail']]


@pytest.mark.asyncio
async def test_목록_q가_공백뿐이면_필터가_아니다(client, tenant_id, fake_queue, blob_tmp):
    """DOC-66 — 공백 q가 '%%'로 전건 매칭되거나 0건이 되지 않고, 미전송과 같다."""
    await upload(client, '환불정책.md', MD)
    await upload(client, '배송정책.md', MD)
    plain, blank = await list_docs(client), await list_docs(client, q='   ')
    assert blank['total'] == plain['total'] == 2
    assert [d['document_id'] for d in blank['items']] == [d['document_id'] for d in plain['items']]


@pytest.mark.asyncio
async def test_목록_limit은_상한으로_보정되고_offset_음수는_0(client, tenant_id, fake_queue, blob_tmp):
    """DOC-62 — 0·-1·201·9999는 상한(200)으로, 200은 그대로. has_more·total은 보정된 값 기준."""
    for i in range(3):
        await upload(client, f'문서{i}.md', MD)
    for limit in (0, -1, DOC_LIST_MAX_LIMIT, DOC_LIST_MAX_LIMIT + 1, 9999):
        page = await list_docs(client, limit=limit)
        assert len(page['items']) == 3 and page['has_more'] is False and page['total'] == 3, limit
    assert len((await list_docs(client, offset=-3))['items']) == 3
    page = await list_docs(client, limit=2)
    assert len(page['items']) == 2 and page['has_more'] is True


# ── 단건 수정 ──


@pytest.mark.asyncio
async def test_단건_수정_대상이_없거나_폴더가_남의_것이면_404이고_문서는_그대로(
    client, tenant_id, other_tenant_id, fake_queue, blob_tmp
):
    """DOC-82 — deleted는 기존 테스트가 고정. folder_id=null은 미분류로 뗀다."""
    res = await client.patch('/kms/documents/999999999', json={'is_searchable': False})
    assert res.status_code == 404 and res.json() == {'detail': 'document not found'}

    folder = await make_folder(client, '정책')
    doc = await upload(client, '환불정책.md', MD, extra_parts=doc_data(folder_id=folder))
    async with _lenient_client(other_tenant_id) as other:
        others_folder = (await other.post('/kms/folders', json={'name': '남의폴더'})).json()['id']
    for target in (999999999, others_folder):
        res = await client.patch(f'/kms/documents/{doc["document_id"]}', json={'folder_id': target})
        assert res.status_code == 404 and res.json() == {'detail': 'folder not found'}, target
    assert (await get_doc(doc['document_id'])).folder_id == folder

    res = await client.patch(f'/kms/documents/{doc["document_id"]}', json={'folder_id': None})
    assert res.status_code == 200 and res.json()['folder_id'] is None


@pytest.mark.asyncio
async def test_단건_검색_끄기만_캐시를_무효화하고_META_행을_남긴다(
    client, tenant_id, fake_queue, blob_tmp, monkeypatch
):
    """DOC-80 — on→off만 캐시를 버린다(off→on은 버릴 캐시가 없다). 두 번 모두 META 행이 생긴다."""
    doc_id = (await upload(client, '환불정책.md', MD))['document_id']
    invalidated: list[int] = []
    real = cache.invalidate_source

    async def _spy(session, tenant: str, document_id: int) -> object:
        invalidated.append(document_id)
        return await real(session, tenant, document_id)

    monkeypatch.setattr(cache, 'invalidate_source', _spy)
    assert (await client.patch(f'/kms/documents/{doc_id}', json={'is_searchable': False})).status_code == 200
    assert invalidated == [doc_id] and await _meta_rows(doc_id) == 1
    assert (await client.patch(f'/kms/documents/{doc_id}', json={'is_searchable': True})).status_code == 200
    assert invalidated == [doc_id] and await _meta_rows(doc_id) == 2


@pytest.mark.asyncio
async def test_xlsx가_아닌_문서의_설명은_빈_값으로_지운다(client, tenant_id, fake_queue, blob_tmp):
    """DOC-89."""
    doc_id = (await upload(client, '환불정책.md', MD, extra_parts=doc_data(description='환불 규정')))['document_id']
    res = await client.patch(f'/kms/documents/{doc_id}', json={'description': ''})
    assert res.status_code == 200 and res.json()['description'] is None
    assert (await get_doc(doc_id)).description is None


@pytest.mark.asyncio
async def test_일괄_수정의_중복_id는_한_번만_처리하고_description은_무시한다(client, tenant_id, fake_queue, blob_tmp):
    """DOC-96 — description 무시는 스펙으로 확정한 현재 동작."""
    doc_id = (await upload(client, '환불정책.md', MD, extra_parts=doc_data(description='원래 설명')))['document_id']
    res = await client.patch(
        '/kms/documents', json={'document_ids': [doc_id, doc_id], 'is_searchable': False, 'description': '바뀐 설명'}
    )
    assert res.status_code == 200
    assert [d['document_id'] for d in res.json()] == [doc_id]
    row = await get_doc(doc_id)
    assert row.is_searchable is False and row.description == '원래 설명'
    assert (await client.delete('/kms/documents', params=[('ids', doc_id), ('ids', doc_id)])).status_code == 204
    assert (await get_doc(doc_id)).status == 'deleted'


@pytest.mark.asyncio
async def test_이미지_비중이_크면_ready이면서_경고_사유가_붙는다(
    client, tenant_id, fake_queue, blob_tmp, fake_embed, monkeypatch
):
    """DOC-114 — FE 경고 표시 계약. 검색·인용은 정상이라 ready 그대로다."""
    monkeypatch.setattr(ingestion, 'pdf_image_area_ratio', lambda path: 62.0)
    doc_id = (await upload(client, '이미지많은문서.md', MD))['document_id']
    assert await ingest(doc_id) == {'done': 1, 'failed': 0}
    row = await get_doc(doc_id)
    assert row.status == 'ready'
    assert row.status_reason.startswith('이미지가 쪽 면적의 62%를 차지합니다.')
    listed = (await list_docs(client))['items'][0]
    assert listed['status'] == 'ready' and listed['status_reason'] == row.status_reason
