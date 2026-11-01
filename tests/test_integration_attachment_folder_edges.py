"""첨부 추출·폴더 API 경계 — 형식별 추출·크기·손상·파일명·텍스트 상한, 폴더 정렬·중복·삭제 조건·메타 반영.

HTTP로만 확인하고, 상태는 DB·대기열 행으로 본다. 색인은 워커 대신 `ingest`(그 문서의 대기열 행만 drain)로 태운다.
"""

import io

import httpx
import pytest
from docx import Document as DocxDocument
from reportlab.pdfgen import canvas
from sqlalchemy import select

from database import AsyncSessionLocal
from rag.models import Document, SearchIndexOutbox
from rag.turn import cache
from routers.attachments import ATTACHMENT_MAX_FILE_BYTES
from schemas.kms import ATTACHMENT_MAX_TEXT_CHARS, FOLDER_DESC_MAX
from tests.conftest import ingest
from tests.helpers_documents import MD, XLSX_TABLE, doc_data, get_doc, make_folder, upload
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


# ── 첨부 ──


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('name', 'content', 'mime', 'expected'),
    [
        ('메모.txt', '반품은 14일.'.encode(), 'text/plain', '반품은 14일.'),
        ('규정.pdf', _pdf_bytes('Refund within 14 days'), 'application/pdf', 'Refund within 14 days'),
        (
            '규정.docx',
            _docx_bytes('단순변심 반품은 14일 이내'),
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            '단순변심 반품은 14일 이내',
        ),
        ('표.xlsx', XLSX_TABLE, XLSX_MIME, None),
    ],
    ids=['txt', 'pdf', 'docx', 'xlsx'],
)
async def test_첨부_추출은_형식별로_텍스트를_돌려준다(client, tenant_id, name, content, mime, expected):
    """ATT-01 — md는 기존 테스트가 고정. 저장하지 않는 것도 기존 불변식 테스트가 고정."""
    res = await client.post('/kms/attachments/extract', files={'file': (name, content, mime)})
    assert res.status_code == 200, res.text
    body = res.json()
    assert body['filename'] == name and body['text']
    if expected is not None:
        assert expected in body['text']


@pytest.mark.asyncio
async def test_첨부_크기는_5MB까지_받는다(client, tenant_id):
    """ATT-05 — 정확히 5MB는 크기 검사를 통과(텍스트 상한에서 413), +1바이트는 크기 413. Content-Length 선차단은 단위 테스트가 고정."""
    exact = b'a ' * (ATTACHMENT_MAX_FILE_BYTES // 2)
    res = await client.post('/kms/attachments/extract', files={'file': ('큰.txt', exact, 'text/plain')})
    assert res.status_code == 413 and '문서가 너무 깁니다' in res.json()['detail']  # 크기는 통과했다

    res = await client.post('/kms/attachments/extract', files={'file': ('큰.txt', exact + b'a', 'text/plain')})
    assert res.status_code == 413 and res.json() == {'detail': '파일이 5MB를 초과합니다.'}


@pytest.mark.asyncio
async def test_손상된_첨부는_422(client, tenant_id):
    """ATT-11."""
    for name, mime in (
        ('손상.pdf', 'application/pdf'),
        ('손상.docx', 'application/octet-stream'),
        ('손상.xlsx', XLSX_MIME),
    ):
        res = await client.post('/kms/attachments/extract', files={'file': (name, b'not a real file', mime)})
        assert res.status_code == 422 and res.json() == UNREADABLE, name


@pytest.mark.asyncio
async def test_첨부_파일명은_정규화_뒤_길이로_판정한다(client, tenant_id):
    """ATT-06 — NFD로 보내면 코드포인트가 늘어나지만 NFC 255자면 통과한다."""
    import unicodedata

    nfc = '가' * 252 + '.md'
    nfd = unicodedata.normalize('NFD', nfc)
    assert len(nfd) > 255 >= len(nfc)
    res = await client.post('/kms/attachments/extract', files={'file': (nfd, MD, 'text/markdown')})
    assert res.status_code == 200 and res.json()['filename'] == nfc


@pytest.mark.asyncio
async def test_첨부_텍스트는_6000자까지(client, tenant_id):
    """ATT-08 — 큰 xlsx 초과는 기존 테스트가 고정."""
    ok = ('ㄱ' * ATTACHMENT_MAX_TEXT_CHARS).encode()
    res = await client.post('/kms/attachments/extract', files={'file': ('딱.txt', ok, 'text/plain')})
    assert res.status_code == 200 and len(res.json()['text']) == ATTACHMENT_MAX_TEXT_CHARS
    over = ('ㄱ' * (ATTACHMENT_MAX_TEXT_CHARS + 1)).encode()
    res = await client.post('/kms/attachments/extract', files={'file': ('넘침.txt', over, 'text/plain')})
    assert res.status_code == 413
    assert res.json()['detail'].startswith('문서가 너무 깁니다 (추출 텍스트 6,001자 > 6,000자)')


# ── 폴더 ──


@pytest.mark.asyncio
async def test_폴더_목록은_이름순이고_설명은_공백을_정리한다(client, tenant_id):
    """FOL-01."""
    for name in ('하', '가', '나'):
        await make_folder(client, name)
    res = await client.post('/kms/folders', json={'name': '다', 'description': '  설명  '})
    assert res.status_code == 200 and res.json()['description'] == '설명'
    assert (await client.post('/kms/folders', json={'name': '라', 'description': '   '})).json()['description'] is None
    assert [f['name'] for f in (await client.get('/kms/folders')).json()] == ['가', '나', '다', '라', '하']


@pytest.mark.asyncio
async def test_폴더_생성_이름_중복은_409(client, tenant_id):
    """FOL-02 — 빈 값·공백은 스키마 422(test_integration_input_validation이 고정)."""
    await make_folder(client, '정책')
    res = await client.post('/kms/folders', json={'name': '정책'})
    assert res.status_code == 409 and res.json() == {'detail': "'정책' 폴더가 이미 있습니다."}


@pytest.mark.asyncio
async def test_폴더_이름을_다른_폴더와_같게_바꾸면_409(client, tenant_id):
    """FOL-02 — 수정 중복은 생성과 문구가 다르다. 실패해도 폴더 이름은 그대로여야 한다."""
    await make_folder(client, '정책')
    other = await make_folder(client, '공지')
    async with _lenient_client(tenant_id) as lenient:
        res = await lenient.patch(f'/kms/folders/{other}', json={'name': '정책'})
    assert res.status_code == 409 and res.json() == {'detail': '같은 이름의 폴더가 이미 있습니다.'}
    assert '공지' in [f['name'] for f in (await client.get('/kms/folders')).json()]


@pytest.mark.asyncio
async def test_폴더_이름_중복은_설명_토글과_함께_보내도_409이고_아무것도_안_바뀐다(client, tenant_id):
    await make_folder(client, '정책')
    other = await make_folder(client, '공지')
    async with _lenient_client(tenant_id) as lenient:
        res = await lenient.patch(
            f'/kms/folders/{other}', json={'name': '정책', 'description': '새 설명', 'is_searchable': False}
        )
    assert res.status_code == 409 and res.json() == {'detail': '같은 이름의 폴더가 이미 있습니다.'}
    got = next(f for f in (await client.get('/kms/folders')).json() if f['id'] == other)
    assert (got['name'], got['description'], got['is_searchable']) == ('공지', None, True)


@pytest.mark.asyncio
async def test_폴더_삭제는_failed_문서도_막고_없는_id는_404(client, tenant_id, fake_queue, blob_tmp, fake_embed):
    """FOL-14 — 문서 있음·deleted만은 기존 테스트가 고정. failed는 '사용자에게 없는 문서'지만 폴더 삭제는 막는다."""
    folder = await make_folder(client, '정책')
    doc = await upload(client, '빈파일.txt', b'', mime='text/plain', extra_parts=doc_data(folder_id=folder))
    await ingest(doc['document_id'])
    assert (await get_doc(doc['document_id'])).status == 'failed'
    res = await client.delete(f'/kms/folders/{folder}')
    assert res.status_code == 409 and res.json()['detail'].startswith('폴더에 문서 1개가 있어')

    res = await client.delete('/kms/folders/999999999')
    assert res.status_code == 404


@pytest.mark.asyncio
async def test_폴더_설명_200자는_통과하고_같은_설명_재전송은_캐시를_건드리지_않는다(
    client, tenant_id, fake_queue, blob_tmp, monkeypatch
):
    """FOL-06 — 201자 거절과 설명 변경 시 무효화는 기존 테스트가 고정."""
    res = await client.post('/kms/folders', json={'name': '정책', 'description': 'ㄱ' * FOLDER_DESC_MAX})
    assert res.status_code == 200
    folder = res.json()['id']
    await upload(client, '환불정책.md', MD, extra_parts=doc_data(folder_id=folder))
    calls: list[int] = []

    async def _spy(session, tenant: str, document_id: int) -> None:
        calls.append(document_id)

    monkeypatch.setattr(cache, 'invalidate_source', _spy)
    assert (
        await client.patch(f'/kms/folders/{folder}', json={'description': 'ㄱ' * FOLDER_DESC_MAX})
    ).status_code == 200
    assert calls == []


@pytest.mark.asyncio
async def test_폴더_이름을_바꾸면_소속_문서에_META_행이_생긴다(client, tenant_id, fake_queue, blob_tmp):
    """FOL-12."""
    folder = await make_folder(client, '정책')
    doc_id = (await upload(client, '환불정책.md', MD, extra_parts=doc_data(folder_id=folder)))['document_id']
    before = await _meta_rows(doc_id)
    assert (await client.patch(f'/kms/folders/{folder}', json={'name': '운영정책'})).status_code == 200
    assert await _meta_rows(doc_id) == before + 1
