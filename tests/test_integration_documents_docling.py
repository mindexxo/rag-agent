"""문서 업로드 → 워커 색인까지 docling 실경로 — 시트·병합·행 묶음·중첩 표·첨부 상한과 그림 캡션 실패 처리."""

import pytest
from sqlalchemy import select

from database import AsyncSessionLocal
from rag import outbox
from rag.models import SearchIndexOutbox
from tests.conftest import ingest
from tests.helpers_documents import DOCX_MIME, chunk_texts, get_doc, upload_xlsx
from tests.helpers_xlsx import XLSX_MIME, max_rows_at_limit, wb_bytes, xlsx_bytes

# ── G. docling 경로 — 업로드 → 워커 색인까지 실제 경로 ─────────────────────


@pytest.mark.asyncio
async def test_G1_숨긴_시트는_업로드를_막지_않고_색인에서_빠진다(client, tenant_id, fake_queue, blob_tmp):
    """숨긴 시트는 읽지 않는다 — 문서를 failed로 만들지 않고 행 상한에도 안 센다."""

    def build(wb):
        wb.active.title = '보임'
        wb.active.append(['등급', '적립률'])
        wb.active.append(['VIP', '2.0'])
        hidden = wb.create_sheet('참조')
        hidden.append(['k'])
        for i in range(5000):  # 크기 상한 초과 — 숨겼으니 거절 사유가 아니다
            hidden.append([i])
        hidden.sheet_state = 'hidden'

    doc = await upload_xlsx(client, '숨김표.xlsx', '등급 적립률', wb_bytes(build))
    await ingest(doc['document_id'])
    assert (await get_doc(doc['document_id'])).status == 'ready'
    texts = await chunk_texts(doc['document_id'])
    assert len(texts) == 1 and '| VIP | 2.0 |' in texts[0]


@pytest.mark.asyncio
async def test_G2_병합_제목은_한_번만_색인된다(client, tenant_id, fake_queue, blob_tmp):
    def build(wb):
        ws = wb.active
        ws.append(['업무 교통비 청구서', None, None])
        ws.merge_cells('A1:C1')
        ws.append(['구분', '항목', '금액'])
        ws.append(['교통', '버스', 1500])

    doc = await upload_xlsx(client, '양식.xlsx', '교통비 양식', wb_bytes(build))
    await ingest(doc['document_id'])
    assert (await get_doc(doc['document_id'])).status == 'ready'
    texts = await chunk_texts(doc['document_id'])
    assert '\n'.join(texts).count('업무 교통비 청구서') == 1  # 제목 블록·표 묶음으로 나뉘어도 제목은 한 번


@pytest.mark.asyncio
async def test_G4_큰_시트는_행_묶음으로_색인되고_근거는_시트_완성본이다(client, tenant_id, fake_queue, blob_tmp):
    """색인(행 묶음·묶음마다 완성본) → 조회(fetch_chunk_map이 table_text를 돌려줌) → 근거 준비(완성본으로 확장)"""
    from rag.search import os_search
    from rag.search.retriever import prepare_evidence

    def build(wb):
        ws = wb.active
        ws.title = '단가'
        ws.append(['상품코드', '상품명', '단가'])
        for i in range(200):
            ws.append([f'A-{i:04d}', f'상품 {i}', 1000 + i])

    doc = await upload_xlsx(client, '단가표.xlsx', '상품 단가표', wb_bytes(build))
    await ingest(doc['document_id'])
    assert (await get_doc(doc['document_id'])).status == 'ready'
    resp = await os_search.client().search(
        index=os_search.settings.opensearch_index,
        body={'size': 100, '_source': ['chunk_id'], 'query': {'term': {'document_id': doc['document_id']}}},
    )
    ids = sorted(int(h['_source']['chunk_id']) for h in resp['hits']['hits'])
    assert len(ids) > 1  # 행 묶음 여럿
    cmap = await os_search.fetch_chunk_map(ids)
    chunks = [cmap[i] for i in ids]
    assert all(c.is_table and c.table_text == chunks[0].table_text for c in chunks)
    ev = prepare_evidence(chunks[::-1], top_k=5)  # 아무 묶음이 걸려도
    assert len(ev) == 1  # 한 칸으로 접히고
    whole = ev[0].text.split('[표 전체]\n', 1)[1]  # 걸린 묶음 먼저 + 표 전체
    assert ev[0].text.startswith('[검색에 걸린 행]\n')
    assert (
        whole == chunks[0].table_text and whole.count('| 상품코드 |') == 1 and 'A-0000' in whole and 'A-0199' in whole
    )  # 시트 전체


@pytest.mark.asyncio
async def test_G3_docx_중첩_표_내용이_색인된다(client, tenant_id, fake_queue, blob_tmp):
    """python-docx 경로는 셀 안의 표를 못 읽었다(인사규정 2건 문장 373·435개 누락) — docling 경로로 색인까지 확인."""
    from io import BytesIO

    from docx import Document as Docx

    d = Docx()
    d.add_heading('인사규정', level=1)
    d.add_paragraph('직급 체계는 아래와 같다.')
    inner = d.add_table(rows=1, cols=1).cell(0, 0).add_table(rows=1, cols=2)
    inner.cell(0, 0).text = '중첩표고유문구'
    inner.cell(0, 1).text = 'CL2'
    buf = BytesIO()
    d.save(buf)
    res = await client.post(
        '/kms/documents',
        files={'file': ('규정.docx', buf.getvalue(), DOCX_MIME)},
    )
    assert res.status_code == 200, res.text
    doc_id = res.json()['document_id']
    await ingest(doc_id)
    assert (await get_doc(doc_id)).status == 'ready'
    assert any('중첩표고유문구' in t for t in await chunk_texts(doc_id))


@pytest.mark.asyncio
async def test_G5_첨부_엑셀이_크면_추출_불가가_아니라_첨부_상한_안내가_나간다(client, tenant_id):
    """색인용 크기 상한이 첨부에 걸리면 "추출할 수 없는 파일"(422)로 사유가 가려진다.
    첨부는 자체 상한(6천 자)의 413 안내를 받아야 한다."""
    content = xlsx_bytes(max_rows_at_limit() + 50)
    res = await client.post('/kms/attachments/extract', files={'file': ('큰표.xlsx', content, XLSX_MIME)})
    assert res.status_code == 413, res.text
    assert '필요한 부분만 잘라 첨부' in res.json()['detail']


# ── H. 색인 시간 상한·캡션 실패 — 업로드 → 워커 drain까지 실제 경로 ─────────────────
# VLM은 로컬 대역(conftest.fake_vlm). 변환은 변환 프로세스에서, 분류·재시도는 outbox에서 — 둘의 이음매를 본다.


def _docx_with_pictures(n: int) -> bytes:
    from io import BytesIO

    from docx import Document as Docx
    from PIL import Image

    png = BytesIO()
    Image.new('RGB', (120, 80), (200, 30, 30)).save(png, format='PNG')
    d = Docx()
    d.add_heading('화면 매뉴얼', level=1)
    d.add_paragraph('로그인 절차를 설명한다.')
    for _ in range(n):
        d.add_picture(BytesIO(png.getvalue()))
    buf = BytesIO()
    d.save(buf)
    return buf.getvalue()


async def _outbox_row(doc_id: int) -> SearchIndexOutbox:
    async with AsyncSessionLocal() as s:
        return (
            await s.execute(
                select(SearchIndexOutbox)
                .where(SearchIndexOutbox.op == outbox.INDEX_DOCUMENT)
                .where(SearchIndexOutbox.payload['document_id'].as_integer() == doc_id)
            )
        ).scalar_one()


@pytest.mark.asyncio
async def test_H1_VLM_장애면_문서는_대기로_재시도되고_복구되면_ready(client, tenant_id, fake_queue, blob_tmp, fake_vlm):
    """VLM 장애는 일시 실패다 — 그림이 자리표시로 남은 채 ready가 되면 조용한 불완전 색인이 된다."""
    fake_vlm.status = 500
    res = await client.post('/kms/documents', files={'file': ('매뉴얼.docx', _docx_with_pictures(1), DOCX_MIME)})
    assert res.status_code == 200, res.text
    doc_id = res.json()['document_id']
    await ingest(doc_id)
    row = await _outbox_row(doc_id)
    assert (await get_doc(doc_id)).status == 'pending'  # failed가 아니다 — 일시 실패
    assert row.status == outbox.PENDING and row.attempts == 1 and '캡션' in row.last_error
    assert row.next_attempt_at is not None  # 백오프가 걸렸다

    fake_vlm.status = 200  # VLM 복구
    await ingest(doc_id)  # row_ids 경로는 백오프를 무시한다(테스트 계약)
    assert (await get_doc(doc_id)).status == 'ready'
    from tests.conftest import FAKE_CAPTION

    assert any(FAKE_CAPTION in t for t in await chunk_texts(doc_id))


@pytest.mark.asyncio
async def test_H2_그림이_상한을_넘으면_사유와_함께_바로_failed(
    client, tenant_id, fake_queue, blob_tmp, fake_vlm, monkeypatch
):
    """결정적 실패라 재시도 없이 1회로 확정되고, 사유가 FE에 그대로 보인다."""
    from config import settings

    monkeypatch.setattr(settings, 'docling_max_pictures', 2)
    res = await client.post('/kms/documents', files={'file': ('캡처.docx', _docx_with_pictures(3), DOCX_MIME)})
    doc_id = res.json()['document_id']
    await ingest(doc_id)
    doc = await get_doc(doc_id)
    assert doc.status == 'failed' and '그림 3장' in doc.status_reason
    assert (await _outbox_row(doc_id)).attempts == 1
    assert fake_vlm.calls == 0  # 넘는 문서에 VLM 시간을 쓰지 않았다
