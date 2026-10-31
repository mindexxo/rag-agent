"""#233 조문 인용 ≠ 조 제목 — 업로드 → 워커 색인 → OpenSearch까지 실제 경로(E2E 목록에서 옮긴 것).

청킹 단위 판정은 tests/test_citation_headings.py가 본다. 여기는 그 결과가 **색인 필드**(본문·`identifiers`)까지 그대로
가는지와 업로드 계약을 본다:
  1. 스타일 없는 워드(조문 폴백): 인용 문장이 본문으로 색인되고, 문서에 없는 조('12조')의 이름표가 없고, 본문 없는 조('7조')와
     '의' 생략 가지 조('5조의1', #250)의 이름표는 있다
  2. PDF(docling): '(이하 …)' 인용 줄이 가짜 이름표('15조')가 되지 않고, 문서 끝 '제 7 조(삭제)'가 색인된다
  3. 헤딩만 있는 문서는 ready가 아니라 failed — 결정 3(섹션 0개 → "추출된 텍스트 없음")

문서는 sample_docs/corpus_v2의 합성 문서(정본 생성기 _src/build_citation_docs.py) — 내용은 전부 지어낸 것이다.
"""
from pathlib import Path

import pytest

from rag.search import os_search
from tests.conftest import ingest
from tests.helpers_documents import chunk_texts, get_doc, upload

pytest.importorskip('docling')

CORPUS = Path(__file__).resolve().parent.parent / 'sample_docs' / 'corpus_v2'
DOCX_MIME = 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'


async def _ingest_file(client, path: Path, mime: str) -> int:
    body = await upload(client, path.name, path.read_bytes(), mime=mime)
    await ingest(body['document_id'])
    doc = await get_doc(body['document_id'])
    assert doc.status == 'ready', doc.status_reason
    return body['document_id']


@pytest.mark.asyncio
async def test_스타일_없는_워드_인용은_본문_없는_조는_이름표(client, tenant_id, fake_queue, blob_tmp):
    doc_id = await _ingest_file(client, CORPUS / 'aromanica' / 'aromanica_16_시향관운영지침.docx', DOCX_MIME)
    text = '\n'.join(await chunk_texts(doc_id))
    assert '제4조 및 제5조에 따라 대여한 시향 도구의 반납 기한은 대여일로부터 10일로 한다.' in text   # 예전: 제목으로 올라가 소실
    assert '제6조 제2항의 휴관일은 매월 첫째 월요일로 한다.' in text
    assert await os_search.search_identifiers(tenant_id, ['12조'], 10) == []        # 예전: "(이하 …)에 따른" 줄이 이름표
    assert await os_search.search_identifiers(tenant_id, ['7조'], 10)               # 본문 없는 '제7조(삭제)'
    assert await os_search.search_identifiers(tenant_id, ['5조의1'], 10)            # '제 5 조1(분실 신고)' (#250)


@pytest.mark.asyncio
async def test_PDF_이하_인용은_이름표가_아니고_문서_끝_조는_색인된다(client, tenant_id, fake_queue, blob_tmp):
    doc_id = await _ingest_file(client, CORPUS / 'goodpeople' / 'goodpeople_16_후원물품보관규정.pdf', 'application/pdf')
    assert await os_search.search_identifiers(tenant_id, ['15조'], 10) == []
    assert await os_search.search_identifiers(tenant_id, ['7조'], 10)
    assert any('제 7 조(삭제)' in t for t in await chunk_texts(doc_id))             # 예전: 문서 끝 헤딩은 버려졌다


@pytest.mark.asyncio
async def test_헤딩만_있는_문서는_failed(client, tenant_id, fake_queue, blob_tmp):
    body = await upload(client, '빈양식.md', '# 양식\n\n## 제1조(목적)\n\n## 제2조(정의)\n'.encode())
    r = await ingest(body['document_id'])
    assert r['failed'] == 1
    doc = await get_doc(body['document_id'])
    assert doc.status == 'failed' and '추출된 텍스트가 없습니다' in doc.status_reason
