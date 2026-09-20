"""eval/assign_folders.py — 폴더 배정이 PG와 엔진 청크 메타를 함께 바꾸는가 (#187).

스크립트는 운영 이미지 밖이지만 골드셋 구축 때 사람이 실제로 돌린다. PG만 바꾸고 outbox 행을 안 남기던
결함(엔진 메타 드리프트 영구, reconcile로도 안 잡힘)을 고정한다. classify 순서 규칙도 여기서 본다.
"""
import sys

import pytest
from sqlalchemy import select

from config import settings
from database import AsyncSessionLocal
from eval.assign_folders import FOLDERS, classify, main
from rag import os_client, os_index, outbox
from rag.models import Document, Folder, SearchIndexOutbox
from tests.conftest import ingest

MD = '# 환불 정책\n\n## 1. 기간\n\n단순변심 반품은 14일 이내 신청한다.\n'.encode()


@pytest.mark.parametrize('filename,title,key', [
    ('상담스크립트_반품.md', '', 'script'),
    ('상품관리기준표.xlsx', '', 'table'),           # 확장자가 '기준'보다 먼저
    ('세척·사용 안내 기준.md', '', 'guide'),        # guide가 policy보다 먼저
    ('환불정책.md', '사용 안내', 'policy'),         # 파일명 '정책'은 안내 낱말이 섞여도 정책
    ('규정집.md', '취업 규정', 'policy'),
    ('아무거나.md', '', 'guide'),                   # 폴백
])
def test_classify_순서(filename, title, key):
    assert classify(filename, title) == key


async def _run(argv: list[str], monkeypatch) -> None:
    monkeypatch.setattr(sys, 'argv', ['assign_folders', *argv])
    await main()


async def _first_chunk_meta(doc_id: int) -> dict:
    resp = await os_client.client().get(index=settings.opensearch_index,
                                        id=str(os_index.chunk_os_id(document_id=doc_id, chunk_index=0)))
    src = resp['_source']
    return {k: src.get(k) for k in ('folder_id', 'folder_name', 'folder_description', 'searchable')}


@pytest.mark.asyncio
async def test_apply는_PG와_엔진_메타를_같은_커밋의_META_행으로_맞춘다(client, tenant_id, fake_queue, blob_tmp,
                                                                fake_embed, monkeypatch):
    res = await client.post('/kms/documents', files={'file': ('환불정책.md', MD, 'text/markdown')})
    assert res.status_code == 200, res.text
    doc_id = res.json()['document_id']
    assert await ingest(doc_id) == {'done': 1, 'failed': 0}
    assert (await _first_chunk_meta(doc_id))['folder_name'] is None     # 미분류로 색인됨

    await _run(['--apply', '--tenant', tenant_id], monkeypatch)

    name, desc = FOLDERS['policy']
    async with AsyncSessionLocal() as s:
        doc = await s.get(Document, doc_id)
        folder = await s.get(Folder, doc.folder_id)
        metas = (await s.execute(select(SearchIndexOutbox)
                                 .where(SearchIndexOutbox.tenant_id == tenant_id)
                                 .where(SearchIndexOutbox.op == outbox.META_DOCUMENTS))).scalars().all()
    assert (folder.name, folder.description, folder.is_searchable) == (name, desc, True)
    assert [(m.status, m.payload) for m in metas] == [('done', {'document_ids': [doc_id]})]   # 스크립트가 바로 drain
    assert await _first_chunk_meta(doc_id) == {
        'folder_id': folder.id, 'folder_name': name, 'folder_description': desc, 'searchable': True}

    # dry-run은 아무것도 남기지 않는다 — 행도 rollback
    await _run(['--clear', '--tenant', tenant_id], monkeypatch)
    assert (await _first_chunk_meta(doc_id))['folder_name'] == name
    async with AsyncSessionLocal() as s:
        assert (await s.get(Document, doc_id)).folder_id == folder.id
        n = len((await s.execute(select(SearchIndexOutbox.id)
                                 .where(SearchIndexOutbox.tenant_id == tenant_id)
                                 .where(SearchIndexOutbox.op == outbox.META_DOCUMENTS))).all())
    assert n == 1

    # --clear --apply: 배정 해제 + 폴더 삭제가 엔진 메타까지 지운다
    await _run(['--clear', '--apply', '--tenant', tenant_id], monkeypatch)
    async with AsyncSessionLocal() as s:
        assert (await s.get(Document, doc_id)).folder_id is None
        assert await s.get(Folder, folder.id) is None
    assert await _first_chunk_meta(doc_id) == {
        'folder_id': None, 'folder_name': None, 'folder_description': None, 'searchable': True}
