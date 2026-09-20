"""failed 문서는 사용자에게 "없는 문서"다 — 재업로드는 새 version 대신 그 행을 되살린다 (#161).

exists·_current_version·handle_upload 셋이 한 기준(ALIVE_DOCUMENT_STATUSES)으로 답해야 하고,
되살릴 때 잔여 청크·캐시·경합까지 함께 정리된다. 규칙의 정의점은 rag/documents.py의 handle_upload.
공용 헬퍼는 tests/helpers_documents.py.
"""
from pathlib import Path

import pytest
from sqlalchemy import func, select, update

from database import AsyncSessionLocal
from rag import cache, outbox
from rag.models import AnswerCache as AnswerCacheRow, Document, SearchIndexOutbox
from tests.conftest import ingest
from tests.helpers_documents import (MD, chunk_texts, doc_data, get_doc, list_docs, make_folder,
                                     post_upload, upload)


# ── #161 failed 문서는 "없는 문서" — 재업로드는 그 행을 되살린다 ──

async def _make_failed(doc_id: int) -> None:
    """워커가 5회 실패해 failed로 굳힌 상태를 DB로 재현한다 — 문서 행과 대기열 행 둘 다."""
    async with AsyncSessionLocal() as s:
        await s.execute(update(Document).where(Document.id == doc_id)
                        .values(status='failed', is_active=False,
                                status_reason='색인 5회 실패: 테스트 유발'))
        await s.execute(update(SearchIndexOutbox)
                        .where(SearchIndexOutbox.payload['document_id'].as_integer() == doc_id)
                        .values(status='failed', attempts=5))
        await s.commit()


@pytest.mark.asyncio
async def test_failed만_있으면_없는_문서로_본다(client, tenant_id, fake_queue, blob_tmp):
    """exists=false, expect_version=0 통과 — 확인창이 뜨지 않는다. 세 판정(exists·_current_version·
    handle_upload)이 같은 기준이어야 하는 것을 이 테스트가 앞의 둘에서 고정한다."""
    doc = await upload(client, '환불정책.md', MD)
    await _make_failed(doc['document_id'])

    ex = (await client.get('/kms/documents/exists', params={'filename': '환불정책.md'})).json()
    assert ex == {'exists': False, 'document_id': None, 'version': None, 'status': None, 'uploaded_at': None}
    assert (await post_upload(client, '환불정책.md', MD, expect_version=0)).status_code == 200   # 새 문서처럼


@pytest.mark.asyncio
async def test_failed_재업로드는_같은_행을_되살린다(client, tenant_id, fake_queue, blob_tmp):
    """새 version이 아니라 그 행이 pending으로 돌아온다 — id·version 유지, 실패 흔적은 지워지고,
    옛 blob은 삭제되며, 등록자·등록일시는 이번 업로드 것으로 바뀐다."""
    v1 = await upload(client, '환불정책.md', MD, headers={'X-User-Id': 'agent-a'})
    await _make_failed(v1['document_id'])
    before = await get_doc(v1['document_id'])
    old_blob = Path(before.blob_path)
    assert old_blob.exists() and before.status == 'failed' and before.status_reason

    res = await upload(client, '환불정책.md', '# 다시 올림\n\n고친 내용\n'.encode(), headers={'X-User-Id': 'agent-b'})
    assert res['document_id'] == v1['document_id']       # 같은 행
    assert res['version'] == 1 and res['status'] == 'pending'
    assert res['status_reason'] is None                  # 옛 실패 사유가 남지 않는다
    assert res['uploaded_by'] == 'agent-b'               # 이번 업로드가 등록자

    after = await get_doc(v1['document_id'])
    assert after.blob_path != before.blob_path
    assert not old_blob.exists() and Path(after.blob_path).exists()   # 옛 blob 삭제, 새 blob 존재
    assert after.uploaded_at > before.uploaded_at        # 목록 최신순에서 위로 온다 (#164)
    # 처음 올린 것과 같은 출발점 — "색인됨" 흔적이 남으면 안 된다
    assert after.is_active is False
    assert after.indexed_at is None and after.char_count is None and after.page_count is None

    # 같은 이름의 행은 여전히 하나 — 새 version이 생기지 않았다
    async with AsyncSessionLocal() as s:
        n = (await s.execute(select(func.count()).select_from(Document)
                             .where(Document.tenant_id == tenant_id))).scalar()
    assert n == 1


@pytest.mark.asyncio
async def test_failed_재업로드_후_ingest하면_ready_v1_한_줄(client, tenant_id, fake_queue, blob_tmp):
    v1 = await upload(client, '환불정책.md', MD)
    await _make_failed(v1['document_id'])
    await upload(client, '환불정책.md', MD)
    assert await ingest(v1['document_id']) == {'done': 1, 'failed': 0}

    doc = await get_doc(v1['document_id'])
    assert doc.status == 'ready' and doc.is_active is True and doc.version == 1
    items = (await list_docs(client))['items']
    assert [(d['version'], d['status']) for d in items] == [(1, 'ready')]   # 두 줄이 아니다


@pytest.mark.asyncio
async def test_failed_재사용_경합은_한_명만_성공(client, tenant_id, fake_queue, blob_tmp, monkeypatch):
    """조건부 UPDATE(status='failed'인 동안만)가 경합을 가른다. 재사용 직전에 다른 세션이 먼저
    되살린 상황을 주입하면 이쪽은 0행 → 409, 행은 하나, 새 blob은 지워진다."""
    v1 = await upload(client, '환불정책.md', MD)
    await _make_failed(v1['document_id'])
    blobs_before = sorted((blob_tmp / tenant_id).iterdir())

    import rag.documents as rd
    real = rd._reuse_failed_row

    async def _race(session, target_id, **values):
        async with AsyncSessionLocal() as other:          # 다른 요청이 먼저 되살렸다
            await other.execute(update(Document).where(Document.id == target_id)
                                .values(status='pending', status_reason=None))
            await other.commit()
        return await real(session, target_id, **values)

    monkeypatch.setattr(rd, '_reuse_failed_row', _race)
    res = await post_upload(client, '환불정책.md', MD)
    assert res.status_code == 409
    assert sorted((blob_tmp / tenant_id).iterdir()) == blobs_before     # 새 blob은 남지 않는다
    async with AsyncSessionLocal() as s:
        n = (await s.execute(select(func.count()).select_from(Document)
                             .where(Document.tenant_id == tenant_id))).scalar()
    assert n == 1


@pytest.mark.asyncio
async def test_failed가_둘이면_최신을_되살리고_나머지는_내린다(client, tenant_id, fake_queue, blob_tmp):
    """옛 코드가 남긴 이력(v1 failed + v2 failed)도 정리된다 — 개발계 실측 0건이지만 방어."""
    v1 = await upload(client, '환불정책.md', MD)
    await _make_failed(v1['document_id'])
    async with AsyncSessionLocal() as s:                  # 옛 방식으로 만들어졌을 v2 failed
        s.add(Document(tenant_id=tenant_id, filename='환불정책.md', mime='text/markdown',
                       blob_path=str(blob_tmp / tenant_id / 'legacy_v2.md'), version=2,
                       status='failed', status_reason='옛 실패'))
        await s.commit()

    res = await upload(client, '환불정책.md', MD)
    assert res['version'] == 2 and res['status'] == 'pending'            # 최신(v2)을 되살림
    assert (await get_doc(v1['document_id'])).status == 'deleted'      # v1은 내려간다
    assert [(d['version'], d['status']) for d in (await list_docs(client))['items']] == [(2, 'pending')]


@pytest.mark.asyncio
async def test_정상_행이_하나라도_있으면_재사용하지_않는다(client, tenant_id, fake_queue, blob_tmp):
    """pending·ready가 있으면 기존 규칙(새 version) — failed 재사용은 정상 행이 없을 때만."""
    v1 = await upload(client, '환불정책.md', MD)                          # pending
    v2 = await upload(client, '환불정책.md', MD)
    assert v2['version'] == 2 and v2['document_id'] != v1['document_id']


async def _make_failed_with_chunks(client, doc_id: int) -> list[str]:
    """**청크가 엔진에 남은** failed — index_pending_document가 ②(색인) 뒤 ③(커밋)에서 실패가
    반복돼 굳은 상태. ready까지 올린 뒤 상태만 되돌려 재현한다. 반환: 남아 있는 청크 본문."""
    await ingest(doc_id)
    await _make_failed(doc_id)
    texts = await chunk_texts(doc_id)
    assert texts, '재현 실패 — 청크가 있어야 한다'
    return texts


@pytest.mark.asyncio
async def test_내려가는_failed의_잔여_청크는_DROP으로_지운다(client, tenant_id, fake_queue, blob_tmp):
    """리뷰 지적: 나머지 failed를 deleted로만 바꾸면 ③ 실패로 남은 청크가 정리 경로 없이 검색에
    영구 노출된다(예전엔 목록에 failed로 보여 사람이 지울 수 있었다). soft_delete와 같은 세 동작."""
    v1 = await upload(client, '환불정책.md', MD)
    await _make_failed_with_chunks(client, v1['document_id'])          # v1: 청크 남은 failed
    async with AsyncSessionLocal() as s:                              # v2 failed (옛 이력)
        s.add(Document(tenant_id=tenant_id, filename='환불정책.md', mime='text/markdown',
                       blob_path=str(blob_tmp / tenant_id / 'legacy_v2.md'), version=2,
                       status='failed', status_reason='옛 실패'))
        await s.commit()
        # 잔여 청크로 인용된 답변 캐시가 있었다고 치자 — 내려가는 v1의 것은 지워져야 한다
        await cache.save_answer(s, tenant_id, '질의 v1', '답 v1', [], [v1['document_id']])
        await s.commit()

    res = await upload(client, '환불정책.md', MD)                    # v2를 되살리고 v1은 내린다
    assert res['version'] == 2
    v1_doc = await get_doc(v1['document_id'])
    assert v1_doc.status == 'deleted' and v1_doc.status_reason == 'failed_superseded'
    async with AsyncSessionLocal() as s:                              # 세 동작 중 캐시 무효화
        remain = (await s.execute(select(AnswerCacheRow.answer)
                                  .where(AnswerCacheRow.tenant_id == tenant_id))).scalars().all()
    assert remain == []

    async with AsyncSessionLocal() as s:                              # DROP 행이 v1을 담고 있다
        rows = (await s.execute(select(SearchIndexOutbox)
                                .where(SearchIndexOutbox.tenant_id == tenant_id)
                                .where(SearchIndexOutbox.status == 'pending')
                                .where(SearchIndexOutbox.op == outbox.DROP_DOCUMENTS))).scalars().all()
    assert [r.payload['document_ids'] for r in rows] == [[v1['document_id']]]

    await ingest(v1['document_id'])                                    # DROP 처리
    assert await chunk_texts(v1['document_id']) == []               # 잔여 청크가 사라졌다


@pytest.mark.asyncio
async def test_되살린_failed의_잔여_청크는_재색인이_교체한다(client, tenant_id, fake_queue, blob_tmp):
    """target 쪽은 별도 DROP이 없어도 된다 — index_parsed_document가 색인 전에 같은 document_id
    청크를 지운다. 내용이 바뀐 blob으로 재색인하면 옛 청크가 남지 않는지 본다."""
    v1 = await upload(client, '환불정책.md', MD)
    old_texts = await _make_failed_with_chunks(client, v1['document_id'])

    new_md = '# 배송 안내\n\n## 1. 기간\n\n도서산간은 3일 더 걸린다.\n'.encode()
    await upload(client, '환불정책.md', new_md)                      # 같은 행을 되살림
    assert await ingest(v1['document_id']) == {'done': 1, 'failed': 0}

    texts = await chunk_texts(v1['document_id'])
    assert texts and all('도서산간' in t or '배송' in t for t in texts)
    assert not any(t in texts for t in old_texts)                       # 옛 청크가 남지 않았다


@pytest.mark.asyncio
async def test_deleted_이력이_섞여_있어도_failed만_살아있으면_되살린다(client, tenant_id, fake_queue, blob_tmp):
    """v1 deleted + v2 failed — 개발계 실측(id 7503)과 같은 이력. deleted는 '살아 있는 행'이 아니므로
    v2를 되살린다(v3를 만들지 않는다). v1은 deleted 그대로."""
    v1 = await upload(client, '환불정책.md', MD)
    await ingest(v1['document_id'])                                    # v1 ready
    v2 = await upload(client, '환불정책.md', MD)
    await ingest(v2['document_id'])                                    # v2 ready, v1 supersede → deleted
    assert (await get_doc(v1['document_id'])).status == 'deleted'
    await _make_failed(v2['document_id'])                              # v2 failed

    res = await upload(client, '환불정책.md', MD)
    assert res['document_id'] == v2['document_id'] and res['version'] == 2 and res['status'] == 'pending'
    assert (await get_doc(v1['document_id'])).status == 'deleted'    # 건드리지 않는다
    async with AsyncSessionLocal() as s:
        n = (await s.execute(select(func.count()).select_from(Document)
                             .where(Document.tenant_id == tenant_id))).scalar()
    assert n == 2                                                      # v3가 생기지 않았다


@pytest.mark.asyncio
async def test_되살릴_때_폴더와_설명은_유지하고_보내면_갱신한다(client, tenant_id, fake_queue, blob_tmp):
    """#165 규칙이 재사용 경로에서도 그대로 — 미전송이면 그 행의 값, 보냈으면 그 값(null이면 미분류)."""
    fa = await make_folder(client, '규정집')
    fb = await make_folder(client, '대외비')
    doc = await upload(client, '환불정책.md', MD,
                        extra_parts=doc_data(folder_id=fa, description='표 설명 1'))
    await _make_failed(doc['document_id'])

    kept = await upload(client, '환불정책.md', MD)                    # 아무것도 안 보냄 → 유지
    assert kept['document_id'] == doc['document_id']
    assert kept['folder_id'] == fa and (await get_doc(doc['document_id'])).description == '표 설명 1'

    await _make_failed(doc['document_id'])
    moved = await upload(client, '환불정책.md', MD,
                          extra_parts=doc_data(folder_id=fb, description='표 설명 2'))   # 보냄 → 갱신
    assert moved['folder_id'] == fb and (await get_doc(doc['document_id'])).description == '표 설명 2'

    await _make_failed(doc['document_id'])
    unfiled = await upload(client, '환불정책.md', MD, extra_parts=doc_data(folder_id=None))   # null → 미분류
    assert unfiled['folder_id'] is None


@pytest.mark.asyncio
async def test_ready와_failed가_공존하면_기존_규칙으로_새_version(client, tenant_id, fake_queue, blob_tmp):
    """옛 코드가 남긴 'ready v1 + failed v2' 이력(개발계 실측 0건). 살아 있는 행이 있으니 재사용하지 않고
    v3를 만든다 — failed v2는 그대로 남고, 정리는 목록 필터(#176)+삭제(#174)로 한다(스콥 밖 동작을 고정)."""
    v1 = await upload(client, '환불정책.md', MD)
    await ingest(v1['document_id'])                                    # v1 ready
    async with AsyncSessionLocal() as s:                              # v2 failed (직접 심음)
        s.add(Document(tenant_id=tenant_id, filename='환불정책.md', mime='text/markdown',
                       blob_path=str(blob_tmp / tenant_id / 'legacy_v2.md'), version=2,
                       status='failed', status_reason='옛 실패'))
        await s.commit()

    res = await upload(client, '환불정책.md', MD)
    assert res['version'] == 3                                         # 재사용 아님
    statuses = sorted((d.version, d.status) for d in (await _all_docs(tenant_id)))
    assert statuses == [(1, 'ready'), (2, 'failed'), (3, 'pending')]   # failed v2는 그대로


async def _all_docs(tenant_id: str) -> list[Document]:
    async with AsyncSessionLocal() as s:
        return list((await s.execute(select(Document).where(Document.tenant_id == tenant_id))).scalars().all())
