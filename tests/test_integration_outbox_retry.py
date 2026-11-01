"""outbox 워커 재시도 계약 — 일시 실패는 attempts·백오프로 미뤄 다시 하고, 결정적 실패는 1회로 확정한다.

1. 색인 일시 실패: attempts·last_error 남기고 다음 회차에 성공
2. 백오프: 일시 실패는 next_attempt_at을 미루고 cron 선정에서 빠진다, 결정적 실패는 1회 확정
3. failed 확정 시 잔여 청크 DROP — DROP 가드는 되살아난 문서를 건너뛴다
"""

from datetime import timedelta

import pytest
from sqlalchemy import func, select, update

from database import AsyncSessionLocal
from rag import outbox
from rag.models import SearchIndexOutbox
from tests.conftest import indexed_chunk_texts, ingest
from tests.helpers_documents import MD, get_doc, upload
from tests.helpers_index_sync import boom_mark_done, drop_rows, latest_index_row, outbox_rows


@pytest.mark.asyncio
async def test_색인_일시_실패는_attempts를_남기고_다음_회차에_성공한다(
    client, tenant_id, fake_queue, blob_tmp, fake_embed, monkeypatch
):
    """OpenSearch가 잠깐 죽은 상황 — 행은 pending으로 남고 attempts·last_error가 쌓이며, 문서는
    failed가 아니다. 복구되면 다음 회차가 같은 행을 끝낸다(at-least-once)."""
    body = await upload(client, '환불정책.md', MD)
    doc_id = body['document_id']

    import rag.ingestion as rd

    real = rd.os_index.index_parsed_document
    calls = {'n': 0}

    async def _flaky(**kw):
        calls['n'] += 1
        if calls['n'] == 1:
            raise ConnectionError('Cannot connect to host localhost:9200')
        return await real(**kw)

    monkeypatch.setattr(rd.os_index, 'index_parsed_document', _flaky)
    assert await ingest(doc_id) == {'done': 0, 'failed': 1}
    doc = await get_doc(doc_id)
    assert doc.status == 'pending'  # failed 아님 — 일시 실패라 MAX_ATTEMPTS 전엔 백오프
    ((op, status, attempts, err),) = await outbox_rows(doc_id)
    assert (op, status, attempts) == ('index_document', 'pending', 1)
    assert 'Cannot connect' in err

    assert await ingest(doc_id) == {'done': 1, 'failed': 0}  # "복구 후" 회차
    doc = await get_doc(doc_id)
    assert doc.status == 'ready' and doc.is_active is True
    ((op, status, attempts, _),) = await outbox_rows(doc_id)
    assert (status, attempts) == ('done', 1)  # 시도 이력은 남는다
    assert await indexed_chunk_texts(doc_id)


# ── 백오프 ─────────────────────────────────────────────────────────────


async def _due_ids() -> set[int]:
    """운영 cron이 이번 회차에 집을 행 id — row_ids 없이 due_pending_stmt를 그대로 실행한다.
    공유 DB라 남의 행도 섞이므로 포함 여부만 단언한다."""
    async with AsyncSessionLocal() as s:
        return {r.id for r in (await s.execute(outbox.due_pending_stmt(limit=1000))).all()}


@pytest.mark.asyncio
async def test_일시_실패는_다음_시도를_미루고_cron_선정에서_빠진다(
    client, tenant_id, fake_queue, blob_tmp, fake_embed, monkeypatch
):
    """엔진 연결 실패 → attempts=1, next_attempt_at≈now+1분, 전체 drain의 선정 대상에서 빠진다.
    시각을 과거로 돌리면 다시 잡힌다. row_ids 지정(ingest)은 백오프를 무시한다(헬퍼 계약)."""
    doc_id = (await upload(client, '환불정책.md', MD))['document_id']
    import rag.ingestion as rd

    async def _down(**kw):
        raise ConnectionError('Cannot connect to host localhost:9200')

    monkeypatch.setattr(rd.os_index, 'index_parsed_document', _down)

    assert await ingest(doc_id) == {'done': 0, 'failed': 1}
    row = await latest_index_row(doc_id)
    assert (row.status, row.attempts) == ('pending', 1)
    async with AsyncSessionLocal() as s:
        now = (await s.execute(select(func.now()))).scalar()
    assert timedelta(seconds=30) < row.next_attempt_at - now <= timedelta(minutes=1)  # backoff_delay(1)=1분
    assert row.id not in await _due_ids()  # cron은 건너뛴다

    assert await ingest(doc_id) == {'done': 0, 'failed': 1}  # row_ids 지정 → 백오프 무시, 즉시 재시도
    row = await latest_index_row(doc_id)
    assert row.attempts == 2
    assert row.next_attempt_at - now > timedelta(minutes=1, seconds=30)  # backoff_delay(2)=2분으로 늘었다

    async with AsyncSessionLocal() as s:  # 시각이 지나면 다시 선정된다
        await s.execute(
            update(SearchIndexOutbox)
            .where(SearchIndexOutbox.id == row.id)
            .values(next_attempt_at=func.now() - timedelta(seconds=1))
        )
        await s.commit()
    assert row.id in await _due_ids()
    assert (await get_doc(doc_id)).status == 'pending'  # 문서는 failed가 아니다


@pytest.mark.asyncio
async def test_일시_실패가_MAX_ATTEMPTS에_닿으면_확정되고_훅이_불린다(
    client, tenant_id, fake_queue, blob_tmp, fake_embed, monkeypatch
):
    doc_id = (await upload(client, '환불정책.md', MD))['document_id']
    import rag.ingestion as rd

    async def _down(**kw):
        raise ConnectionError('Cannot connect')

    monkeypatch.setattr(rd.os_index, 'index_parsed_document', _down)
    monkeypatch.setattr(outbox, 'MAX_ATTEMPTS', 2)
    hook_calls = []
    monkeypatch.setattr(outbox, '_on_failed', lambda op, row_id, err: hook_calls.append((op, row_id, err)))
    before = outbox.SEARCH_INDEX_FAILED_TOTAL.labels(op=outbox.INDEX_DOCUMENT)._value.get()

    assert await ingest(doc_id) == {'done': 0, 'failed': 1}
    assert hook_calls == []  # 1회차 — 아직 백오프
    assert await ingest(doc_id) == {'done': 0, 'failed': 1}
    row = await latest_index_row(doc_id)
    assert (row.status, row.attempts) == ('failed', 2)
    assert (await get_doc(doc_id)).status == 'failed'
    assert hook_calls == [(outbox.INDEX_DOCUMENT, row.id, 'Cannot connect')]
    # 훅을 대역으로 바꿨으니 카운터는 안 올랐어야 한다 — 훅이 카운터의 유일한 증가 지점임을 같이 확인
    assert outbox.SEARCH_INDEX_FAILED_TOTAL.labels(op=outbox.INDEX_DOCUMENT)._value.get() == before


@pytest.mark.asyncio
async def test_결정적_실패는_META_행도_1회로_확정하고_카운터가_오른다(
    client, tenant_id, fake_queue, blob_tmp, fake_embed, monkeypatch
):
    """INDEX_DOCUMENT 아닌 op에서도 분류가 같다 — META_DOCUMENTS 핸들러가 프로그래밍 오류(결정적)로
    죽으면 재시도 없이 failed, 확정 카운터 +1(진짜 _on_failed 경로)."""
    doc_id = (await upload(client, '환불정책.md', MD))['document_id']
    assert await ingest(doc_id) == {'done': 1, 'failed': 0}
    res = await client.patch(f'/kms/documents/{doc_id}', json={'is_searchable': False})
    assert res.status_code == 200, res.text

    async def _bug(session, ids):
        raise KeyError('folder_id')

    monkeypatch.setattr(outbox.os_index, 'sync_meta_documents_now', _bug)
    before = outbox.SEARCH_INDEX_FAILED_TOTAL.labels(op=outbox.META_DOCUMENTS)._value.get()

    assert await ingest(doc_id) == {'done': 0, 'failed': 1}
    rows = [r for r in await outbox_rows(doc_id) if r[0] == outbox.META_DOCUMENTS]
    assert rows == [(outbox.META_DOCUMENTS, 'failed', 1, "'folder_id'")]
    assert outbox.SEARCH_INDEX_FAILED_TOTAL.labels(op=outbox.META_DOCUMENTS)._value.get() == before + 1


# ── failed 확정 시 잔여 청크 DROP ──────────────────────────────────────


@pytest.mark.asyncio
async def test_커밋_단계만_실패해_failed가_된_문서의_청크는_다음_회차에_지워진다(
    client, tenant_id, fake_queue, blob_tmp, fake_embed, monkeypatch
):
    doc_id = (await upload(client, '환불정책.md', MD))['document_id']
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(outbox, 'mark_done', boom_mark_done)
        assert await ingest(doc_id) == {'done': 0, 'failed': 1}
    doc = await get_doc(doc_id)
    assert doc.status == 'failed' and '③ 커밋 직전 폭발' in doc.status_reason
    assert len(await indexed_chunk_texts(doc_id)) > 0  # 바로 그 불일치 — failed인데 청크가 있다
    (drop,) = await drop_rows(doc_id)
    assert drop.status == 'pending' and drop.payload == {'document_ids': [doc_id]}

    assert await ingest(doc_id) == {'done': 1, 'failed': 0}  # DROP 행 — 엔진·mark_done 정상
    assert await indexed_chunk_texts(doc_id) == []  # failed = 청크 0, 이제 참
    assert (await drop_rows(doc_id))[0].status == 'done'


@pytest.mark.asyncio
async def test_DROP은_되살아난_문서를_건너뛴다(client, tenant_id, fake_queue, blob_tmp, fake_embed, monkeypatch):
    """failed 확정의 DROP이 미뤄진 사이 재업로드(failed 행 재사용)가 먼저 색인을 끝냈다 — 늦게 도는
    DROP이 새 청크를 지우면 ready인데 검색에 없는 문서가 된다. 살아 있는 문서는 건너뛰어야 한다."""
    doc_id = (await upload(client, '환불정책.md', MD))['document_id']
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(outbox, 'mark_done', boom_mark_done)
        assert await ingest(doc_id) == {'done': 0, 'failed': 1}
    (drop,) = await drop_rows(doc_id)

    again = await upload(client, '환불정책.md', MD)  # failed 행 재사용 — 같은 id, pending
    assert again['document_id'] == doc_id and again['version'] == 1
    index_row = await latest_index_row(doc_id)
    assert index_row.id > drop.id and index_row.status == 'pending'

    # 순서를 뒤집어 재현: INDEX 먼저(ready), 그 뒤 미뤄졌던 DROP
    assert await outbox.drain_once(row_ids=[index_row.id]) == {'done': 1, 'failed': 0}
    assert (await get_doc(doc_id)).status == 'ready'
    texts = await indexed_chunk_texts(doc_id)
    assert len(texts) > 0
    assert await outbox.drain_once(row_ids=[drop.id]) == {'done': 1, 'failed': 0}
    assert await indexed_chunk_texts(doc_id) == texts  # 건너뛰었다 — 청크 그대로
    assert (await drop_rows(doc_id))[0].status == 'done'
