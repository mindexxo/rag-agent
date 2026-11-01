"""색인 대기열 내부 경계 — 엔진 다운·회차 겹침·워커 취소·알 수 없는 연산·확정 실패·색인 중 토글·일시 장애 재시도·회차 설정·재동기화 대조 모드.

대기열 행은 `ingest`(그 문서 행만 drain)나 `outbox.drain_once(row_ids=…)`로 처리한다. 장애는 모듈 속성 패치로 주입한다.
"""

import asyncio
from datetime import timedelta

import pytest
from sqlalchemy import func, select, update

from database import AsyncSessionLocal
from rag import ingestion, os_reconcile, outbox
from rag.models import SearchIndexOutbox
from rag.search import os_client, os_index
from tests.conftest import indexed_chunk_texts, ingest
from tests.helpers_documents import MD, get_doc, upload


async def _rows(doc_id: int) -> list[SearchIndexOutbox]:
    async with AsyncSessionLocal() as s:
        rows = (await s.execute(select(SearchIndexOutbox).order_by(SearchIndexOutbox.id))).scalars().all()
    return [
        r for r in rows if r.payload.get('document_id') == doc_id or doc_id in (r.payload.get('document_ids') or [])
    ]


async def _make_due(doc_id: int) -> None:
    """백오프 대기 중인 행을 지금 처리할 수 있게 앞당긴다."""
    async with AsyncSessionLocal() as s:
        for r in await _rows(doc_id):
            await s.execute(
                update(SearchIndexOutbox)
                .where(SearchIndexOutbox.id == r.id)
                .values(next_attempt_at=func.now() - timedelta(seconds=1))
            )
        await s.commit()


@pytest.mark.asyncio
async def test_엔진이_죽으면_drain은_건너뛰고_업로드_삭제는_되며_복구_뒤_반영된다(
    client, tenant_id, fake_queue, blob_tmp, fake_embed, monkeypatch
):
    """OB-18 — 건너뛴 회차는 attempts를 늘리지 않는다."""

    async def _down() -> bool:
        return False

    monkeypatch.setattr(os_client, 'ensure_index_soft', _down)
    keep = (await upload(client, '유지.md', MD))['document_id']
    gone = (await upload(client, '삭제.md', MD))['document_id']
    assert (await client.delete(f'/kms/documents/{gone}')).status_code == 204
    assert await ingest(keep) == {'done': 0, 'failed': 0, 'skipped': True}
    assert [(r.status, r.attempts) for r in await _rows(keep)] == [('pending', 0)]

    monkeypatch.undo()
    assert await ingest(keep) == {'done': 1, 'failed': 0}
    assert (await get_doc(keep)).status == 'ready'
    assert await ingest(gone) == {'done': 2, 'failed': 0}  # 색인 행은 닫히고 DROP 행이 처리된다
    assert await indexed_chunk_texts(gone) == []


@pytest.mark.asyncio
async def test_이전_회차가_돌고_있으면_즉시_빠진다(client, tenant_id, fake_queue, blob_tmp):
    """OB-17."""
    doc_id = (await upload(client, '환불정책.md', MD))['document_id']
    async with outbox._drain_lock:
        assert await ingest(doc_id) == {'done': 0, 'failed': 0, 'skipped': True}
    assert [r.status for r in await _rows(doc_id)] == ['pending']


@pytest.mark.asyncio
async def test_워커가_취소되면_행은_pending으로_남고_attempts가_늘지_않는다(
    client, tenant_id, fake_queue, blob_tmp, fake_embed, monkeypatch
):
    """OB-15 — 실제 프로세스 kill은 수동. 여기서는 처리 도중 CancelledError."""
    doc_id = (await upload(client, '환불정책.md', MD))['document_id']

    async def _cancelled(*a: object, **kw: object) -> None:
        raise asyncio.CancelledError

    monkeypatch.setattr(ingestion, 'index_pending_document', _cancelled)
    with pytest.raises(asyncio.CancelledError):
        await ingest(doc_id)
    assert [(r.status, r.attempts) for r in await _rows(doc_id)] == [('pending', 0)]

    monkeypatch.undo()
    assert await ingest(doc_id) == {'done': 1, 'failed': 0}
    assert (await get_doc(doc_id)).status == 'ready'


@pytest.mark.asyncio
async def test_알_수_없는_연산은_한_번에_failed(tenant_id):
    """OB-10 — 결정적 실패라 재시도하지 않는다."""
    async with AsyncSessionLocal() as s:
        row = SearchIndexOutbox(tenant_id=tenant_id, op='bogus_op', payload={}, status='pending')
        s.add(row)
        await s.commit()
        row_id = row.id
    assert await outbox.drain_once(row_ids=[row_id]) == {'done': 0, 'failed': 1}
    async with AsyncSessionLocal() as s:
        saved = await s.get(SearchIndexOutbox, row_id)
    assert (saved.status, saved.attempts) == ('failed', 1) and '알 수 없는 outbox op' in saved.last_error


@pytest.mark.asyncio
async def test_확정_실패_때_문서가_이미_pending이_아니면_상태는_두고_DROP만_건다(
    client, tenant_id, fake_queue, blob_tmp, monkeypatch
):
    """OB-10 — 처리 도중 삭제된 문서를 failed로 되돌리지 않는다. 잔여 청크용 DROP 행은 남긴다."""
    doc_id = (await upload(client, '환불정책.md', MD))['document_id']

    async def _delete_then_fail(document_id: int, **kw: object) -> None:
        assert (await client.delete(f'/kms/documents/{document_id}')).status_code == 204
        raise ValueError('파싱 실패')

    monkeypatch.setattr(ingestion, 'index_pending_document', _delete_then_fail)
    assert (await ingest(doc_id))['failed'] == 1
    assert (await get_doc(doc_id)).status == 'deleted'
    ops = [(r.op, r.status) for r in await _rows(doc_id)]
    assert ops[0] == ('index_document', 'failed') and ops.count(('drop_documents', 'pending')) == 2


@pytest.mark.asyncio
async def test_색인_도중_검색을_끄면_완료_때_META_행을_얹는다(
    client, tenant_id, fake_queue, blob_tmp, fake_embed, monkeypatch
):
    """OB-30 — ① 시점 값으로 넣은 메타가 낡았으면 ready 커밋에 META 행을 더한다."""
    doc_id = (await upload(client, '환불정책.md', MD))['document_id']
    real = os_index.index_parsed_document

    async def _index_then_toggle(**kw: object) -> None:
        await real(**kw)
        assert (await client.patch(f'/kms/documents/{doc_id}', json={'is_searchable': False})).status_code == 200

    monkeypatch.setattr(os_index, 'index_parsed_document', _index_then_toggle)
    assert await ingest(doc_id) == {'done': 1, 'failed': 0}
    assert (await get_doc(doc_id)).status == 'ready'
    metas = [r for r in await _rows(doc_id) if r.op == 'meta_documents']
    assert len(metas) == 2  # 토글이 만든 행 + 완료 시점 보정 행


@pytest.mark.asyncio
async def test_임베딩_서버가_죽으면_백오프로_재시도하고_복구되면_ready(
    client, tenant_id, fake_queue, blob_tmp, fake_embed, monkeypatch
):
    """ST-13 — 연결 실패는 일시 장애다. 문서는 pending, 행은 attempts 1·다음 시도 시각이 잡힌다."""
    doc_id = (await upload(client, '환불정책.md', MD))['document_id']
    real = ingestion.embed_texts

    async def _down(*a: object, **kw: object) -> list:
        raise ConnectionError('embedding server down')

    monkeypatch.setattr(ingestion, 'embed_texts', _down)
    assert await ingest(doc_id) == {'done': 0, 'failed': 1}
    row = (await _rows(doc_id))[0]
    assert (row.status, row.attempts) == ('pending', 1) and row.next_attempt_at is not None
    assert (await get_doc(doc_id)).status == 'pending'

    monkeypatch.setattr(ingestion, 'embed_texts', real)
    await _make_due(doc_id)
    assert await ingest(doc_id) == {'done': 1, 'failed': 0}
    assert (await get_doc(doc_id)).status == 'ready'


def test_워커_cron과_회차_예산():
    """OB-01·OB-12·CA-21 — drain·스윕은 매분, 캐시 청소는 19:30 UTC(KST 04:30), 회차 예산 = 변환 마감 + 꼬리 + 여유."""
    from config import settings
    from rag.worker import WorkerSettings

    jobs = {j.coroutine.__name__: j for j in WorkerSettings.cron_jobs}
    assert jobs['drain_search_index_outbox'].minute == set(range(60))
    assert jobs['sweep_stale_generating'].minute == set(range(60))
    assert (jobs['sweep_stale_cache'].hour, jobs['sweep_stale_cache'].minute) == (19, 30)
    assert outbox.BATCH == 50
    assert WorkerSettings.job_timeout == outbox.DRAIN_ROUND_SECONDS
    assert outbox.DRAIN_ROUND_SECONDS == int(settings.docling_document_timeout_seconds) + 300 + 180
    assert outbox._fits_round(0) and not outbox._fits_round(outbox.ROUND_SLACK_SECONDS + 1)


@pytest.mark.asyncio
async def test_재동기화_대조_모드는_출력만_하고_쓰지_않는다(
    client, tenant_id, fake_queue, blob_tmp, fake_embed, monkeypatch, capsys
):
    """RC-01 — 기본 실행(--apply 없음)은 누락·잉여 건수와 id를 출력하고 PG·색인을 바꾸지 않는다."""
    doc_id = (await upload(client, '환불정책.md', MD))['document_id']
    await ingest(doc_id)
    await os_index._delete_by_terms('document_id', [doc_id])  # 누락 재현

    monkeypatch.setattr('sys.argv', ['os_reconcile', '--tenant', tenant_id])
    await os_reconcile.main()
    out = capsys.readouterr().out
    assert f'문서 누락(ready인데 OS에 없음) 1건: [{doc_id}]' in out
    assert (await get_doc(doc_id)).status == 'ready'
    assert [r.op for r in await _rows(doc_id)] == ['index_document']
