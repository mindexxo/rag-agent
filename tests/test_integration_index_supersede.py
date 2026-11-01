"""재업로드 supersede 계약 — 구버전 정리는 신버전 커밋 뒤 outbox DROP 행으로만 하고, 처리 순서에 무관하게 신버전이 이긴다.

1. 처리 순서 무관: 백오프로 미뤄진 v1 행보다 v2 행이 먼저 끝나도 신버전이 이긴다
2. supersede 삭제는 ③ 커밋의 DROP 행으로: ③이 실패하면 구버전은 PG·엔진 모두 그대로(검색됨)
"""

import pytest

from rag import outbox
from tests.conftest import indexed_chunk_texts, ingest
from tests.helpers_documents import MD, get_doc, upload
from tests.helpers_index_sync import boom_mark_done, cited_ids, drop_rows, latest_index_row

# ── 처리 순서 무관 ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_백오프로_미뤄진_v1보다_v2가_먼저_끝나도_신버전이_이긴다(
    client, tenant_id, fake_queue, blob_tmp, fake_embed, monkeypatch
):
    """v1 INDEX가 일시 실패로 물러난 사이 v2가 올라와 먼저 색인된다. 그 뒤 v1 차례가 와도 v2를
    supersede하면 안 된다 — v2 ready·active·검색됨, v1 deleted·청크 0."""
    v1 = (await upload(client, '환불정책.md', MD))['document_id']
    import rag.ingestion as rd

    async def _down(**kw):
        raise ConnectionError('Cannot connect')

    with pytest.MonkeyPatch.context() as mp:  # 이것만 걷는다 — fake_embed는 유지
        mp.setattr(rd.os_index, 'index_parsed_document', _down)
        assert await ingest(v1) == {'done': 0, 'failed': 1}  # v1 행: pending, 백오프

    v2 = (await upload(client, '환불정책.md', MD.replace('14일'.encode(), '30일'.encode())))['document_id']
    assert v2 != v1
    v1_row, v2_row = await latest_index_row(v1), await latest_index_row(v2)
    assert v1_row.id < v2_row.id and v1_row.status == v2_row.status == 'pending'

    # 역순 재현 — 미뤄진 v1을 건너뛰고 v2만 처리
    assert await outbox.drain_once(row_ids=[v2_row.id]) == {'done': 1, 'failed': 0}
    d1, d2 = await get_doc(v1), await get_doc(v2)
    assert (d2.status, d2.is_active) == ('ready', True)
    assert (d1.status, d1.is_active) == ('deleted', False)  # pending 구버전도 supersede된다
    assert any('30일' in t for t in await indexed_chunk_texts(v2))

    # v1 차례 — ①의 "pending 아님 → done" 분기. v2는 그대로.
    assert await outbox.drain_once(row_ids=[v1_row.id]) == {'done': 1, 'failed': 0}
    assert (await latest_index_row(v1)).status == 'done'
    d1, d2 = await get_doc(v1), await get_doc(v2)
    assert (d2.status, d2.is_active) == ('ready', True)
    assert d1.status == 'deleted'
    assert await ingest(v1) == {'done': 1, 'failed': 0}  # v2의 ③이 남긴 DROP[v1] 행
    assert await indexed_chunk_texts(v1) == []
    assert (await cited_ids(tenant_id)) & {v1, v2} == {v2}


# ── supersede 구버전 삭제는 커밋 뒤 outbox로 ────────────────────────────


@pytest.mark.asyncio
async def test_재업로드_커밋이_실패하면_구버전은_PG_엔진_모두_그대로다(
    client, tenant_id, fake_queue, blob_tmp, fake_embed, monkeypatch
):
    """③ 커밋만 죽는다(②는 끝나 v2 청크가 엔진에 있다). 구버전 v1은 ready·active·청크 그대로·검색됨이어야
    한다 — 아니면 "목록엔 정상인데 검색엔 없는" v1이 영구로 남는다."""
    v1 = (await upload(client, '환불정책.md', MD))['document_id']
    assert await ingest(v1) == {'done': 1, 'failed': 0}
    v1_texts = await indexed_chunk_texts(v1)
    assert v1_texts

    v2 = (await upload(client, '환불정책.md', MD.replace('14일'.encode(), '30일'.encode())))['document_id']
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(outbox, 'mark_done', boom_mark_done)
        assert await ingest(v2) == {'done': 0, 'failed': 1}  # 결정적 → 1회로 v2 failed

    d1, d2 = await get_doc(v1), await get_doc(v2)
    assert (d1.status, d1.is_active) == ('ready', True)  # 구버전 무사
    assert d2.status == 'failed'
    assert await indexed_chunk_texts(v1) == v1_texts  # 엔진도 그대로
    assert (await cited_ids(tenant_id)) & {v1, v2} >= {v1}  # 검색에 여전히 잡힌다
    drops = await drop_rows(v1)
    assert drops == []  # DROP[v1]은 ③과 함께 롤백됐다
    assert len(await drop_rows(v2)) == 1  # v2 잔여 청크 DROP은 failed 확정이 남김

    assert await ingest(v2) == {'done': 1, 'failed': 0}  # 그 DROP — v2 청크 정리
    assert await indexed_chunk_texts(v2) == []
    assert await indexed_chunk_texts(v1) == v1_texts


@pytest.mark.asyncio
async def test_재업로드_성공은_구버전_DROP_행을_같은_커밋에_남긴다(client, tenant_id, fake_queue, blob_tmp, fake_embed):
    v1 = (await upload(client, '환불정책.md', MD))['document_id']
    assert await ingest(v1) == {'done': 1, 'failed': 0}
    v2 = (await upload(client, '환불정책.md', MD.replace('14일'.encode(), '30일'.encode())))['document_id']
    assert await ingest(v2) == {'done': 1, 'failed': 0}

    (drop,) = await drop_rows(v1)
    assert drop.status == 'pending' and drop.payload == {'document_ids': [v1]}
    assert (await get_doc(v1)).status == 'deleted'
    assert await indexed_chunk_texts(v1)  # 다음 회차까지 공존 — 제품 결정 범위
    assert await ingest(v1) == {'done': 1, 'failed': 0}
    assert await indexed_chunk_texts(v1) == []
    assert (await cited_ids(tenant_id)) & {v1, v2} == {v2}
