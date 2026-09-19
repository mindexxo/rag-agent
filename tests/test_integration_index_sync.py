"""색인 동기화 계약 (#139) — E2E 수동 케이스 중 코드로 덮을 수 있는 것을 테스트로 옮긴 것.

라우터는 대기열 행만 남기고 색인은 워커가 한다. 여기서는 워커 대신 `ingest`/`sync_faq`(그 문서·FAQ의
행만 drain)로 같은 경로를 태운다. 지연 창("drain 전엔 아직 보인다")은 제품 결정이므로 그대로 단언한다.

  1. 인용 메타(파일명·버전·페이지·폴더명·폴더 설명)는 엔진 _source에서 온다      (E2E #1)
  2. 문서 검색토글: META 부분 갱신 — 제외/복귀, 재색인 없음(indexed_at 불변)        (E2E #5)
  3. 색인 일시 실패: attempts·last_error 남기고 다음 회차에 성공                     (E2E #7)
  4. reconcile(문서): 색인에서 사라진 ready 문서는 재등재, PG에 없는 문서 청크는 삭제  (E2E #9)
  5. reconcile(FAQ): 같은 계약                                                        (배포 체크리스트 D)
  6. 백오프(#185): 일시 실패는 next_attempt_at을 미루고 cron 선정에서 빠진다, 결정적 실패는 1회 확정,
     확정 시 _on_failed 훅·카운터
  7. failed 확정 시 잔여 청크 DROP(#184): ③ 커밋만 실패한 문서의 청크가 다음 회차에 0, DROP 가드는
     되살아난 문서를 건너뛴다
"""
import json
from datetime import timedelta

import pytest
from sqlalchemy import func, select, update

from database import AsyncSessionLocal
from rag import os_index, os_reconcile, outbox
from rag.models import Document, SearchIndexOutbox
from rag.retriever import retrieve_candidates
from tests.conftest import faq_doc, indexed_chunk_texts, ingest, sync_faq

MD = '# 환불 정책\n\n## 1. 기간\n\n단순변심 반품은 14일 이내 신청한다.\n'.encode()


async def _upload(client, filename, content=MD):
    res = await client.post('/kms/documents', files={'file': (filename, content, 'text/markdown')})
    assert res.status_code == 200, res.text
    return res.json()


async def _doc(doc_id):
    async with AsyncSessionLocal() as s:
        return await s.get(Document, doc_id)


async def _rows(doc_id):
    async with AsyncSessionLocal() as s:
        rows = (await s.execute(select(SearchIndexOutbox).order_by(SearchIndexOutbox.id))).scalars().all()
    return [(r.op, r.status, r.attempts, r.last_error) for r in rows
            if r.payload.get('document_id') == doc_id or doc_id in (r.payload.get('document_ids') or [])]


async def _cited_ids(tenant_id, query='반품 기간') -> set[int]:
    async with AsyncSessionLocal() as s:
        cands = await retrieve_candidates(s, tenant_id, query, top_n=20)
    return {c.document_id for c in cands.chunks}


@pytest.mark.asyncio
async def test_인용_메타는_엔진_source에서_온다(client, tenant_id, fake_queue, blob_tmp, fake_embed):
    """PG를 되묻지 않으므로 RetrievedChunk의 모든 필드가 색인 문서에서 채워져야 한다 — 하나라도
    비면 인용 표시·리랭커 입력(폴더 설명)이 조용히 빈다."""
    folder = (await client.post('/kms/folders', json={'name': '정책', 'description': '환불·반품 규정 모음'})).json()
    body = await _upload(client, '환불정책.md')
    assert (await client.patch(f"/kms/documents/{body['document_id']}",
                               json={'folder_id': folder['id']})).status_code == 200
    await ingest(body['document_id'])           # 폴더 이동 META 행도 함께 처리된다

    async with AsyncSessionLocal() as s:
        cands = await retrieve_candidates(s, tenant_id, '반품 기간', top_n=20)
    mine = [c for c in cands.chunks if c.document_id == body['document_id']]
    assert mine, '방금 색인한 문서가 후보에 없다'
    c = mine[0]
    assert c.filename == '환불정책.md' and c.version == 1
    assert c.folder_name == '정책' and c.folder_description == '환불·반품 규정 모음'
    assert c.heading_path and c.page is None and c.text     # md: 헤딩 있음·페이지 없음


@pytest.mark.asyncio
async def test_문서_검색토글은_META_부분갱신으로_제외_복귀한다(client, tenant_id, fake_queue, blob_tmp, fake_embed):
    body = await _upload(client, '환불정책.md')
    await ingest(body['document_id'])
    doc_id = body['document_id']
    indexed_at = (await _doc(doc_id)).indexed_at
    assert doc_id in await _cited_ids(tenant_id)

    assert (await client.patch(f'/kms/documents/{doc_id}', json={'is_searchable': False})).status_code == 200
    assert doc_id in await _cited_ids(tenant_id)            # drain 전 — 가이드대로 아직 보인다
    r = await ingest(doc_id)                                # META_DOCUMENTS 행 하나
    assert r == {'done': 1, 'failed': 0}
    assert doc_id not in await _cited_ids(tenant_id)        # 제외
    assert (await _doc(doc_id)).indexed_at == indexed_at    # 재색인 없음 — 플래그만 바뀜
    assert [op for op, *_ in await _rows(doc_id)] == ['index_document', 'meta_documents']

    assert (await client.patch(f'/kms/documents/{doc_id}', json={'is_searchable': True})).status_code == 200
    await ingest(doc_id)
    assert doc_id in await _cited_ids(tenant_id)            # 복귀
    assert (await _doc(doc_id)).indexed_at == indexed_at


@pytest.mark.asyncio
async def test_색인_일시_실패는_attempts를_남기고_다음_회차에_성공한다(client, tenant_id, fake_queue, blob_tmp,
                                                                fake_embed, monkeypatch):
    """OpenSearch가 잠깐 죽은 상황 — 행은 pending으로 남고 attempts·last_error가 쌓이며, 문서는
    failed가 아니다. 복구되면 다음 회차가 같은 행을 끝낸다(at-least-once)."""
    body = await _upload(client, '환불정책.md')
    doc_id = body['document_id']

    import rag.documents as rd
    real = rd.os_index.index_parsed_document
    calls = {'n': 0}

    async def _flaky(**kw):
        calls['n'] += 1
        if calls['n'] == 1:
            raise ConnectionError('Cannot connect to host localhost:9200')
        return await real(**kw)

    monkeypatch.setattr(rd.os_index, 'index_parsed_document', _flaky)
    assert await ingest(doc_id) == {'done': 0, 'failed': 1}
    doc = await _doc(doc_id)
    assert doc.status == 'pending'                          # failed 아님 — 일시 실패라 MAX_ATTEMPTS 전엔 백오프(#185)
    (op, status, attempts, err), = await _rows(doc_id)
    assert (op, status, attempts) == ('index_document', 'pending', 1)
    assert 'Cannot connect' in err

    assert await ingest(doc_id) == {'done': 1, 'failed': 0}  # "복구 후" 회차
    doc = await _doc(doc_id)
    assert doc.status == 'ready' and doc.is_active is True
    (op, status, attempts, _), = await _rows(doc_id)
    assert (status, attempts) == ('done', 1)                 # 시도 이력은 남는다
    assert await indexed_chunk_texts(doc_id)


@pytest.mark.asyncio
async def test_reconcile은_사라진_ready_문서를_재등재하고_유령_청크를_지운다(client, tenant_id, fake_queue,
                                                                     blob_tmp, fake_embed):
    body = await _upload(client, '환불정책.md')
    doc_id = body['document_id']
    await ingest(doc_id)
    # 누락 재현: 색인에서만 지운다(스냅샷 복구 뒤 유실 등). PG는 ready 그대로.
    await os_index._delete_by_terms('document_id', [doc_id])
    assert await indexed_chunk_texts(doc_id) == []
    # 잉여 재현: PG에 없는 document_id의 청크를 이 테넌트로 넣는다.
    ghost = os_index.build_doc(
        os_index._ParsedChunk(cid=os_index.chunk_os_id(document_id=999_999_001, chunk_index=0),
                                tenant_id=tenant_id, document_id=999_999_001, faq_id=None,
                                text='유령', heading_path=[], page=None, meta={}, dense=[0.01] * 1024),
        '유령.md', 1, searchable=True)
    await os_index.bulk_index([ghost])

    async with AsyncSessionLocal() as s:
        r = await os_reconcile.reconcile(s, tenant_id)        # 테넌트 스코프 — 남의 문서는 건드리지 않는다
    assert r['indexed'] == 1 and r['deleted'] == 1 and r['unit'] == 'document'
    assert (await _doc(doc_id)).status == 'pending'         # 재등재 = pending 되돌림 + INDEX 행
    assert await indexed_chunk_texts(999_999_001) == []

    assert await ingest(doc_id) == {'done': 1, 'failed': 0}  # 워커가 다시 색인
    assert (await _doc(doc_id)).status == 'ready'
    assert await indexed_chunk_texts(doc_id)
    async with AsyncSessionLocal() as s:
        assert (await os_reconcile.reconcile(s, tenant_id))['indexed'] == 0


@pytest.mark.asyncio
async def test_reconcile_faqs는_사라진_FAQ를_재등재하고_유령을_지운다(client, tenant_id, fake_embed):
    res = await client.post('/kms/faqs', json={'question': '환불 기간은?', 'variants': [], 'answer': '7일'})
    faq_id = res.json()['id']
    await sync_faq(faq_id)
    assert await faq_doc(faq_id)
    await os_index._delete_by_terms('faq_id', [faq_id])
    ghost = os_index.build_doc(
        os_index._ParsedChunk(cid=os_index.chunk_os_id(faq_id=999_999_002), tenant_id=tenant_id,
                                document_id=None, faq_id=999_999_002, text='Q: 유령 A: 유령',
                                heading_path=['유령'], page=None, meta={}, dense=[0.01] * 1024),
        None, None, searchable=True)
    await os_index.bulk_index([ghost])

    async with AsyncSessionLocal() as s:
        r = await os_reconcile.reconcile_faqs(s, tenant_id)
    assert r['indexed'] == 1 and r['deleted'] == 1 and r['unit'] == 'faq'
    assert await faq_doc(999_999_002) is None
    assert await sync_faq(faq_id) == {'done': 1, 'failed': 0}
    assert await faq_doc(faq_id)


@pytest.mark.asyncio
async def test_업로드_시_지정한_폴더가_색인_메타까지_간다(client, tenant_id, fake_queue, blob_tmp, fake_embed):
    """#165: 업로드 시점 folder_id는 INDEX_DOCUMENT 경로 하나로 색인까지 간다 — PATCH가 필요 없다.

    위 1번(인용 메타) 테스트는 업로드 후 PATCH로 옮기는 경로를 본다. 여기서 보는 건 그 전 단계,
    즉 처음부터 폴더를 달고 들어온 문서가 META 부분갱신 없이도 폴더 메타를 갖느냐다.
    """
    folder = (await client.post('/kms/folders',
                                json={'name': '정책', 'description': '환불·반품 규정 모음'})).json()
    res = await client.post('/kms/documents', files={
        'file': ('환불정책.md', MD, 'text/markdown'),
        'document-data': ('blob', json.dumps({'folder_id': folder['id']}), 'application/json'),
    })
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
    """#165: 폴더의 참조 off가 **신규 업로드에도** 먹는지.

    이미 색인된 문서는 폴더 토글이 META 부분갱신으로 반영된다(위 2번). 처음부터 off 폴더로
    들어온 문서는 그 경로를 타지 않고 INDEX_DOCUMENT로 색인되므로, 색인 시점의
    effective_searchable 계산에 폴더 상태가 들어가야 한다 — 빠지면 대외비 폴더에 올린 문서가
    검색에 그대로 노출된다.
    """
    folder = (await client.post('/kms/folders', json={'name': '대외비'})).json()
    assert (await client.patch(f"/kms/folders/{folder['id']}",
                               json={'is_searchable': False})).status_code == 200

    res = await client.post('/kms/documents', files={
        'file': ('환불정책.md', MD, 'text/markdown'),
        'document-data': ('blob', json.dumps({'folder_id': folder['id']}), 'application/json'),
    })
    assert res.status_code == 200, res.text
    doc_id = res.json()['document_id']
    await ingest(doc_id)

    async with AsyncSessionLocal() as s:
        cands = await retrieve_candidates(s, tenant_id, '반품 기간', top_n=20)
    assert [c for c in cands.chunks if c.document_id == doc_id] == [], '참조 off 폴더 문서가 검색 후보에 떴다'


@pytest.mark.asyncio
async def test_일괄_참조끄기가_한_행으로_모든_문서를_제외한다(client, tenant_id, fake_queue,
                                                        blob_tmp, fake_embed):
    """#166: 일괄 변경이 남기는 META 행은 **하나**인데, 그 한 행이 담긴 문서 전부를 갱신해야 한다.

    위 2번(단건 토글)은 문서 하나짜리 payload를 본다. 여기서 보는 건 payload의 document_ids가
    여러 개일 때다 — 워커가 최종값이 같은 문서끼리 묶어 한 번에 갱신하는 경로(sync_meta_documents_now)라
    한쪽만 반영되면 "체크해서 껐는데 일부가 계속 검색되는" 형태로 샌다.
    """
    a = await _upload(client, '환불정책.md')
    b = await _upload(client, '배송정책.md')
    await ingest(a['document_id'])
    await ingest(b['document_id'])
    assert {a['document_id'], b['document_id']} <= await _cited_ids(tenant_id)

    res = await client.patch('/kms/documents', json={
        'document_ids': [a['document_id'], b['document_id']], 'is_searchable': False})
    assert res.status_code == 200, res.text

    # 한 행에 둘 다 담겨 있으므로 drain은 한 번이면 된다 (pending_row_ids가 document_ids 배열도 찾는다)
    assert await ingest(a['document_id']) == {'done': 1, 'failed': 0}
    ids = await _cited_ids(tenant_id)
    assert a['document_id'] not in ids and b['document_id'] not in ids


# ── 6. 백오프 (#185) ─────────────────────────────────────────────────────────────

async def _row(doc_id) -> SearchIndexOutbox:
    """문서의 **최신** INDEX_DOCUMENT 행 — attempts·next_attempt_at·status를 본다.
    failed 행을 재사용해 재업로드하면(#161) 같은 document_id로 행이 둘이라 최신을 고른다."""
    async with AsyncSessionLocal() as s:
        return (await s.execute(select(SearchIndexOutbox)
                                .where(SearchIndexOutbox.op == outbox.INDEX_DOCUMENT)
                                .where(SearchIndexOutbox.payload['document_id'].as_integer() == doc_id)
                                .order_by(SearchIndexOutbox.id.desc()).limit(1))
                ).scalars().one()


async def _due_ids() -> set[int]:
    """운영 cron이 이번 회차에 집을 행 id — row_ids 없이 due_pending_stmt를 그대로 실행한다.
    공유 DB라 남의 행도 섞이므로 포함 여부만 단언한다."""
    async with AsyncSessionLocal() as s:
        return {r.id for r in (await s.execute(outbox.due_pending_stmt(limit=1000))).all()}


@pytest.mark.asyncio
async def test_일시_실패는_다음_시도를_미루고_cron_선정에서_빠진다(client, tenant_id, fake_queue, blob_tmp,
                                                              fake_embed, monkeypatch):
    """엔진 연결 실패 → attempts=1, next_attempt_at≈now+1분, 전체 drain의 선정 대상에서 빠진다.
    시각을 과거로 돌리면 다시 잡힌다. row_ids 지정(ingest)은 백오프를 무시한다(헬퍼 계약)."""
    doc_id = (await _upload(client, '환불정책.md'))['document_id']
    import rag.documents as rd

    async def _down(**kw):
        raise ConnectionError('Cannot connect to host localhost:9200')
    monkeypatch.setattr(rd.os_index, 'index_parsed_document', _down)

    assert await ingest(doc_id) == {'done': 0, 'failed': 1}
    row = await _row(doc_id)
    assert (row.status, row.attempts) == ('pending', 1)
    async with AsyncSessionLocal() as s:
        now = (await s.execute(select(func.now()))).scalar()
    assert timedelta(seconds=30) < row.next_attempt_at - now <= timedelta(minutes=1)   # backoff_delay(1)=1분
    assert row.id not in await _due_ids()                       # cron은 건너뛴다

    assert await ingest(doc_id) == {'done': 0, 'failed': 1}     # row_ids 지정 → 백오프 무시, 즉시 재시도
    row = await _row(doc_id)
    assert row.attempts == 2
    assert row.next_attempt_at - now > timedelta(minutes=1, seconds=30)   # backoff_delay(2)=2분으로 늘었다

    async with AsyncSessionLocal() as s:                        # 시각이 지나면 다시 선정된다
        await s.execute(update(SearchIndexOutbox).where(SearchIndexOutbox.id == row.id)
                        .values(next_attempt_at=func.now() - timedelta(seconds=1)))
        await s.commit()
    assert row.id in await _due_ids()
    assert (await _doc(doc_id)).status == 'pending'             # 문서는 failed가 아니다


@pytest.mark.asyncio
async def test_일시_실패가_MAX_ATTEMPTS에_닿으면_확정되고_훅이_불린다(client, tenant_id, fake_queue, blob_tmp,
                                                                  fake_embed, monkeypatch):
    doc_id = (await _upload(client, '환불정책.md'))['document_id']
    import rag.documents as rd

    async def _down(**kw):
        raise ConnectionError('Cannot connect')
    monkeypatch.setattr(rd.os_index, 'index_parsed_document', _down)
    monkeypatch.setattr(outbox, 'MAX_ATTEMPTS', 2)
    hook_calls = []
    monkeypatch.setattr(outbox, '_on_failed', lambda op, row_id, err: hook_calls.append((op, row_id, err)))
    before = outbox.SEARCH_INDEX_FAILED_TOTAL.labels(op=outbox.INDEX_DOCUMENT)._value.get()

    assert await ingest(doc_id) == {'done': 0, 'failed': 1}
    assert hook_calls == []                                     # 1회차 — 아직 백오프
    assert await ingest(doc_id) == {'done': 0, 'failed': 1}
    row = await _row(doc_id)
    assert (row.status, row.attempts) == ('failed', 2)
    assert (await _doc(doc_id)).status == 'failed'
    assert hook_calls == [(outbox.INDEX_DOCUMENT, row.id, 'Cannot connect')]
    # 훅을 대역으로 바꿨으니 카운터는 안 올랐어야 한다 — 훅이 카운터의 유일한 증가 지점임을 같이 확인
    assert outbox.SEARCH_INDEX_FAILED_TOTAL.labels(op=outbox.INDEX_DOCUMENT)._value.get() == before


@pytest.mark.asyncio
async def test_결정적_실패는_META_행도_1회로_확정하고_카운터가_오른다(client, tenant_id, fake_queue, blob_tmp,
                                                                fake_embed, monkeypatch):
    """INDEX_DOCUMENT 아닌 op에서도 분류가 같다 — META_DOCUMENTS 핸들러가 프로그래밍 오류(결정적)로
    죽으면 재시도 없이 failed, 확정 카운터 +1(진짜 _on_failed 경로)."""
    doc_id = (await _upload(client, '환불정책.md'))['document_id']
    assert await ingest(doc_id) == {'done': 1, 'failed': 0}
    res = await client.patch(f'/kms/documents/{doc_id}', json={'is_searchable': False})
    assert res.status_code == 200, res.text

    async def _bug(session, ids):
        raise KeyError('folder_id')
    monkeypatch.setattr(outbox.os_index, 'sync_meta_documents_now', _bug)
    before = outbox.SEARCH_INDEX_FAILED_TOTAL.labels(op=outbox.META_DOCUMENTS)._value.get()

    assert await ingest(doc_id) == {'done': 0, 'failed': 1}
    rows = [r for r in await _rows(doc_id) if r[0] == outbox.META_DOCUMENTS]
    assert rows == [(outbox.META_DOCUMENTS, 'failed', 1, "'folder_id'")]
    assert outbox.SEARCH_INDEX_FAILED_TOTAL.labels(op=outbox.META_DOCUMENTS)._value.get() == before + 1


# ── 7. failed 확정 시 잔여 청크 DROP (#184) ──────────────────────────────────────

async def _drop_rows(doc_id):
    async with AsyncSessionLocal() as s:
        return (await s.execute(select(SearchIndexOutbox)
                                .where(SearchIndexOutbox.op == outbox.DROP_DOCUMENTS)
                                .where(SearchIndexOutbox.payload['document_ids'].contains([doc_id]))
                                .order_by(SearchIndexOutbox.id))).scalars().all()


def _break_commit_stage(monkeypatch):
    """③ 커밋 단계만 죽인다 — ②는 끝나 엔진에 청크가 있고, mark_done(③이 ready 승격과 함께 부른다)이 폭발.
    결정적 예외라 1회로 failed 확정(#185)."""
    async def _boom(session, row_id):
        raise RuntimeError('③ 커밋 직전 폭발')
    monkeypatch.setattr(outbox, 'mark_done', _boom)


@pytest.mark.asyncio
async def test_커밋_단계만_실패해_failed가_된_문서의_청크는_다음_회차에_지워진다(client, tenant_id, fake_queue,
                                                                        blob_tmp, fake_embed, monkeypatch):
    doc_id = (await _upload(client, '환불정책.md'))['document_id']
    _break_commit_stage(monkeypatch)
    assert await ingest(doc_id) == {'done': 0, 'failed': 1}
    doc = await _doc(doc_id)
    assert doc.status == 'failed' and '③ 커밋 직전 폭발' in doc.status_reason
    assert len(await indexed_chunk_texts(doc_id)) > 0             # 바로 그 불일치 — failed인데 청크가 있다
    (drop,) = await _drop_rows(doc_id)
    assert drop.status == 'pending' and drop.payload == {'document_ids': [doc_id]}

    monkeypatch.undo()                                          # 엔진·mark_done 정상
    assert await ingest(doc_id) == {'done': 1, 'failed': 0}     # DROP 행
    assert await indexed_chunk_texts(doc_id) == []              # failed = 청크 0, 이제 참
    assert (await _drop_rows(doc_id))[0].status == 'done'


@pytest.mark.asyncio
async def test_DROP은_되살아난_문서를_건너뛴다(client, tenant_id, fake_queue, blob_tmp, fake_embed, monkeypatch):
    """failed 확정의 DROP이 미뤄진 사이 재업로드(#161 행 재사용)가 먼저 색인을 끝냈다 — 늦게 도는
    DROP이 새 청크를 지우면 ready인데 검색에 없는 문서가 된다. 살아 있는 문서는 건너뛰어야 한다."""
    doc_id = (await _upload(client, '환불정책.md'))['document_id']
    _break_commit_stage(monkeypatch)
    assert await ingest(doc_id) == {'done': 0, 'failed': 1}
    monkeypatch.undo()
    (drop,) = await _drop_rows(doc_id)

    again = await _upload(client, '환불정책.md')                 # failed 행 재사용 — 같은 id, pending
    assert again['document_id'] == doc_id and again['version'] == 1
    index_row = await _row(doc_id)
    assert index_row.id > drop.id and index_row.status == 'pending'

    # 순서를 뒤집어 재현: INDEX 먼저(ready), 그 뒤 미뤄졌던 DROP
    assert await outbox.drain_once(row_ids=[index_row.id]) == {'done': 1, 'failed': 0}
    assert (await _doc(doc_id)).status == 'ready'
    texts = await indexed_chunk_texts(doc_id)
    assert len(texts) > 0
    assert await outbox.drain_once(row_ids=[drop.id]) == {'done': 1, 'failed': 0}
    assert await indexed_chunk_texts(doc_id) == texts           # 건너뛰었다 — 청크 그대로
    assert (await _drop_rows(doc_id))[0].status == 'done'
