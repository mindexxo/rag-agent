"""문서가 들어오는 경로 — 업로드·색인·버전 정책.

업로드(API)→인덱싱(그 문서의 outbox 행만 drain — 워커 프로세스 없이 핸들러 경로 검증)→
버전 엎어치기(supersede)→dedupe→실패 경로(drain이 failed 확정), 그리고 원본 다운로드.
업로드는 arq를 만지지 않는다 — 문서와 색인 대기열(outbox) 행이 같은 트랜잭션에 커밋된다.

같은 축의 나머지: _folders(등록정보·폴더·일괄 변경) _delete(삭제) _list(목록) _failed(실패 문서)
_reupload(재업로드 뒤 실제 질의 경로의 캐시·인용) _version(동일 파일명 확인·낙관적 잠금).
공용 헬퍼는 tests/helpers_documents.py.
"""

from pathlib import Path
from urllib.parse import quote

import pytest
from sqlalchemy import select

from config import settings
from database import AsyncSessionLocal
from rag import outbox
from rag.models import AnswerCache as AnswerCacheRow
from rag.models import Document, SearchIndexOutbox
from rag.turn import cache
from tests.conftest import ingest
from tests.helpers_documents import MD, chunk_texts, get_doc, upload


@pytest.mark.asyncio
async def test_업로드_인덱싱_ready_승격(client, tenant_id, fake_queue, blob_tmp):
    body = await upload(client, '환불정책.md', MD)
    assert body['status'] == 'pending'  # 업로드 직후
    async with AsyncSessionLocal() as s:  # 대기열 행이 문서와 함께 커밋됨
        rows = await outbox.pending_row_ids(s, document_id=body['document_id'])
    assert len(rows) == 1

    await ingest(body['document_id'])  # 워커 잡 직접 실행

    doc = await get_doc(body['document_id'])
    assert doc.status == 'ready' and doc.is_active is True
    texts = await chunk_texts(body['document_id'])
    assert texts and any('14일' in t for t in texts)
    assert doc.char_count == sum(len(t) for t in texts)


@pytest.mark.asyncio
async def test_같은_파일명_재업로드는_엎어치기(client, tenant_id, fake_queue, blob_tmp, fake_embed):
    v1 = await upload(client, '환불정책.md', MD)
    await ingest(v1['document_id'])

    # v1을 근거로 만든 캐시가 있다고 가정
    async with AsyncSessionLocal() as session:
        await cache.save_answer(session, tenant_id, '반품 기간', '14일', [], [v1['document_id']])
        await session.commit()

    v2 = await upload(client, '환불정책.md', MD.replace(b'14', b'30'))
    assert v2['document_id'] != v1['document_id'] and v2['version'] == 2
    # 빈 창 없음: v2가 처리되기 전엔 v1이 그대로 검색된다 — 옛 청크를 먼저 지우지 않는다.
    assert (await get_doc(v1['document_id'])).is_active is True
    assert any('14일' in t for t in await chunk_texts(v1['document_id']))
    await ingest(v2['document_id'])

    old, new = await get_doc(v1['document_id']), await get_doc(v2['document_id'])
    assert new.status == 'ready' and new.is_active is True
    assert old.status == 'deleted' and old.is_active is False  # supersede
    assert any('30일' in t for t in await chunk_texts(v2['document_id']))
    # 옛 청크는 커밋이 남긴 DROP 행이 다음 회차에 지운다 — 그 전까진 남아 있다(짧은 공존 허용)
    assert any('14일' in t for t in await chunk_texts(v1['document_id']))
    assert await ingest(v1['document_id']) == {'done': 1, 'failed': 0}  # 그 DROP 행
    assert await chunk_texts(v1['document_id']) == []  # 옛 청크 제거
    async with AsyncSessionLocal() as session:  # 옛 근거 캐시 무효화
        rows = (
            (await session.execute(select(AnswerCacheRow).where(AnswerCacheRow.tenant_id == tenant_id))).scalars().all()
        )
        assert rows == []


@pytest.mark.asyncio
async def test_같은_내용_재업로드도_새_버전(client, tenant_id, fake_queue, blob_tmp):
    """내용 해시 dedupe는 없다 — 식별 기준은 filename 하나뿐.

    같은 이름이면 내용이 같아도 새 version이 되고 인덱싱도 다시 돈다.
    ("같은 이름이면 물어보고, 확인하면 대체"라는 단일 규칙을 유지하기 위한 선택)
    """
    v1 = await upload(client, '환불정책.md', MD)
    await ingest(v1['document_id'])

    again = await upload(client, '환불정책.md', MD)  # 내용까지 동일
    assert again['document_id'] != v1['document_id']  # 재사용하지 않는다
    assert again['version'] == v1['version'] + 1
    async with AsyncSessionLocal() as s:  # 인덱싱이 다시 대기열에 오른다
        assert len(await outbox.pending_row_ids(s, document_id=again['document_id'])) == 1


@pytest.mark.asyncio
async def test_인덱싱_예외시_failed_기록(client, tenant_id, fake_queue, blob_tmp, monkeypatch):
    body = await upload(client, '환불정책.md', MD)

    import rag.ingestion as ri

    async def _boom(texts):
        raise RuntimeError('임베딩 서버 폭발')

    monkeypatch.setattr(ri, 'embed_texts', _boom)
    # RuntimeError는 결정적 실패로 분류돼 1회로 확정된다 — MAX_ATTEMPTS를 낮출 필요가 없다.
    r = await ingest(body['document_id'])  # 핸들러는 예외를 올리고, drain이 failed를 찍는다
    assert r == {'done': 0, 'failed': 1}

    doc = await get_doc(body['document_id'])
    assert doc.status == 'failed'
    assert '임베딩 서버 폭발' in doc.status_reason  # 재업로드 판단 근거가 남는다
    async with AsyncSessionLocal() as s:
        row = (
            (
                await s.execute(
                    select(SearchIndexOutbox)
                    .where(SearchIndexOutbox.op == outbox.INDEX_DOCUMENT)
                    .where(SearchIndexOutbox.payload['document_id'].as_integer() == body['document_id'])
                )
            )
            .scalars()
            .one()
        )
        # failed 확정은 잔여 청크 DROP을 같은 커밋에 남긴다 — 여기선 청크가 없어 0건 삭제로 끝난다
        drops = (
            (
                await s.execute(
                    select(SearchIndexOutbox)
                    .where(SearchIndexOutbox.op == outbox.DROP_DOCUMENTS)
                    .where(SearchIndexOutbox.payload['document_ids'].contains([body['document_id']]))
                )
            )
            .scalars()
            .all()
        )
    assert row.status == 'failed' and row.attempts == 1
    assert len(drops) == 1 and drops[0].status == 'pending'
    assert await ingest(body['document_id']) == {'done': 1, 'failed': 0}  # 그 DROP — 없는 청크, 무해


@pytest.mark.asyncio
async def test_docling_변환_실패는_failed로_끝나고_비상_경로로_색인하지_않는다(
    client, tenant_id, fake_queue, blob_tmp, monkeypatch
):
    """색인 파서는 docling 하나다. 변환이 실패하면 pdfplumber 등으로 조용히 저품질 색인을 만들지 않고 failed+사유로 끝난다.
    RuntimeError(불완전 변환 가드와 같은 종류)는 결정적 실패라 1회로 확정된다."""
    from io import BytesIO

    from reportlab.pdfgen import canvas

    import rag.chunking as rc

    buf = BytesIO()
    c = canvas.Canvas(buf)
    c.drawString(60, 760, 'Refund within 14 days of delivery.')
    c.save()
    body = await upload(client, '환불정책.pdf', buf.getvalue(), mime='application/pdf')

    def _boom(*args, **kwargs):
        raise RuntimeError('docling 변환 불완전(테스트 주입)')

    monkeypatch.setattr(rc.convert_process, 'docling_convert', _boom)  # PDF·DOCX·MD·XLSX 경로가 전부 이 이름을 부른다
    assert await ingest(body['document_id']) == {'done': 0, 'failed': 1}
    assert not hasattr(settings, 'docling_enabled')  # 비상 스위치가 되살아나면 여기서 깨진다

    doc = await get_doc(body['document_id'])
    assert doc.status == 'failed' and '테스트 주입' in doc.status_reason
    assert await chunk_texts(body['document_id']) == []  # 비상 경로가 있었다면 pdfplumber 청크가 여기 남았다
    async with AsyncSessionLocal() as s:
        row = (
            (
                await s.execute(
                    select(SearchIndexOutbox)
                    .where(SearchIndexOutbox.op == outbox.INDEX_DOCUMENT)
                    .where(SearchIndexOutbox.payload['document_id'].as_integer() == body['document_id'])
                )
            )
            .scalars()
            .one()
        )
    assert row.status == 'failed' and row.attempts == 1


@pytest.mark.asyncio
async def test_빈_파일은_failed_유령_ready_방지(client, tenant_id, fake_queue, blob_tmp):
    body = await upload(client, '빈문서.md', b'')
    r = await ingest(body['document_id'])  # 핸들러가 ValueError → 결정적 → 1회로 failed 확정
    assert r['failed'] == 1
    doc = await get_doc(body['document_id'])
    assert doc.status == 'failed'  # 청크 0개 → ready 승격 금지 (C2)
    assert '추출된 텍스트가 없습니다' in doc.status_reason


@pytest.mark.asyncio
async def test_cp949_txt_안깨지고_인덱싱(client, tenant_id, fake_queue, blob_tmp):
    content = '배송비는 삼천원입니다. 도서산간은 오천원 추가.'.encode('cp949')
    body = await upload(client, '공지.txt', content, mime='text/plain')
    await ingest(body['document_id'])

    doc = await get_doc(body['document_id'])
    assert doc.status == 'ready'
    texts = await chunk_texts(body['document_id'])
    assert any('삼천원' in t for t in texts)  # P2 CP949 실전 검증
    assert not any('�' in t for t in texts)


@pytest.mark.asyncio
async def test_미지원_확장자는_400(client, tenant_id, fake_queue, blob_tmp):
    res = await client.post('/kms/documents', files={'file': ('악성.exe', b'MZ', 'application/octet-stream')})
    assert res.status_code == 400


@pytest.mark.asyncio
async def test_supersede는_타_테넌트_같은_파일명을_건드리지_않음(client, tenant_id, fake_queue, blob_tmp):
    """뮤테이션 생존자 B 킬 — supersede 조회의 tenant WHERE가 빠지면
    같은 filename을 쓰는 다른 테넌트 문서를 내려버린다 (교차 테넌트 훼손)."""
    import uuid

    import httpx

    from main import app

    other = str(uuid.uuid4())
    try:
        # 테넌트 B가 같은 파일명으로 먼저 업로드·인덱싱
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url='http://testserver', headers={'X-Tenant-Id': other}
        ) as other_client:
            b_doc = await upload(other_client, '환불정책.md', MD.replace(b'14', b'99'))
        await ingest(b_doc['document_id'])

        # 테넌트 A가 같은 파일명 업로드 → 인덱싱 (supersede 실행)
        a_doc = await upload(client, '환불정책.md', MD)
        await ingest(a_doc['document_id'])

        b_after = await get_doc(b_doc['document_id'])
        assert b_after.status == 'ready' and b_after.is_active is True  # B는 무사해야 함
        assert await chunk_texts(b_doc['document_id']) != []
    finally:
        from sqlalchemy import delete as sa_delete

        from rag.models import AnswerCache as ACRow
        from rag.search import os_index

        await os_index._delete_by_terms('tenant_id', [other])
        async with AsyncSessionLocal() as session:
            await session.execute(sa_delete(ACRow).where(ACRow.tenant_id == other))
            await session.execute(sa_delete(Document).where(Document.tenant_id == other))
            await session.commit()


@pytest.mark.asyncio
async def test_이미_처리된_문서의_대기열_행은_무해하게_done된다(client, tenant_id, fake_queue, blob_tmp):
    """outbox라 업로드는 Redis를 만지지 않는다 — 대기열 행이 문서와 같은 트랜잭션에 커밋되므로 큐 등록 실패 모드가 없다.
    대신 at-least-once의 전제를 고정한다: 이미 ready인 문서에 INDEX_DOCUMENT 행이 또 있어도
    (재시도 겹침·수동 재등재) 핸들러는 다시 색인하지 않고 행만 done으로 넘긴다.
    """
    body = await upload(client, '환불정책.md', MD)
    await ingest(body['document_id'])
    doc = await get_doc(body['document_id'])
    assert doc.status == 'ready'
    before = await chunk_texts(doc.id)
    async with AsyncSessionLocal() as s:
        outbox.enqueue(s, tenant_id, outbox.INDEX_DOCUMENT, document_id=doc.id)  # 중복 등재
        await s.commit()

    r = await ingest(body['document_id'])
    assert r == {'done': 1, 'failed': 0}
    async with AsyncSessionLocal() as s:
        left = await outbox.pending_row_ids(s, document_id=doc.id)
    assert await chunk_texts(doc.id) == before  # 재색인 안 함
    assert left == []  # 행은 done


# ===== 원본 다운로드 (GET /documents/{id}/download) ===========================
# 답변 인용(sources)의 document_id에서 연결되는 경로다. 소프트 삭제 후에도 살아 있어야 한다 —
# 과거 대화가 인용한 원본을 계속 받을 수 있게 row·blob을 보존하는 것이 F5의 설계다.


@pytest.mark.asyncio
async def test_다운로드는_저장된_원본_바이트와_파일명을_준다(client, tenant_id, fake_queue, blob_tmp):
    body = await upload(client, '환불정책.md', MD)

    res = await client.get(f"/kms/documents/{body['document_id']}/download")
    assert res.status_code == 200, res.text
    assert res.content == MD  # blob은 업로드 바이트 그대로다
    # 한글 파일명은 RFC 5987 퍼센트 인코딩으로 나간다. 조각이 아니라 전체를 대조한다.
    assert quote('환불정책.md') in res.headers['content-disposition']


@pytest.mark.asyncio
async def test_다운로드는_소프트_삭제_후에도_된다(client, tenant_id, fake_queue, blob_tmp):
    """삭제는 row·blob을 보존한다 — 과거 대화의 인용에서 원본을 계속 받을 수 있어야 한다."""
    body = await upload(client, '환불정책.md', MD)
    assert (await client.delete(f"/kms/documents/{body['document_id']}")).status_code == 204

    res = await client.get(f"/kms/documents/{body['document_id']}/download")
    assert res.status_code == 200
    assert res.content == MD


@pytest.mark.asyncio
async def test_다운로드는_테넌트_격리(client, tenant_id, fake_queue, blob_tmp):
    body = await upload(client, '환불정책.md', MD)

    res = await client.get(f"/kms/documents/{body['document_id']}/download", headers={'X-Tenant-Id': 'other-tenant'})
    assert res.status_code == 404  # 있는 id여도 남의 것이면 없는 것


@pytest.mark.asyncio
async def test_blob_파일이_사라졌으면_410(client, tenant_id, fake_queue, blob_tmp):
    """DB는 가리키는데 파일이 없는 상태. 404(문서 없음)와 구분해서 알려준다."""
    body = await upload(client, '환불정책.md', MD)
    Path((await get_doc(body['document_id'])).blob_path).unlink()

    res = await client.get(f"/kms/documents/{body['document_id']}/download")
    assert res.status_code == 410
