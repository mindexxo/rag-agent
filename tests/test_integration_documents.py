"""D-4: 문서 수명주기 통합 테스트.

업로드(API)→인덱싱(그 문서의 outbox 행만 drain — 워커 프로세스 없이 핸들러 경로 검증)→
버전 엎어치기(supersede)→dedupe→실패 경로(drain이 failed 확정)→소프트 삭제.
업로드는 arq를 만지지 않는다 — 문서와 색인 대기열 행이 같은 트랜잭션에 커밋된다 (#139 outbox).
"""
import json
from datetime import timedelta
from pathlib import Path

import pytest
from sqlalchemy import delete as sql_delete, func, select, update

from database import AsyncSessionLocal
from rag import cache
from rag import outbox
from rag.models import SearchIndexOutbox
from tests.conftest import ingest
from tests.helpers_documents import (MD, chunk_texts, doc_data, get_doc, list_docs,
                                     make_folder, post_upload, upload)
from rag.models import AnswerCache as AnswerCacheRow, Conversation, Document, Message


@pytest.mark.asyncio
async def test_업로드_인덱싱_ready_승격(client, tenant_id, fake_queue, blob_tmp):
    body = await upload(client, '환불정책.md', MD)
    assert body['status'] == 'pending'                       # 업로드 직후
    async with AsyncSessionLocal() as s:                                 # 대기열 행이 문서와 함께 커밋됨 (#139)
        rows = await outbox.pending_row_ids(s, document_id=body['document_id'])
    assert len(rows) == 1

    await ingest(body['document_id'])                 # 워커 잡 직접 실행

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
    # 빈 창 없음(E2E #6): v2가 처리되기 전엔 v1이 그대로 검색된다 — 옛 청크를 먼저 지우지 않는다.
    assert (await get_doc(v1['document_id'])).is_active is True
    assert any('14일' in t for t in await chunk_texts(v1['document_id']))
    await ingest(v2['document_id'])

    old, new = await get_doc(v1['document_id']), await get_doc(v2['document_id'])
    assert new.status == 'ready' and new.is_active is True
    assert old.status == 'deleted' and old.is_active is False   # supersede
    assert any('30일' in t for t in await chunk_texts(v2['document_id']))
    # 옛 청크는 ③ 커밋이 남긴 DROP 행이 다음 회차에 지운다(#186) — 그 전까진 남아 있다(≤1분 공존, 제품 결정)
    assert any('14일' in t for t in await chunk_texts(v1['document_id']))
    assert await ingest(v1['document_id']) == {'done': 1, 'failed': 0}   # 그 DROP 행
    assert await chunk_texts(v1['document_id']) == []                    # 옛 청크 제거
    async with AsyncSessionLocal() as session:                   # 옛 근거 캐시 무효화
        rows = (await session.execute(
            select(AnswerCacheRow).where(AnswerCacheRow.tenant_id == tenant_id)
        )).scalars().all()
        assert rows == []


@pytest.mark.asyncio
async def test_같은_내용_재업로드도_새_버전(client, tenant_id, fake_queue, blob_tmp):
    """내용 해시 dedupe 제거(2026-08-05) — 식별 기준은 filename 하나뿐.

    같은 이름이면 내용이 같아도 새 version이 되고 인덱싱도 다시 돈다.
    ("같은 이름이면 물어보고, 확인하면 대체"라는 단일 규칙을 유지하기 위한 선택)
    """
    v1 = await upload(client, '환불정책.md', MD)
    await ingest(v1['document_id'])

    again = await upload(client, '환불정책.md', MD)             # 내용까지 동일
    assert again['document_id'] != v1['document_id']             # 재사용하지 않는다
    assert again['version'] == v1['version'] + 1
    async with AsyncSessionLocal() as s:                         # 인덱싱이 다시 대기열에 오른다 (#139)
        assert len(await outbox.pending_row_ids(s, document_id=again['document_id'])) == 1


@pytest.mark.asyncio
async def test_인덱싱_예외시_failed_기록(client, tenant_id, fake_queue, blob_tmp, monkeypatch):
    body = await upload(client, '환불정책.md', MD)

    import rag.documents as rd

    async def _boom(texts):
        raise RuntimeError('임베딩 서버 폭발')

    monkeypatch.setattr(rd, 'embed_texts', _boom)
    # RuntimeError는 결정적 실패로 분류돼 1회로 확정된다(#185) — MAX_ATTEMPTS를 낮출 필요가 없다.
    r = await ingest(body['document_id'])              # 핸들러는 예외를 올리고, drain이 failed를 찍는다 (#139)
    assert r == {'done': 0, 'failed': 1}

    doc = await get_doc(body['document_id'])
    assert doc.status == 'failed'
    assert '임베딩 서버 폭발' in doc.status_reason               # 재업로드 판단 근거가 남는다
    async with AsyncSessionLocal() as s:
        row = (await s.execute(select(SearchIndexOutbox)
                               .where(SearchIndexOutbox.op == outbox.INDEX_DOCUMENT)
                               .where(SearchIndexOutbox.payload['document_id'].as_integer()
                                      == body['document_id']))).scalars().one()
        # failed 확정은 잔여 청크 DROP을 같은 커밋에 남긴다(#184) — 여기선 청크가 없어 0건 삭제로 끝난다
        drops = (await s.execute(select(SearchIndexOutbox)
                                 .where(SearchIndexOutbox.op == outbox.DROP_DOCUMENTS)
                                 .where(SearchIndexOutbox.payload['document_ids']
                                        .contains([body['document_id']])))).scalars().all()
    assert row.status == 'failed' and row.attempts == 1
    assert len(drops) == 1 and drops[0].status == 'pending'
    assert await ingest(body['document_id']) == {'done': 1, 'failed': 0}   # 그 DROP — 없는 청크, 무해


@pytest.mark.asyncio
async def test_빈_파일은_failed_유령_ready_방지(client, tenant_id, fake_queue, blob_tmp):
    body = await upload(client, '빈문서.md', b'')
    r = await ingest(body['document_id'])                        # 핸들러가 ValueError → 결정적 → 1회로 failed 확정(#185)
    assert r['failed'] == 1
    doc = await get_doc(body['document_id'])
    assert doc.status == 'failed'                                # 청크 0개 → ready 승격 금지 (C2)
    assert '추출된 텍스트가 없습니다' in doc.status_reason


@pytest.mark.asyncio
async def test_cp949_txt_안깨지고_인덱싱(client, tenant_id, fake_queue, blob_tmp):
    content = '배송비는 삼천원입니다. 도서산간은 오천원 추가.'.encode('cp949')
    body = await upload(client, '공지.txt', content, mime='text/plain')
    await ingest(body['document_id'])

    doc = await get_doc(body['document_id'])
    assert doc.status == 'ready'
    texts = await chunk_texts(body['document_id'])
    assert any('삼천원' in t for t in texts)                     # P2 CP949 실전 검증
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
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                     base_url='http://testserver',
                                     headers={'X-Tenant-Id': other}) as other_client:
            b_doc = await upload(other_client, '환불정책.md', MD.replace(b'14', b'99'))
        await ingest(b_doc['document_id'])

        # 테넌트 A가 같은 파일명 업로드 → 인덱싱 (supersede 실행)
        a_doc = await upload(client, '환불정책.md', MD)
        await ingest(a_doc['document_id'])

        b_after = await get_doc(b_doc['document_id'])
        assert b_after.status == 'ready' and b_after.is_active is True   # B는 무사해야 함
        assert await chunk_texts(b_doc['document_id']) != []
    finally:
        from sqlalchemy import delete as sa_delete

        from rag.models import AnswerCache as ACRow
        from rag import os_index
        await os_index._delete_by_terms('tenant_id', [other])
        async with AsyncSessionLocal() as session:
            await session.execute(sa_delete(ACRow).where(ACRow.tenant_id == other))
            await session.execute(sa_delete(Document).where(Document.tenant_id == other))
            await session.commit()


@pytest.mark.asyncio
async def test_이미_처리된_문서의_대기열_행은_무해하게_done된다(client, tenant_id, fake_queue, blob_tmp):
    """(구 P1-4 "큐 등록 실패→failed" 테스트를 대체) #139 outbox 전환으로 업로드는 Redis를 만지지
    않는다 — 대기열 행이 문서와 같은 트랜잭션에 커밋되므로 그 실패 모드 자체가 사라졌다.
    대신 at-least-once의 전제를 고정한다: 이미 ready인 문서에 INDEX_DOCUMENT 행이 또 있어도
    (재시도 겹침·수동 재등재) 핸들러는 다시 색인하지 않고 행만 done으로 넘긴다.
    """
    body = await upload(client, '환불정책.md', MD)
    await ingest(body['document_id'])
    doc = await get_doc(body['document_id'])
    assert doc.status == 'ready'
    before = await chunk_texts(doc.id)
    async with AsyncSessionLocal() as s:
        outbox.enqueue(s, tenant_id, outbox.INDEX_DOCUMENT, document_id=doc.id)   # 중복 등재
        await s.commit()

    r = await ingest(body['document_id'])
    assert r == {'done': 1, 'failed': 0}
    async with AsyncSessionLocal() as s:
        left = await outbox.pending_row_ids(s, document_id=doc.id)
    assert await chunk_texts(doc.id) == before                  # 재색인 안 함
    assert left == []                                            # 행은 done


@pytest.mark.asyncio
async def test_소프트_삭제_청크와_캐시_제거_row_보존(client, tenant_id, fake_queue, blob_tmp, fake_embed):
    body = await upload(client, '환불정책.md', MD)
    await ingest(body['document_id'])
    async with AsyncSessionLocal() as session:
        await cache.save_answer(session, tenant_id, '반품 기간', '14일', [], [body['document_id']])
        await session.commit()

    res = await client.delete(f"/kms/documents/{body['document_id']}")
    assert res.status_code == 204

    doc = await get_doc(body['document_id'])
    assert doc is not None                                       # row 보존 (과거 인용 다운로드용)
    assert doc.status == 'deleted' and doc.is_active is False
    assert await chunk_texts(body['document_id']) != []                  # drain 전 — 가이드대로 아직 색인에 있다 (E2E #4)
    await ingest(body['document_id'])                                     # DROP_DOCUMENTS 행 처리
    assert await chunk_texts(body['document_id']) == []                  # 검색 인덱스에서 제거
    async with AsyncSessionLocal() as session:
        rows = (await session.execute(
            select(AnswerCacheRow).where(AnswerCacheRow.tenant_id == tenant_id)
        )).scalars().all()
        assert rows == []                                        # 근거 캐시 무효화


# ===== 업로드 전 동일 파일명 확인 (GET /documents/exists) =====================

@pytest.mark.asyncio
async def test_exists_없는_파일명은_false(client, tenant_id, fake_queue, blob_tmp):
    res = await client.get('/kms/documents/exists', params={'filename': '없는문서.md'})
    assert res.status_code == 200
    assert res.json() == {'exists': False, 'document_id': None, 'version': None,
                          'status': None, 'uploaded_at': None}


@pytest.mark.asyncio
async def test_exists_기존_문서는_현재_버전을_알려준다(client, tenant_id, fake_queue, blob_tmp):
    v1 = await upload(client, '환불정책.md', MD)
    await ingest(v1['document_id'])

    body = (await client.get('/kms/documents/exists', params={'filename': '환불정책.md'})).json()
    assert body['exists'] is True
    assert body['document_id'] == v1['document_id']
    assert body['version'] == 1                      # 업로드하면 v2가 된다는 안내용
    assert body['status'] == 'ready'


@pytest.mark.asyncio
async def test_exists_는_최신_버전을_본다(client, tenant_id, fake_queue, blob_tmp):
    v1 = await upload(client, '환불정책.md', MD)
    await ingest(v1['document_id'])
    v2 = await upload(client, '환불정책.md', MD.replace(b'14', b'30'))
    await ingest(v2['document_id'])

    body = (await client.get('/kms/documents/exists', params={'filename': '환불정책.md'})).json()
    assert body['document_id'] == v2['document_id']  # supersede된 v1이 아니라 살아 있는 v2
    assert body['version'] == 2


@pytest.mark.asyncio
async def test_exists_는_완전일치만_본다(client, tenant_id, fake_queue, blob_tmp):
    """정책: 대소문자·공백 구분. supersede 기준과 동일해야 한다."""
    v1 = await upload(client, '환불정책.md', MD)
    await ingest(v1['document_id'])

    for other in ('환불정책.MD', '환불정책 .md', '환불정책_최종.md'):
        body = (await client.get('/kms/documents/exists', params={'filename': other})).json()
        assert body['exists'] is False, other


@pytest.mark.asyncio
async def test_exists_는_테넌트_격리(client, tenant_id, fake_queue, blob_tmp):
    v1 = await upload(client, '환불정책.md', MD)
    await ingest(v1['document_id'])

    res = await client.get('/kms/documents/exists', params={'filename': '환불정책.md'},
                           headers={'X-Tenant-Id': 'other-tenant'})
    assert res.json()['exists'] is False


# ===== 낙관적 잠금 (expect_version) ==========================================
# 확인창 조회~업로드 사이에 DB가 바뀌면 409. 조회만으로는 창이 남으므로 서버가 함께 검사한다.

@pytest.mark.asyncio
async def test_expect_version_0_은_없는_이름에서_통과(client, tenant_id, fake_queue, blob_tmp):
    res = await post_upload(client, '환불정책.md', MD, expect_version=0)
    assert res.status_code == 200 and res.json()['version'] == 1


@pytest.mark.asyncio
async def test_expect_version_0_인데_이미_있으면_409(client, tenant_id, fake_queue, blob_tmp):
    """'별도 문서로 등록' 경로 — 고른 이름이 그 사이 선점됐다."""
    v1 = await upload(client, '환불정책(1).md', MD)
    await ingest(v1['document_id'])

    res = await post_upload(client, '환불정책(1).md', MD, expect_version=0)
    assert res.status_code == 409
    body = res.json()
    assert body['filename'] == '환불정책(1).md'    # 어느 이름이 걸렸는지
    assert body['current_version'] == 1            # FE가 다음 번호를 추천하는 근거
    assert isinstance(body['detail'], str)         # 공용 에러 토스트가 그대로 쓴다


@pytest.mark.asyncio
async def test_expect_version_일치하면_대체된다(client, tenant_id, fake_queue, blob_tmp, fake_embed):
    v1 = await upload(client, '환불정책.md', MD)
    await ingest(v1['document_id'])

    res = await post_upload(client, '환불정책.md', MD.replace(b'14', b'30'), expect_version=1)
    assert res.status_code == 200 and res.json()['version'] == 2


@pytest.mark.asyncio
async def test_expect_version_어긋나면_409(client, tenant_id, fake_queue, blob_tmp, fake_embed):
    """화면에서 v1을 봤지만 그 사이 남이 v2를 올렸다 → 조용히 대체하지 않고 409."""
    v1 = await upload(client, '환불정책.md', MD)
    await ingest(v1['document_id'])
    v2 = await upload(client, '환불정책.md', MD.replace(b'14', b'30'))
    await ingest(v2['document_id'])

    res = await post_upload(client, '환불정책.md', MD, expect_version=1)
    assert res.status_code == 409
    assert res.json()['current_version'] == 2      # 다시 물어볼 때 쓸 값


@pytest.mark.asyncio
async def test_expect_version_미전송이면_검사하지_않는다(client, tenant_id, fake_queue, blob_tmp, fake_embed):
    """하위호환 — 구버전 FE는 파라미터 없이 부르고, 그 경우 기존처럼 대체된다."""
    v1 = await upload(client, '환불정책.md', MD)
    await ingest(v1['document_id'])

    res = await post_upload(client, '환불정책.md', MD)
    assert res.status_code == 200 and res.json()['version'] == 2


@pytest.mark.asyncio
async def test_expect_version_판정은_exists와_같은_기준(client, tenant_id, fake_queue, blob_tmp, fake_embed):
    """소프트 삭제된 이름은 '없음'(0)으로 본다 — exists가 그렇게 답하므로 기준이 같아야 한다."""
    v1 = await upload(client, '환불정책.md', MD)
    await ingest(v1['document_id'])
    assert (await client.delete(f"/kms/documents/{v1['document_id']}")).status_code == 204

    assert (await client.get('/kms/documents/exists',
                             params={'filename': '환불정책.md'})).json()['exists'] is False
    res = await post_upload(client, '환불정책.md', MD, expect_version=0)
    assert res.status_code == 200                  # 기준이 어긋나면 여기서 409가 난다


@pytest.mark.asyncio
async def test_expect_version_은_테넌트별로_판정(client, tenant_id, other_tenant_id, fake_queue, blob_tmp):
    """남의 테넌트에 같은 이름이 있어도 내 쪽은 '없음'이다."""
    res = await client.post('/kms/documents',
                            files={'file': ('환불정책.md', MD, 'text/markdown')},
                            headers={'X-Tenant-Id': other_tenant_id})
    assert res.status_code == 200

    assert (await post_upload(client, '환불정책.md', MD, expect_version=0)).status_code == 200


@pytest.mark.asyncio
async def test_409면_blob이_남지_않는다(client, tenant_id, fake_queue, blob_tmp):
    """blob은 insert 전에 쓰므로, 409로 빠질 때 지우지 않으면 참조 없는 파일이 쌓인다."""
    v1 = await upload(client, '환불정책.md', MD)
    await ingest(v1['document_id'])
    before = sorted((blob_tmp / tenant_id).iterdir())

    assert (await post_upload(client, '환불정책.md', MD, expect_version=0)).status_code == 409
    assert sorted((blob_tmp / tenant_id).iterdir()) == before
    # 대기열 행도 문서와 같은 트랜잭션이라 409 롤백에 함께 사라져야 한다 (#139 outbox 원자성).
    # v1의 행은 ingest로 done이 됐으니 pending은 0이어야 한다.
    async with AsyncSessionLocal() as s:
        pending = (await s.execute(select(func.count()).select_from(SearchIndexOutbox)
                                   .where(SearchIndexOutbox.tenant_id == tenant_id)
                                   .where(SearchIndexOutbox.status == 'pending'))).scalar()
    assert pending == 0


@pytest.mark.asyncio
async def test_동시_삽입은_유니크_인덱스가_막고_409(client, tenant_id, fake_queue, blob_tmp, monkeypatch):
    """조회를 함께 통과한 두 요청 중 하나는 UNIQUE(tenant_id, filename, version)에 걸린다.

    실제 동시 요청 대신 insert가 이미 있는 (filename, version)을 쓰도록 만들어 그 경로만 본다.
    이전에는 이 위반이 그대로 터져 500이었다.
    """
    v1 = await upload(client, '환불정책.md', MD)
    await ingest(v1['document_id'])

    import routers.documents as rd
    real = rd.handle_upload

    # 인자는 그대로 흘려보낸다 — handle_upload에 파라미터가 하나 늘 때마다 이 대역이 깨져
    # 무관한 테스트가 실패하던 것을 끊는다(#164·#165에서 연달아 겪었다).
    async def _collide(session, t, filename, blob_path, **kw):
        doc, stale = await real(session, t, filename, blob_path, **kw)
        doc.version = 1          # 남이 방금 v1을 넣은 것과 같은 결과
        return doc, stale

    monkeypatch.setattr(rd, 'handle_upload', _collide)

    res = await post_upload(client, '환불정책.md', MD)
    assert res.status_code == 409
    assert res.json()['current_version'] == 1
    assert sorted((blob_tmp / tenant_id).iterdir())      # 남은 건 v1의 blob 하나뿐



# ── #164 등록일시·등록자 ──────────────────────────────────────

@pytest.mark.asyncio
async def test_목록은_등록일시_내림차순이고_등록자를_싣는다(client, tenant_id, fake_queue, blob_tmp):
    """목록 API의 기본 정렬 키는 uploaded_at이다 (#164).

    정렬 키가 정말 uploaded_at인지 보려면 id 순서와 어긋나게 만들어야 한다 —
    그냥 두면 id.desc()와 결과가 같아 무엇으로 정렬했는지 구분되지 않는다.
    """
    first = await upload(client, '가정책.md', MD, headers={'X-User-Id': 'agent-a'})
    second = await upload(client, '나정책.md', MD, headers={'X-User-Id': 'agent-b'})

    items = (await client.get('/kms/documents')).json()['items']
    assert [d['document_id'] for d in items] == [second['document_id'], first['document_id']]
    assert items[0]['uploaded_by'] == 'agent-b'
    assert items[0]['uploaded_at']                      # datetime 직렬화 확인

    # 먼저 올린 문서의 등록일시만 미래로 옮긴다 → id 순서와 반대가 되어야 한다.
    # (aware 값을 그대로 대입한다 — 시각 컬럼 매핑이 TIMESTAMPTZ에 맞춰져 있다, rag/models.py의 Base)
    async with AsyncSessionLocal() as s:
        doc = await s.get(Document, first['document_id'])
        doc.uploaded_at = doc.uploaded_at + timedelta(days=1)
        await s.commit()

    items = (await client.get('/kms/documents')).json()['items']
    assert [d['document_id'] for d in items] == [first['document_id'], second['document_id']]


@pytest.mark.asyncio
async def test_등록자는_X_User_Id로_기록되고_미전송이면_null(client, tenant_id, fake_queue, blob_tmp):
    """#164: 헤더가 없으면 NULL로 남긴다 — conversation의 'test-user' 폴백을 쓰지 않는다.
    그 가짜 값이 문서 관리 화면의 등록자 칸에 그대로 뜨기 때문."""
    with_user = await upload(client, '가정책.md', MD, headers={'X-User-Id': 'agent-a'})
    without = await upload(client, '나정책.md', MD)

    assert with_user['uploaded_by'] == 'agent-a'
    assert without['uploaded_by'] is None
    assert (await get_doc(with_user['document_id'])).uploaded_by == 'agent-a'
    assert (await get_doc(without['document_id'])).uploaded_by is None


@pytest.mark.asyncio
async def test_재업로드_등록자는_계승하지_않는다(client, tenant_id, fake_queue, blob_tmp):
    """#164: 문서에 건 설정(is_searchable)은 개정판에 계승되지만, 등록자는 그 버전을 올린
    사람으로 갱신된다 — 계승하면 최신 개정을 누가 했는지 화면에서 알 수 없다."""
    v1 = await upload(client, '환불정책.md', MD, headers={'X-User-Id': 'agent-a'})
    await ingest(v1['document_id'])
    assert (await client.patch(f"/kms/documents/{v1['document_id']}",
                               json={'is_searchable': False})).status_code == 200

    v2 = await upload(client, '환불정책.md', MD, headers={'X-User-Id': 'agent-b'})
    assert v2['version'] == 2
    assert v2['is_searchable'] is False       # 설정은 계승
    assert v2['uploaded_by'] == 'agent-b'     # 등록자는 갱신

    # 헤더 없이 올린 버전이 직전 등록자를 물려받지 않는지 — "계승하지 않는다"가 깨지는
    # 가장 흔한 형태(prev 폴백)를 직접 찍는다.
    v3 = await upload(client, '환불정책.md', MD)
    assert v3['version'] == 3
    assert v3['uploaded_by'] is None


# ── #165 업로드 시 폴더 지정 (document-data 파트) ─────────────

@pytest.mark.asyncio
async def test_업로드_시_폴더_지정(client, tenant_id, fake_queue, blob_tmp):
    fid = await make_folder(client, '규정집')
    body = await upload(client, '환불정책.md', MD, extra_parts=doc_data(folder_id=fid))
    assert body['folder_id'] == fid
    assert (await get_doc(body['document_id'])).folder_id == fid


@pytest.mark.asyncio
async def test_document_data를_문자열_파트로_보내도_동작(client, tenant_id, fake_queue, blob_tmp):
    """브라우저는 Blob(=UploadFile)로, 다른 클라이언트는 문자열 필드로 보낼 수 있다 — 둘 다 받는다."""
    fid = await make_folder(client, '규정집')
    body = await upload(client, '환불정책.md', MD,
                         extra_parts={'document-data': (None, json.dumps({'folder_id': fid}))})
    assert body['folder_id'] == fid


@pytest.mark.asyncio
async def test_folder_id_null이면_미분류로_뗀다(client, tenant_id, fake_queue, blob_tmp):
    """null 전송 = 미분류. 값만 보면 '안 보냄'과 같아서, 보냈는지 여부로 갈린다."""
    fid = await make_folder(client, '규정집')
    v1 = await upload(client, '환불정책.md', MD, extra_parts=doc_data(folder_id=fid))
    await ingest(v1['document_id'])

    v2 = await upload(client, '환불정책.md', MD, extra_parts=doc_data(folder_id=None))
    assert v2['version'] == 2
    assert v2['folder_id'] is None


@pytest.mark.asyncio
async def test_folder_id_미전송이면_직전_버전에서_계승(client, tenant_id, fake_queue, blob_tmp):
    fid = await make_folder(client, '규정집')
    v1 = await upload(client, '환불정책.md', MD, extra_parts=doc_data(folder_id=fid))
    await ingest(v1['document_id'])

    # document-data는 보내되 folder_id 키만 없다 → 계승
    v2 = await upload(client, '환불정책.md', MD, extra_parts=doc_data(description='표 설명'))
    assert v2['folder_id'] == fid
    assert (await get_doc(v2['document_id'])).description == '표 설명'   # JSON 경로도 값이 산다
    # 파트 자체를 안 보내도 계승 (옛 방식 호출)
    v3 = await upload(client, '환불정책.md', MD)
    assert v3['folder_id'] == fid


@pytest.mark.asyncio
async def test_없는_폴더는_404_문서도_blob도_안_남는다(client, tenant_id, fake_queue, blob_tmp):
    res = await client.post('/kms/documents',
                            files={'file': ('환불정책.md', MD, 'text/markdown'),
                                   **doc_data(folder_id=999999)})
    assert res.status_code == 404
    async with AsyncSessionLocal() as s:
        cnt = (await s.execute(select(func.count()).select_from(Document)
                               .where(Document.tenant_id == tenant_id))).scalar()
    assert cnt == 0
    assert not (blob_tmp / tenant_id).exists() or not list((blob_tmp / tenant_id).iterdir())


@pytest.mark.asyncio
async def test_다른_테넌트_폴더는_404(client, tenant_id, other_tenant_id, fake_queue, blob_tmp):
    """격리 — 남의 폴더 id를 넣어도 통과하면 안 된다."""
    res = await client.post('/kms/folders', json={'name': '남의폴더'},
                            headers={'X-Tenant-Id': other_tenant_id})
    other_fid = res.json()['id']

    res = await client.post('/kms/documents',
                            files={'file': ('환불정책.md', MD, 'text/markdown'),
                                   **doc_data(folder_id=other_fid)})
    assert res.status_code == 404


@pytest.mark.asyncio
async def test_document_data와_평면필드_동시_전송은_400(client, tenant_id, fake_queue, blob_tmp):
    """한쪽을 조용히 무시하면 '보냈는데 안 먹는' 버그가 된다 — 명시 거절."""
    res = await client.post('/kms/documents',
                            files={'file': ('환불정책.md', MD, 'text/markdown'),
                                   **doc_data(description='새 방식')},
                            data={'description': '옛 방식'})
    assert res.status_code == 400


@pytest.mark.asyncio
async def test_잘못된_document_data는_422(client, tenant_id, fake_queue, blob_tmp):
    for bad in ('{깨진 json', json.dumps({'folder_id': 'abc'}), json.dumps({'오타필드': 1})):
        res = await client.post('/kms/documents',
                                files={'file': ('환불정책.md', MD, 'text/markdown'),
                                       'document-data': ('blob', bad, 'application/json')})
        assert res.status_code == 422, f'{bad} → {res.status_code}'


@pytest.mark.asyncio
async def test_옛_평면필드_방식은_그대로_동작(client, tenant_id, fake_queue, blob_tmp):
    """FE 전환 전 호출 호환 — description·expect_version 평면 필드."""
    res = await client.post('/kms/documents',
                            files={'file': ('환불정책.md', MD, 'text/markdown')},
                            data={'description': '표 설명', 'expect_version': '0'})
    assert res.status_code == 200, res.text
    assert (await get_doc(res.json()['document_id'])).description == '표 설명'


@pytest.mark.asyncio
async def test_신규_문서에_폴더를_안_주면_미분류(client, tenant_id, fake_queue, blob_tmp):
    """계승할 직전 버전이 없을 때의 기본값 — 파트를 안 보내도, 보내되 folder_id만 빼도 미분류."""
    a = await upload(client, '환불정책.md', MD)
    b = await upload(client, '배송정책.md', MD, extra_parts=doc_data(description='표 설명'))
    assert a['folder_id'] is None and b['folder_id'] is None


# ── #166 문서 일괄 변경 (PATCH /kms/documents) ────────────────

async def _bulk(client, **body):
    return await client.patch('/kms/documents', json=body)


@pytest.mark.asyncio
async def test_일괄_폴더이동과_참조토글(client, tenant_id, fake_queue, blob_tmp):
    fid = await make_folder(client, '규정집')
    a = await upload(client, '환불정책.md', MD)
    b = await upload(client, '배송정책.md', MD)

    res = await _bulk(client, document_ids=[a['document_id'], b['document_id']],
                      folder_id=fid, is_searchable=False)
    assert res.status_code == 200, res.text
    body = res.json()
    assert {d['document_id'] for d in body} == {a['document_id'], b['document_id']}
    assert all(d['folder_id'] == fid and d['is_searchable'] is False for d in body)
    assert all(d['ref_count'] is None for d in body)      # 목록 API 전용 집계
    for d in (a, b):
        doc = await get_doc(d['document_id'])
        assert doc.folder_id == fid and doc.is_searchable is False


@pytest.mark.asyncio
async def test_일괄_folder_id_null이면_미분류(client, tenant_id, fake_queue, blob_tmp):
    fid = await make_folder(client, '규정집')
    a = await upload(client, '환불정책.md', MD, extra_parts=doc_data(folder_id=fid))
    res = await _bulk(client, document_ids=[a['document_id']], folder_id=None)
    assert res.status_code == 200, res.text
    assert res.json()[0]['folder_id'] is None


@pytest.mark.asyncio
async def test_일괄_대상이_하나라도_어긋나면_아무것도_안_바뀐다(
        client, tenant_id, other_tenant_id, fake_queue, blob_tmp):
    """없는 id·다른 테넌트 id·삭제된 문서 — 셋 다 전체 거절이고, 성공분도 반영되지 않는다."""
    fid = await make_folder(client, '규정집')
    mine = await upload(client, '환불정책.md', MD)

    # 다른 테넌트의 문서
    other = await client.post('/kms/documents', files={'file': ('남의문서.md', MD, 'text/markdown')},
                              headers={'X-Tenant-Id': other_tenant_id})
    # 삭제된 문서
    gone = await upload(client, '지울문서.md', MD)
    assert (await client.delete(f"/kms/documents/{gone['document_id']}")).status_code == 204

    for label, bad in (('없는 id', 999999),
                       ('다른 테넌트', other.json()['document_id']),
                       ('삭제된 문서', gone['document_id'])):
        res = await _bulk(client, document_ids=[mine['document_id'], bad], folder_id=fid)
        assert res.status_code == 404, f'{label} → {res.status_code}'
        assert (await get_doc(mine['document_id'])).folder_id is None, f'{label}: 성공분이 반영됐다'


@pytest.mark.asyncio
async def test_일괄_실효참조가_꺼지는_문서만_캐시_무효화(client, tenant_id, fake_queue, blob_tmp):
    """문서마다 현재 폴더·스위치가 달라 on→off 전이도 제각각이다 — 일괄 판정하면 안 된다.

    off 폴더에 있던 문서는 이미 실효 off라 지울 캐시가 없고, on이던 문서만 무효화 대상이다.
    """
    off_folder = await make_folder(client, '대외비')
    assert (await client.patch(f'/kms/folders/{off_folder}',
                               json={'is_searchable': False})).status_code == 200

    on_doc = await upload(client, '환불정책.md', MD)                                   # 미분류 = 실효 on
    off_doc = await upload(client, '배송정책.md', MD, extra_parts=doc_data(folder_id=off_folder))

    async with AsyncSessionLocal() as s:
        await cache.save_answer(s, tenant_id, '질의 하나', '답1', [], [on_doc['document_id']])
        await cache.save_answer(s, tenant_id, '질의 둘', '답2', [], [off_doc['document_id']])
        await s.commit()

    # 둘 다 문서 스위치를 끈다 — 실효 참조가 실제로 on→off로 바뀌는 건 on_doc뿐이다
    res = await _bulk(client, document_ids=[on_doc['document_id'], off_doc['document_id']],
                      is_searchable=False)
    assert res.status_code == 200, res.text

    async with AsyncSessionLocal() as s:
        remain = (await s.execute(select(AnswerCacheRow.answer)
                                  .where(AnswerCacheRow.tenant_id == tenant_id))).scalars().all()
    assert remain == ['답2'], '이미 off였던 문서의 캐시까지 지웠거나, on이던 문서 캐시가 남았다'


@pytest.mark.asyncio
async def test_일괄_색인_갱신은_한_행에_담긴다(client, tenant_id, fake_queue, blob_tmp):
    a = await upload(client, '환불정책.md', MD)
    b = await upload(client, '배송정책.md', MD)
    ids = [a['document_id'], b['document_id']]
    async with AsyncSessionLocal() as s:      # 업로드가 만든 INDEX_DOCUMENT 행을 비우고 본다
        await s.execute(sql_delete(SearchIndexOutbox).where(SearchIndexOutbox.tenant_id == tenant_id))
        await s.commit()

    assert (await _bulk(client, document_ids=ids, is_searchable=False)).status_code == 200

    async with AsyncSessionLocal() as s:
        rows = (await s.execute(select(SearchIndexOutbox)
                                .where(SearchIndexOutbox.tenant_id == tenant_id))).scalars().all()
    assert len(rows) == 1 and rows[0].op == outbox.META_DOCUMENTS
    assert sorted(rows[0].payload['document_ids']) == sorted(ids)


@pytest.mark.asyncio
async def test_일괄_상한_초과는_422(client, tenant_id, fake_queue, blob_tmp):
    res = await _bulk(client, document_ids=list(range(1, 202)), is_searchable=False)
    assert res.status_code == 422


@pytest.mark.asyncio
async def test_일괄_바꿀_것이_없으면_무동작(client, tenant_id, fake_queue, blob_tmp):
    """빈 배열이든 바꿀 필드가 없든 빈 목록 200 — 이 경로에선 id 유효성도 따지지 않는다."""
    assert (await _bulk(client, document_ids=[], folder_id=None)).json() == []
    assert (await _bulk(client, document_ids=[999999])).json() == []


@pytest.mark.asyncio
async def test_삭제된_문서는_단건_PATCH도_404(client, tenant_id, fake_queue, blob_tmp):
    """#166: 목록에 뜨지도 않는 죽은 행의 컬럼이 조용히 바뀌던 것을 막는다."""
    doc = await upload(client, '환불정책.md', MD)
    assert (await client.delete(f"/kms/documents/{doc['document_id']}")).status_code == 204
    res = await client.patch(f"/kms/documents/{doc['document_id']}", json={'is_searchable': False})
    assert res.status_code == 404


@pytest.mark.asyncio
async def test_일괄_폴더가_없거나_남의_것이면_404(client, tenant_id, other_tenant_id,
                                                fake_queue, blob_tmp):
    """대상 문서는 멀쩡해도 옮길 폴더가 어긋나면 전체 거절 — 문서도 안 바뀐다."""
    doc = await upload(client, '환불정책.md', MD)
    other_folder = (await client.post('/kms/folders', json={'name': '남의폴더'},
                                      headers={'X-Tenant-Id': other_tenant_id})).json()['id']

    for label, fid in (('없는 폴더', 999999), ('다른 테넌트 폴더', other_folder)):
        res = await _bulk(client, document_ids=[doc['document_id']], folder_id=fid)
        assert res.status_code == 404, f'{label} → {res.status_code}'
    assert (await get_doc(doc['document_id'])).folder_id is None


@pytest.mark.asyncio
async def test_일괄_상한_경계(client, tenant_id, fake_queue, blob_tmp):
    """200건은 스키마를 통과하고(대상이 없어 404), 201건은 스키마에서 422.

    유효한 문서 200건을 실제로 만들지 않고 경계만 본다 — 통과 여부는 대상 조회 이전에
    갈리므로 404가 나오면 스키마를 지난 것이다.
    """
    assert (await _bulk(client, document_ids=list(range(1, 201)),
                        is_searchable=False)).status_code == 404
    assert (await _bulk(client, document_ids=list(range(1, 202)),
                        is_searchable=False)).status_code == 422


@pytest.mark.asyncio
async def test_일괄_404는_어긋난_id를_알려준다(client, tenant_id, fake_queue, blob_tmp):
    """화면이 '무엇이 문제인지' 보여줄 수 있어야 한다 — 어느 id가 걸렸는지 응답에 싣는다."""
    doc = await upload(client, '환불정책.md', MD)
    res = await _bulk(client, document_ids=[doc['document_id'], 999999], is_searchable=False)
    assert res.status_code == 404
    assert '999999' in res.json()['detail']
    assert str(doc['document_id']) not in res.json()['detail']   # 멀쩡한 id는 안 싣는다


# ── #174 문서 일괄 삭제 (DELETE /kms/documents?ids=) ──────────

async def _bulk_delete(client, ids, tenant=None):
    headers = {'X-Tenant-Id': tenant} if tenant else None
    return await client.request('DELETE', '/kms/documents', params={'ids': ids}, headers=headers)


@pytest.mark.asyncio
async def test_일괄_삭제는_목록에서_사라지고_전_버전을_함께_내린다(client, tenant_id, fake_queue, blob_tmp):
    a_v1 = await upload(client, '환불정책.md', MD)
    await ingest(a_v1['document_id'])
    a_v2 = await upload(client, '환불정책.md', MD)          # 같은 이름 재업로드 → v2
    b = await upload(client, '배송정책.md', MD)

    assert (await _bulk_delete(client, [a_v2['document_id'], b['document_id']])).status_code == 204

    # 요청한 건 v2지만 같은 filename의 v1도 함께 내려간다
    for did in (a_v1['document_id'], a_v2['document_id'], b['document_id']):
        assert (await get_doc(did)).status == 'deleted'
    assert (await client.get('/kms/documents')).json()['items'] == []


@pytest.mark.asyncio
async def test_일괄_삭제는_이미_삭제된_문서를_통과시킨다(client, tenant_id, fake_queue, blob_tmp):
    """멱등 — 삭제는 '그 문서가 없는 상태'를 만드는 것이라 이미 그 상태면 성공이다.
    단건 DELETE가 원래 그렇게 동작했고 같은 규칙을 쓴다."""
    gone = await upload(client, '환불정책.md', MD)
    alive = await upload(client, '배송정책.md', MD)
    assert (await client.delete(f"/kms/documents/{gone['document_id']}")).status_code == 204

    res = await _bulk_delete(client, [gone['document_id'], alive['document_id']])
    assert res.status_code == 204
    assert (await get_doc(alive['document_id'])).status == 'deleted'


@pytest.mark.asyncio
async def test_일괄_삭제_대상이_어긋나면_아무것도_안_지운다(client, tenant_id, other_tenant_id,
                                                        fake_queue, blob_tmp):
    mine = await upload(client, '환불정책.md', MD)
    other = await client.post('/kms/documents', files={'file': ('남의문서.md', MD, 'text/markdown')},
                              headers={'X-Tenant-Id': other_tenant_id})

    for label, bad in (('없는 id', 999999), ('다른 테넌트', other.json()['document_id'])):
        res = await _bulk_delete(client, [mine['document_id'], bad])
        assert res.status_code == 404, f'{label} → {res.status_code}'
        assert (await get_doc(mine['document_id'])).status != 'deleted', f'{label}: 성공분이 지워졌다'
    assert '999999' in (await _bulk_delete(client, [999999])).json()['detail']


@pytest.mark.asyncio
async def test_일괄_삭제_캐시_무효화와_색인_제거(client, tenant_id, fake_queue, blob_tmp):
    """라우터를 거쳐 캐시가 지워지는지, DROP 대기열이 한 행에 전 버전을 담는지."""
    a = await upload(client, '환불정책.md', MD)
    b = await upload(client, '배송정책.md', MD)
    await ingest(a['document_id'])
    await ingest(b['document_id'])

    async with AsyncSessionLocal() as s:
        await cache.save_answer(s, tenant_id, '질의 하나', '답1', [], [a['document_id']])
        await cache.save_answer(s, tenant_id, '질의 둘', '답2', [], [b['document_id']])
        await s.commit()
        await s.execute(sql_delete(SearchIndexOutbox).where(SearchIndexOutbox.tenant_id == tenant_id))
        await s.commit()

    assert (await _bulk_delete(client, [a['document_id'], b['document_id']])).status_code == 204

    async with AsyncSessionLocal() as s:
        remain = (await s.execute(select(AnswerCacheRow.answer)
                                  .where(AnswerCacheRow.tenant_id == tenant_id))).scalars().all()
        rows = (await s.execute(select(SearchIndexOutbox)
                                .where(SearchIndexOutbox.tenant_id == tenant_id))).scalars().all()
    assert remain == []                                  # 둘 다 근거가 사라졌다
    assert len(rows) == 1 and rows[0].op == outbox.DROP_DOCUMENTS
    assert sorted(rows[0].payload['document_ids']) == sorted([a['document_id'], b['document_id']])

    await ingest(a['document_id'])                       # 한 행에 둘 다 담겨 있다
    assert await chunk_texts(a['document_id']) == []
    assert await chunk_texts(b['document_id']) == []


@pytest.mark.asyncio
async def test_일괄_삭제_상한과_빈_요청(client, tenant_id, fake_queue, blob_tmp):
    assert (await _bulk_delete(client, list(range(1, 201)))).status_code == 404   # 스키마는 통과
    assert (await _bulk_delete(client, list(range(1, 202)))).status_code == 422
    assert (await _bulk_delete(client, [])).status_code == 204                   # 무동작


@pytest.mark.asyncio
async def test_단건_삭제_동작_보존(client, tenant_id, fake_queue, blob_tmp):
    """공용 함수로 옮긴 뒤에도 단건 계약이 그대로인지 — 없는 id는 404, 재호출은 204(멱등)."""
    doc = await upload(client, '환불정책.md', MD)
    assert (await client.delete(f"/kms/documents/{doc['document_id']}")).status_code == 204
    assert (await client.delete(f"/kms/documents/{doc['document_id']}")).status_code == 204
    assert (await client.delete('/kms/documents/999999')).status_code == 404


# ── #176 목록 페이징 + 검색·필터 ──────────────────────────────

@pytest.mark.asyncio
async def test_목록_페이징_total과_has_more(client, tenant_id, fake_queue, blob_tmp):
    ids = [(await upload(client, f'정책{i}.md', MD))['document_id'] for i in range(5)]

    first = await list_docs(client, limit=2, offset=0)
    assert [d['document_id'] for d in first['items']] == ids[:-3:-1]   # 최신순
    assert first['total'] == 5 and first['has_more'] is True

    last = await list_docs(client, limit=2, offset=4)
    assert len(last['items']) == 1                    # 마지막은 부분 페이지
    assert last['total'] == 5 and last['has_more'] is False

    beyond = await list_docs(client, limit=2, offset=99)
    assert beyond['items'] == [] and beyond['total'] == 5 and beyond['has_more'] is False


@pytest.mark.asyncio
async def test_목록_limit은_범위를_벗어나면_상한으로(client, tenant_id, fake_queue, blob_tmp):
    await upload(client, '정책.md', MD)
    for bad in (0, -1, 9999):
        assert (await list_docs(client, limit=bad))['total'] == 1     # 422가 아니라 상한으로 클램프
    assert (await list_docs(client, offset=-5))['items']              # 음수 offset은 0으로


@pytest.mark.asyncio
async def test_목록_파일명_검색(client, tenant_id, fake_queue, blob_tmp):
    await upload(client, '환불정책.md', MD)
    await upload(client, '배송정책.md', MD)
    await upload(client, 'kms_01_비밀번호.md', MD)

    assert (await list_docs(client, q='환불'))['total'] == 1
    assert (await list_docs(client, q='정책'))['total'] == 2
    assert (await list_docs(client, q='없는이름'))['total'] == 0
    assert (await list_docs(client, q=''))['total'] == 3            # 빈 검색어는 필터 없음

    # LIKE 이스케이프 — _는 '아무 글자 하나'라, 이스케이프하지 않으면 kmsX01도 걸린다
    assert (await list_docs(client, q='kms_01'))['total'] == 1
    assert (await list_docs(client, q='kmsX01'))['total'] == 0


@pytest.mark.asyncio
async def test_목록_검색어가_NFD여도_걸린다(client, tenant_id, fake_queue, blob_tmp):
    """파일명은 NFC로 저장된다(#34). 이름을 복사해 붙여넣으면 NFD일 수 있다 — macOS 경로."""
    import unicodedata
    await upload(client, '환불정책.md', MD)
    nfd = unicodedata.normalize('NFD', '환불')
    assert nfd != '환불'                                  # 실제로 분해형이 맞는지 먼저 확인
    assert (await list_docs(client, q=nfd))['total'] == 1


@pytest.mark.asyncio
async def test_목록_폴더_필터와_미분류(client, tenant_id, fake_queue, blob_tmp):
    fid = await make_folder(client, '규정집')
    in_folder = await upload(client, '환불정책.md', MD, extra_parts=doc_data(folder_id=fid))
    unfiled = await upload(client, '배송정책.md', MD)

    assert [d['document_id'] for d in (await list_docs(client, folder_id=fid))['items']] \
        == [in_folder['document_id']]
    assert [d['document_id'] for d in (await list_docs(client, folder_id=0))['items']] \
        == [unfiled['document_id']]                       # 0 = 미분류
    assert (await list_docs(client))['total'] == 2            # 필터 없으면 전부


@pytest.mark.asyncio
async def test_목록_상태_필터(client, tenant_id, fake_queue, blob_tmp):
    doc = await upload(client, '환불정책.md', MD)         # 업로드 직후는 pending
    assert (await list_docs(client, status='pending'))['total'] == 1
    assert (await list_docs(client, status='ready'))['total'] == 0
    assert (await list_docs(client, status=['ready', 'pending']))['total'] == 1   # 복수 선택

    await ingest(doc['document_id'])                       # ready로 승격
    assert (await list_docs(client, status='ready'))['total'] == 1

    res = await client.get('/kms/documents', params={'status': 'done'})
    assert res.status_code == 422                          # 어휘에 없는 값은 조용히 0건이 아니라 거절


@pytest.mark.asyncio
async def test_목록_답변사용_필터(client, tenant_id, fake_queue, blob_tmp):
    on = await upload(client, '환불정책.md', MD)
    off = await upload(client, '배송정책.md', MD)
    assert (await client.patch(f"/kms/documents/{off['document_id']}",
                               json={'is_searchable': False})).status_code == 200

    assert [d['document_id'] for d in (await list_docs(client, is_searchable=True))['items']] \
        == [on['document_id']]
    assert [d['document_id'] for d in (await list_docs(client, is_searchable=False))['items']] \
        == [off['document_id']]


@pytest.mark.asyncio
async def test_목록_ref_count는_이번_페이지만_집계한다(client, tenant_id, fake_queue, blob_tmp):
    """집계를 페이지의 파일명으로 좁혔다(#176) — 좁히는 조건이 틀리면 0이 되거나 남의 값이 붙는다."""
    cited = await upload(client, '환불정책.md', MD)
    await upload(client, '배송정책.md', MD)

    async with AsyncSessionLocal() as s:
        conv = Conversation(tenant_id=tenant_id, created_by='agent-a')
        s.add(conv)
        await s.flush()
        s.add(Message(tenant_id=tenant_id, conversation_id=conv.id, role='assistant',
                      content='답변', cited_docs=['환불정책.md']))
        await s.commit()

    body = await list_docs(client)
    counts = {d['filename']: d['ref_count'] for d in body['items']}
    assert counts == {'환불정책.md': 1, '배송정책.md': 0}

    # 인용된 문서가 페이지에 없을 때도 다른 문서 값이 오염되지 않는다
    only_other = await list_docs(client, q='배송')
    assert [d['ref_count'] for d in only_other['items']] == [0]
    assert cited['document_id'] not in [d['document_id'] for d in only_other['items']]


@pytest.mark.asyncio
async def test_답변사용_필터는_문서_스위치_기준이다(client, tenant_id, fake_queue, blob_tmp):
    """#176 결정을 고정한다 — 폴더가 참조 off여도 **문서 스위치가 on이면** is_searchable=true에 잡힌다.

    실제로 검색에 쓰이는지는 폴더와의 곱(실효 참조)이지만, 그 판정을 이 필터에 넣는 것은
    제품 결정이지 버그 수정이 아니다(같은 규칙의 네 번째 사본이 된다는 이유로 미뤘다).
    나중에 '고치려는' 변경이 오면 이 테스트가 먼저 걸려 결정을 다시 보게 한다.
    """
    off_folder = await make_folder(client, '대외비')
    assert (await client.patch(f'/kms/folders/{off_folder}',
                               json={'is_searchable': False})).status_code == 200
    doc = await upload(client, '환불정책.md', MD, extra_parts=doc_data(folder_id=off_folder))
    assert doc['is_searchable'] is True                      # 문서 스위치는 그대로 on

    on = (await list_docs(client, is_searchable=True))['items']
    assert [d['document_id'] for d in on] == [doc['document_id']]
    assert (await list_docs(client, is_searchable=False))['items'] == []


@pytest.mark.asyncio
async def test_목록_문서가_없으면_빈_페이지(client, tenant_id):
    """문서가 하나도 없는 테넌트 — items·total·has_more의 기본값이 화면에서 깨지지 않아야 한다."""
    body = await list_docs(client)
    assert body == {'items': [], 'total': 0, 'has_more': False}


@pytest.mark.asyncio
async def test_목록_필터를_건_채로_페이징하면_total도_필터_기준(client, tenant_id, fake_queue, blob_tmp):
    """total이 필터를 안 거치면 '전체 6건'이라 써놓고 3건만 보이는 화면이 된다.

    조건을 한 곳에서 만들어 count와 페이지 쿼리가 함께 쓰는지를 보는 테스트다 —
    필터와 페이징을 따로 검증하면 이 어긋남이 안 잡힌다.
    """
    fid = await make_folder(client, '규정집')
    for i in range(3):                                   # 폴더 안 3건
        await upload(client, f'규정{i}.md', MD, extra_parts=doc_data(folder_id=fid))
    for i in range(3):                                   # 미분류 3건
        await upload(client, f'기타{i}.md', MD)

    first = await list_docs(client, folder_id=fid, limit=2, offset=0)
    assert first['total'] == 3                           # 6이 아니라 3 — 필터 기준
    assert len(first['items']) == 2 and first['has_more'] is True
    assert all(d['folder_id'] == fid for d in first['items'])

    second = await list_docs(client, folder_id=fid, limit=2, offset=2)
    assert second['total'] == 3 and len(second['items']) == 1
    assert second['has_more'] is False                    # 필터 기준으로 마지막 페이지

    # 검색어 + 페이징도 같은 규칙
    q_page = await list_docs(client, q='규정', limit=2, offset=2)
    assert q_page['total'] == 3 and len(q_page['items']) == 1 and q_page['has_more'] is False


# ── #181 응답 시각은 KST 오프셋 ──────────────────────────────

@pytest.mark.asyncio
async def test_uploaded_at은_KST_오프셋으로_나가고_시점은_같다(client, tenant_id, fake_queue, blob_tmp):
    from datetime import datetime
    body = await upload(client, '환불정책.md', MD)
    assert body['uploaded_at'].endswith('+09:00')
    # 응답 문자열을 다시 파싱하면 DB의 UTC 값과 같은 시점 — 변환이지 이동이 아니다
    assert datetime.fromisoformat(body['uploaded_at']) == (await get_doc(body['document_id'])).uploaded_at
    # 목록·exists도 같은 형식 (스키마 세 곳이 같은 타입을 쓴다)
    assert (await list_docs(client))['items'][0]['uploaded_at'].endswith('+09:00')
    ex = (await client.get('/kms/documents/exists', params={'filename': '환불정책.md'})).json()
    assert ex['uploaded_at'].endswith('+09:00')


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
