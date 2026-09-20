"""문서가 들어오는 경로 — 업로드·색인·버전 정책 (#188 D로 분할).

업로드(API)→인덱싱(그 문서의 outbox 행만 drain — 워커 프로세스 없이 핸들러 경로 검증)→
버전 엎어치기(supersede)→dedupe→실패 경로(drain이 failed 확정), 그리고 업로드 전 확인
(GET /documents/exists)과 낙관적 잠금(expect_version).
업로드는 arq를 만지지 않는다 — 문서와 색인 대기열 행이 같은 트랜잭션에 커밋된다 (#139 outbox).

같은 축의 나머지: _folders(등록정보·폴더·일괄 변경) _delete(삭제) _list(목록) _failed(#161).
공용 헬퍼는 tests/helpers_documents.py.
"""
import pytest
from sqlalchemy import func, select

from database import AsyncSessionLocal
from rag import cache, outbox
from rag.models import AnswerCache as AnswerCacheRow, Document, SearchIndexOutbox
from tests.conftest import ingest
from tests.helpers_documents import MD, chunk_texts, get_doc, post_upload, upload


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
