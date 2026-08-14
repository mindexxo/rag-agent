"""D-4: 문서 수명주기 통합 테스트.

업로드(API)→인덱싱(잡 함수 직접 await — 워커 프로세스 없이 잡 로직 검증)→
버전 엎어치기(supersede)→dedupe→실패 경로(failed 기록)→소프트 삭제.
arq 큐는 가짜로 치환해 enqueue 여부만 기록한다 (큐 배달 자체는 arq 라이브러리 몫).
"""
import pytest
from sqlalchemy import select

from database import AsyncSessionLocal
from rag.cache import AnswerCache
from rag.documents import index_pending_document
from rag.models import AnswerCache as AnswerCacheRow, Chunk, Document


async def _upload(client, filename: str, content: bytes, mime='text/markdown') -> dict:
    res = await client.post('/kms/documents', files={'file': (filename, content, mime)})
    assert res.status_code == 200, res.text
    return res.json()


async def _get_doc(doc_id: int) -> Document:
    async with AsyncSessionLocal() as session:
        return await session.get(Document, doc_id)


async def _chunk_texts(doc_id: int) -> list[str]:
    async with AsyncSessionLocal() as session:
        return list((await session.execute(
            select(Chunk.text).where(Chunk.document_id == doc_id)
        )).scalars().all())


MD = '# 환불 정책\n\n## 1. 기간\n\n단순변심 반품은 14일 이내 신청한다.\n'.encode()


@pytest.mark.asyncio
async def test_업로드_인덱싱_ready_승격(client, tenant_id, fake_queue, blob_tmp):
    body = await _upload(client, '환불정책.md', MD)
    assert body['status'] == 'pending'                       # 업로드 직후
    assert fake_queue == [('index_document', body['document_id'])]    # 잡 등록됨

    await index_pending_document(body['document_id'])                 # 워커 잡 직접 실행

    doc = await _get_doc(body['document_id'])
    assert doc.status == 'ready' and doc.is_active is True
    texts = await _chunk_texts(body['document_id'])
    assert texts and any('14일' in t for t in texts)
    assert doc.char_count == sum(len(t) for t in texts)


@pytest.mark.asyncio
async def test_같은_파일명_재업로드는_엎어치기(client, tenant_id, fake_queue, blob_tmp, fake_embed):
    v1 = await _upload(client, '환불정책.md', MD)
    await index_pending_document(v1['document_id'])

    # v1을 근거로 만든 캐시가 있다고 가정
    async with AsyncSessionLocal() as session:
        await AnswerCache().set(session, tenant_id, '반품 기간', '14일', [], [v1['document_id']])
        await session.commit()

    v2 = await _upload(client, '환불정책.md', MD.replace(b'14', b'30'))
    assert v2['document_id'] != v1['document_id'] and v2['version'] == 2
    await index_pending_document(v2['document_id'])

    old, new = await _get_doc(v1['document_id']), await _get_doc(v2['document_id'])
    assert new.status == 'ready' and new.is_active is True
    assert old.status == 'deleted' and old.is_active is False   # supersede
    assert await _chunk_texts(v1['document_id']) == []                    # 옛 청크 제거
    assert any('30일' in t for t in await _chunk_texts(v2['document_id']))
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
    v1 = await _upload(client, '환불정책.md', MD)
    await index_pending_document(v1['document_id'])
    n_jobs = len(fake_queue)

    again = await _upload(client, '환불정책.md', MD)             # 내용까지 동일
    assert again['document_id'] != v1['document_id']             # 재사용하지 않는다
    assert again['version'] == v1['version'] + 1
    assert len(fake_queue) == n_jobs + 1                         # 인덱싱 재실행


@pytest.mark.asyncio
async def test_인덱싱_예외시_failed_기록(client, tenant_id, fake_queue, blob_tmp, monkeypatch):
    body = await _upload(client, '환불정책.md', MD)

    import rag.documents as rd

    async def _boom(texts):
        raise RuntimeError('임베딩 서버 폭발')

    monkeypatch.setattr(rd, 'embed_texts', _boom)
    await index_pending_document(body['document_id'])                     # 예외를 삼키고 failed 기록해야 함

    doc = await _get_doc(body['document_id'])
    assert doc.status == 'failed'
    assert '임베딩 서버 폭발' in doc.status_reason               # pending 고착 방지 (P1-5a)


@pytest.mark.asyncio
async def test_빈_파일은_failed_유령_ready_방지(client, tenant_id, fake_queue, blob_tmp):
    body = await _upload(client, '빈문서.md', b'')
    await index_pending_document(body['document_id'])
    doc = await _get_doc(body['document_id'])
    assert doc.status == 'failed'                                # 청크 0개 → ready 승격 금지 (C2)


@pytest.mark.asyncio
async def test_cp949_txt_안깨지고_인덱싱(client, tenant_id, fake_queue, blob_tmp):
    content = '배송비는 삼천원입니다. 도서산간은 오천원 추가.'.encode('cp949')
    body = await _upload(client, '공지.txt', content, mime='text/plain')
    await index_pending_document(body['document_id'])

    doc = await _get_doc(body['document_id'])
    assert doc.status == 'ready'
    texts = await _chunk_texts(body['document_id'])
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
            b_doc = await _upload(other_client, '환불정책.md', MD.replace(b'14', b'99'))
        await index_pending_document(b_doc['document_id'])

        # 테넌트 A가 같은 파일명 업로드 → 인덱싱 (supersede 실행)
        a_doc = await _upload(client, '환불정책.md', MD)
        await index_pending_document(a_doc['document_id'])

        b_after = await _get_doc(b_doc['document_id'])
        assert b_after.status == 'ready' and b_after.is_active is True   # B는 무사해야 함
        assert await _chunk_texts(b_doc['document_id']) != []
    finally:
        from sqlalchemy import delete as sa_delete

        from rag.models import AnswerCache as ACRow
        async with AsyncSessionLocal() as session:
            await session.execute(sa_delete(Chunk).where(Chunk.tenant_id == other))
            await session.execute(sa_delete(ACRow).where(ACRow.tenant_id == other))
            await session.execute(sa_delete(Document).where(Document.tenant_id == other))
            await session.commit()


@pytest.mark.asyncio
async def test_enqueue_실패시_failed_기록(client, tenant_id, blob_tmp, monkeypatch):
    """P1-4 수정 검증 — 큐 등록 실패가 pending 영구 고착으로 남지 않는다 (뮤테이션 생존자 C 킬)."""
    import routers.documents as rd

    async def _broken_pool(*a, **kw):
        raise ConnectionError('Redis 순단')

    monkeypatch.setattr(rd, 'create_pool', _broken_pool)
    body = await _upload(client, '환불정책.md', MD)

    assert body['status'] == 'failed'
    assert '큐 등록 실패' in body['status_reason']


@pytest.mark.asyncio
async def test_소프트_삭제_청크와_캐시_제거_row_보존(client, tenant_id, fake_queue, blob_tmp, fake_embed):
    body = await _upload(client, '환불정책.md', MD)
    await index_pending_document(body['document_id'])
    async with AsyncSessionLocal() as session:
        await AnswerCache().set(session, tenant_id, '반품 기간', '14일', [], [body['document_id']])
        await session.commit()

    res = await client.delete(f"/kms/documents/{body['document_id']}")
    assert res.status_code == 204

    doc = await _get_doc(body['document_id'])
    assert doc is not None                                       # row 보존 (과거 인용 다운로드용)
    assert doc.status == 'deleted' and doc.is_active is False
    assert await _chunk_texts(body['document_id']) == []                  # 검색 인덱스에서 제거
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
    v1 = await _upload(client, '환불정책.md', MD)
    await index_pending_document(v1['document_id'])

    body = (await client.get('/kms/documents/exists', params={'filename': '환불정책.md'})).json()
    assert body['exists'] is True
    assert body['document_id'] == v1['document_id']
    assert body['version'] == 1                      # 업로드하면 v2가 된다는 안내용
    assert body['status'] == 'ready'


@pytest.mark.asyncio
async def test_exists_는_최신_버전을_본다(client, tenant_id, fake_queue, blob_tmp):
    v1 = await _upload(client, '환불정책.md', MD)
    await index_pending_document(v1['document_id'])
    v2 = await _upload(client, '환불정책.md', MD.replace(b'14', b'30'))
    await index_pending_document(v2['document_id'])

    body = (await client.get('/kms/documents/exists', params={'filename': '환불정책.md'})).json()
    assert body['document_id'] == v2['document_id']  # supersede된 v1이 아니라 살아 있는 v2
    assert body['version'] == 2


@pytest.mark.asyncio
async def test_exists_는_완전일치만_본다(client, tenant_id, fake_queue, blob_tmp):
    """정책: 대소문자·공백 구분. supersede 기준과 동일해야 한다."""
    v1 = await _upload(client, '환불정책.md', MD)
    await index_pending_document(v1['document_id'])

    for other in ('환불정책.MD', '환불정책 .md', '환불정책_최종.md'):
        body = (await client.get('/kms/documents/exists', params={'filename': other})).json()
        assert body['exists'] is False, other


@pytest.mark.asyncio
async def test_exists_는_테넌트_격리(client, tenant_id, fake_queue, blob_tmp):
    v1 = await _upload(client, '환불정책.md', MD)
    await index_pending_document(v1['document_id'])

    res = await client.get('/kms/documents/exists', params={'filename': '환불정책.md'},
                           headers={'X-Tenant-Id': 'other-tenant'})
    assert res.json()['exists'] is False


# ===== 낙관적 잠금 (expect_version) ==========================================
# 확인창 조회~업로드 사이에 DB가 바뀌면 409. 조회만으로는 창이 남으므로 서버가 함께 검사한다.

async def _post(client, filename: str, content: bytes, expect_version=None):
    data = {} if expect_version is None else {'expect_version': str(expect_version)}
    return await client.post('/kms/documents',
                             files={'file': (filename, content, 'text/markdown')},
                             data=data)


@pytest.mark.asyncio
async def test_expect_version_0_은_없는_이름에서_통과(client, tenant_id, fake_queue, blob_tmp):
    res = await _post(client, '환불정책.md', MD, expect_version=0)
    assert res.status_code == 200 and res.json()['version'] == 1


@pytest.mark.asyncio
async def test_expect_version_0_인데_이미_있으면_409(client, tenant_id, fake_queue, blob_tmp):
    """'별도 문서로 등록' 경로 — 고른 이름이 그 사이 선점됐다."""
    v1 = await _upload(client, '환불정책(1).md', MD)
    await index_pending_document(v1['document_id'])

    res = await _post(client, '환불정책(1).md', MD, expect_version=0)
    assert res.status_code == 409
    body = res.json()
    assert body['filename'] == '환불정책(1).md'    # 어느 이름이 걸렸는지
    assert body['current_version'] == 1            # FE가 다음 번호를 추천하는 근거
    assert isinstance(body['detail'], str)         # 공용 에러 토스트가 그대로 쓴다


@pytest.mark.asyncio
async def test_expect_version_일치하면_대체된다(client, tenant_id, fake_queue, blob_tmp, fake_embed):
    v1 = await _upload(client, '환불정책.md', MD)
    await index_pending_document(v1['document_id'])

    res = await _post(client, '환불정책.md', MD.replace(b'14', b'30'), expect_version=1)
    assert res.status_code == 200 and res.json()['version'] == 2


@pytest.mark.asyncio
async def test_expect_version_어긋나면_409(client, tenant_id, fake_queue, blob_tmp, fake_embed):
    """화면에서 v1을 봤지만 그 사이 남이 v2를 올렸다 → 조용히 대체하지 않고 409."""
    v1 = await _upload(client, '환불정책.md', MD)
    await index_pending_document(v1['document_id'])
    v2 = await _upload(client, '환불정책.md', MD.replace(b'14', b'30'))
    await index_pending_document(v2['document_id'])

    res = await _post(client, '환불정책.md', MD, expect_version=1)
    assert res.status_code == 409
    assert res.json()['current_version'] == 2      # 다시 물어볼 때 쓸 값


@pytest.mark.asyncio
async def test_expect_version_미전송이면_검사하지_않는다(client, tenant_id, fake_queue, blob_tmp, fake_embed):
    """하위호환 — 구버전 FE는 파라미터 없이 부르고, 그 경우 기존처럼 대체된다."""
    v1 = await _upload(client, '환불정책.md', MD)
    await index_pending_document(v1['document_id'])

    res = await _post(client, '환불정책.md', MD)
    assert res.status_code == 200 and res.json()['version'] == 2


@pytest.mark.asyncio
async def test_expect_version_판정은_exists와_같은_기준(client, tenant_id, fake_queue, blob_tmp, fake_embed):
    """소프트 삭제된 이름은 '없음'(0)으로 본다 — exists가 그렇게 답하므로 기준이 같아야 한다."""
    v1 = await _upload(client, '환불정책.md', MD)
    await index_pending_document(v1['document_id'])
    assert (await client.delete(f"/kms/documents/{v1['document_id']}")).status_code == 204

    assert (await client.get('/kms/documents/exists',
                             params={'filename': '환불정책.md'})).json()['exists'] is False
    res = await _post(client, '환불정책.md', MD, expect_version=0)
    assert res.status_code == 200                  # 기준이 어긋나면 여기서 409가 난다


@pytest.mark.asyncio
async def test_expect_version_은_테넌트별로_판정(client, tenant_id, other_tenant_id, fake_queue, blob_tmp):
    """남의 테넌트에 같은 이름이 있어도 내 쪽은 '없음'이다."""
    res = await client.post('/kms/documents',
                            files={'file': ('환불정책.md', MD, 'text/markdown')},
                            headers={'X-Tenant-Id': other_tenant_id})
    assert res.status_code == 200

    assert (await _post(client, '환불정책.md', MD, expect_version=0)).status_code == 200


@pytest.mark.asyncio
async def test_409면_blob이_남지_않는다(client, tenant_id, fake_queue, blob_tmp):
    """blob은 insert 전에 쓰므로, 409로 빠질 때 지우지 않으면 참조 없는 파일이 쌓인다."""
    v1 = await _upload(client, '환불정책.md', MD)
    await index_pending_document(v1['document_id'])
    before = sorted((blob_tmp / tenant_id).iterdir())

    assert (await _post(client, '환불정책.md', MD, expect_version=0)).status_code == 409
    assert sorted((blob_tmp / tenant_id).iterdir()) == before


@pytest.mark.asyncio
async def test_동시_삽입은_유니크_인덱스가_막고_409(client, tenant_id, fake_queue, blob_tmp, monkeypatch):
    """조회를 함께 통과한 두 요청 중 하나는 UNIQUE(tenant_id, filename, version)에 걸린다.

    실제 동시 요청 대신 insert가 이미 있는 (filename, version)을 쓰도록 만들어 그 경로만 본다.
    이전에는 이 위반이 그대로 터져 500이었다.
    """
    v1 = await _upload(client, '환불정책.md', MD)
    await index_pending_document(v1['document_id'])

    import routers.documents as rd
    real = rd.handle_upload

    async def _collide(session, t, filename, mime, blob_path, description=None):
        doc = await real(session, t, filename, mime, blob_path, description=description)
        doc.version = 1          # 남이 방금 v1을 넣은 것과 같은 결과
        return doc

    monkeypatch.setattr(rd, 'handle_upload', _collide)

    res = await _post(client, '환불정책.md', MD)
    assert res.status_code == 409
    assert res.json()['current_version'] == 1
    assert sorted((blob_tmp / tenant_id).iterdir())      # 남은 건 v1의 blob 하나뿐

