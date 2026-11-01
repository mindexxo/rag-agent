"""같은 파일명 재업로드 확인 — 동일 파일명 조회(GET /documents/exists)와 낙관적 잠금(expect_version)."""

import pytest
from sqlalchemy import func, select

from database import AsyncSessionLocal
from rag.models import SearchIndexOutbox
from tests.conftest import ingest
from tests.helpers_documents import MD, post_upload, upload

# ===== 업로드 전 동일 파일명 확인 (GET /documents/exists) =====================


@pytest.mark.asyncio
async def test_exists_없는_파일명은_false(client, tenant_id, fake_queue, blob_tmp):
    res = await client.get('/kms/documents/exists', params={'filename': '없는문서.md'})
    assert res.status_code == 200
    assert res.json() == {'exists': False, 'document_id': None, 'version': None, 'status': None, 'uploaded_at': None}


@pytest.mark.asyncio
async def test_exists_기존_문서는_현재_버전을_알려준다(client, tenant_id, fake_queue, blob_tmp):
    v1 = await upload(client, '환불정책.md', MD)
    await ingest(v1['document_id'])

    body = (await client.get('/kms/documents/exists', params={'filename': '환불정책.md'})).json()
    assert body['exists'] is True
    assert body['document_id'] == v1['document_id']
    assert body['version'] == 1  # 업로드하면 v2가 된다는 안내용
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

    res = await client.get(
        '/kms/documents/exists', params={'filename': '환불정책.md'}, headers={'X-Tenant-Id': 'other-tenant'}
    )
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
    assert body['filename'] == '환불정책(1).md'  # 어느 이름이 걸렸는지
    assert body['current_version'] == 1  # FE가 다음 번호를 추천하는 근거
    assert isinstance(body['detail'], str)  # 공용 에러 토스트가 그대로 쓴다


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
    assert res.json()['current_version'] == 2  # 다시 물어볼 때 쓸 값


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

    assert (await client.get('/kms/documents/exists', params={'filename': '환불정책.md'})).json()['exists'] is False
    res = await post_upload(client, '환불정책.md', MD, expect_version=0)
    assert res.status_code == 200  # 기준이 어긋나면 여기서 409가 난다


@pytest.mark.asyncio
async def test_expect_version_은_테넌트별로_판정(client, tenant_id, other_tenant_id, fake_queue, blob_tmp):
    """남의 테넌트에 같은 이름이 있어도 내 쪽은 '없음'이다."""
    res = await client.post(
        '/kms/documents', files={'file': ('환불정책.md', MD, 'text/markdown')}, headers={'X-Tenant-Id': other_tenant_id}
    )
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
    # 대기열 행도 문서와 같은 트랜잭션이라 409 롤백에 함께 사라져야 한다(outbox 원자성).
    # v1의 행은 ingest로 done이 됐으니 pending은 0이어야 한다.
    async with AsyncSessionLocal() as s:
        pending = (
            await s.execute(
                select(func.count())
                .select_from(SearchIndexOutbox)
                .where(SearchIndexOutbox.tenant_id == tenant_id)
                .where(SearchIndexOutbox.status == 'pending')
            )
        ).scalar()
    assert pending == 0


@pytest.mark.asyncio
async def test_동시_삽입은_유니크_인덱스가_막고_409(client, tenant_id, fake_queue, blob_tmp, monkeypatch):
    """조회를 함께 통과한 두 요청 중 하나는 UNIQUE(tenant_id, filename, version)에 걸린다.

    실제 동시 요청 대신 insert가 이미 있는 (filename, version)을 쓰도록 만들어 그 경로만 본다.
    """
    v1 = await upload(client, '환불정책.md', MD)
    await ingest(v1['document_id'])

    import rag.upload as rd

    real = rd.handle_upload

    # 인자는 그대로 흘려보낸다 — handle_upload에 파라미터가 늘 때마다 이 대역이 깨져
    # 무관한 테스트가 실패하지 않게.
    async def _collide(session, t, filename, blob_path, **kw):
        result = await real(session, t, filename, blob_path, **kw)
        result.document.version = 1  # 남이 방금 v1을 넣은 것과 같은 결과
        return result

    monkeypatch.setattr(rd, 'handle_upload', _collide)

    res = await post_upload(client, '환불정책.md', MD)
    assert res.status_code == 409
    assert res.json()['current_version'] == 1
    assert sorted((blob_tmp / tenant_id).iterdir())  # 남은 건 v1의 blob 하나뿐
