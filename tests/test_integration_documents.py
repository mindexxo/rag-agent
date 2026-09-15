"""D-4: 문서 수명주기 통합 테스트.

업로드(API)→인덱싱(그 문서의 outbox 행만 drain — 워커 프로세스 없이 핸들러 경로 검증)→
버전 엎어치기(supersede)→dedupe→실패 경로(drain이 failed 확정)→소프트 삭제.
업로드는 arq를 만지지 않는다 — 문서와 색인 대기열 행이 같은 트랜잭션에 커밋된다 (#139 outbox).
"""
import json
from datetime import timedelta

import pytest
from sqlalchemy import delete as sql_delete, func, select

from database import AsyncSessionLocal
from rag import cache
from rag import outbox
from rag.models import SearchIndexOutbox
from tests.conftest import indexed_chunk_texts, ingest
from rag.models import AnswerCache as AnswerCacheRow, Document


async def _upload(client, filename: str, content: bytes, mime='text/markdown', headers=None,
                  extra_parts=None) -> dict:
    files = {'file': (filename, content, mime), **(extra_parts or {})}
    res = await client.post('/kms/documents', files=files, headers=headers)
    assert res.status_code == 200, res.text
    return res.json()


def _doc_data(**fields) -> dict:
    """document-data 파트 (#165) — 브라우저가 Blob으로 실을 때와 같은 모양.
    filename이 붙어 서버에는 UploadFile로 도착한다."""
    return {'document-data': ('blob', json.dumps(fields), 'application/json')}


async def _get_doc(doc_id: int) -> Document:
    async with AsyncSessionLocal() as session:
        return await session.get(Document, doc_id)


async def _chunk_texts(doc_id: int) -> list[str]:
    return await indexed_chunk_texts(doc_id)        # 청크는 색인에만 있다(#139)


MD = '# 환불 정책\n\n## 1. 기간\n\n단순변심 반품은 14일 이내 신청한다.\n'.encode()


@pytest.mark.asyncio
async def test_업로드_인덱싱_ready_승격(client, tenant_id, fake_queue, blob_tmp):
    body = await _upload(client, '환불정책.md', MD)
    assert body['status'] == 'pending'                       # 업로드 직후
    async with AsyncSessionLocal() as s:                                 # 대기열 행이 문서와 함께 커밋됨 (#139)
        rows = await outbox.pending_row_ids(s, document_id=body['document_id'])
    assert len(rows) == 1

    await ingest(body['document_id'])                 # 워커 잡 직접 실행

    doc = await _get_doc(body['document_id'])
    assert doc.status == 'ready' and doc.is_active is True
    texts = await _chunk_texts(body['document_id'])
    assert texts and any('14일' in t for t in texts)
    assert doc.char_count == sum(len(t) for t in texts)


@pytest.mark.asyncio
async def test_같은_파일명_재업로드는_엎어치기(client, tenant_id, fake_queue, blob_tmp, fake_embed):
    v1 = await _upload(client, '환불정책.md', MD)
    await ingest(v1['document_id'])

    # v1을 근거로 만든 캐시가 있다고 가정
    async with AsyncSessionLocal() as session:
        await cache.save_answer(session, tenant_id, '반품 기간', '14일', [], [v1['document_id']])
        await session.commit()

    v2 = await _upload(client, '환불정책.md', MD.replace(b'14', b'30'))
    assert v2['document_id'] != v1['document_id'] and v2['version'] == 2
    # 빈 창 없음(E2E #6): v2가 처리되기 전엔 v1이 그대로 검색된다 — 옛 청크를 먼저 지우지 않는다.
    assert (await _get_doc(v1['document_id'])).is_active is True
    assert any('14일' in t for t in await _chunk_texts(v1['document_id']))
    await ingest(v2['document_id'])

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
    await ingest(v1['document_id'])

    again = await _upload(client, '환불정책.md', MD)             # 내용까지 동일
    assert again['document_id'] != v1['document_id']             # 재사용하지 않는다
    assert again['version'] == v1['version'] + 1
    async with AsyncSessionLocal() as s:                         # 인덱싱이 다시 대기열에 오른다 (#139)
        assert len(await outbox.pending_row_ids(s, document_id=again['document_id'])) == 1


@pytest.mark.asyncio
async def test_인덱싱_예외시_failed_기록(client, tenant_id, fake_queue, blob_tmp, monkeypatch):
    body = await _upload(client, '환불정책.md', MD)

    import rag.documents as rd

    async def _boom(texts):
        raise RuntimeError('임베딩 서버 폭발')

    monkeypatch.setattr(rd, 'embed_texts', _boom)
    monkeypatch.setattr(outbox, 'MAX_ATTEMPTS', 1)     # 운영은 5회 — 테스트는 한 번에 확정
    r = await ingest(body['document_id'])              # 핸들러는 예외를 올리고, drain이 failed를 찍는다 (#139)
    assert r == {'done': 0, 'failed': 1}

    doc = await _get_doc(body['document_id'])
    assert doc.status == 'failed'
    assert '임베딩 서버 폭발' in doc.status_reason               # 재업로드 판단 근거가 남는다
    async with AsyncSessionLocal() as s:
        row = (await s.execute(select(SearchIndexOutbox)
                               .where(SearchIndexOutbox.payload['document_id'].as_integer()
                                      == body['document_id']))).scalars().one()
    assert row.status == 'failed' and row.attempts == 1


@pytest.mark.asyncio
async def test_빈_파일은_failed_유령_ready_방지(client, tenant_id, fake_queue, blob_tmp, monkeypatch):
    monkeypatch.setattr(outbox, 'MAX_ATTEMPTS', 1)
    body = await _upload(client, '빈문서.md', b'')
    r = await ingest(body['document_id'])                        # 핸들러가 ValueError → drain이 failed 확정
    assert r['failed'] == 1
    doc = await _get_doc(body['document_id'])
    assert doc.status == 'failed'                                # 청크 0개 → ready 승격 금지 (C2)
    assert '추출된 텍스트가 없습니다' in doc.status_reason


@pytest.mark.asyncio
async def test_cp949_txt_안깨지고_인덱싱(client, tenant_id, fake_queue, blob_tmp):
    content = '배송비는 삼천원입니다. 도서산간은 오천원 추가.'.encode('cp949')
    body = await _upload(client, '공지.txt', content, mime='text/plain')
    await ingest(body['document_id'])

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
        await ingest(b_doc['document_id'])

        # 테넌트 A가 같은 파일명 업로드 → 인덱싱 (supersede 실행)
        a_doc = await _upload(client, '환불정책.md', MD)
        await ingest(a_doc['document_id'])

        b_after = await _get_doc(b_doc['document_id'])
        assert b_after.status == 'ready' and b_after.is_active is True   # B는 무사해야 함
        assert await _chunk_texts(b_doc['document_id']) != []
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
    body = await _upload(client, '환불정책.md', MD)
    await ingest(body['document_id'])
    doc = await _get_doc(body['document_id'])
    assert doc.status == 'ready'
    before = await _chunk_texts(doc.id)
    async with AsyncSessionLocal() as s:
        outbox.enqueue(s, tenant_id, outbox.INDEX_DOCUMENT, document_id=doc.id)   # 중복 등재
        await s.commit()

    r = await ingest(body['document_id'])
    assert r == {'done': 1, 'failed': 0}
    async with AsyncSessionLocal() as s:
        left = await outbox.pending_row_ids(s, document_id=doc.id)
    assert await _chunk_texts(doc.id) == before                  # 재색인 안 함
    assert left == []                                            # 행은 done


@pytest.mark.asyncio
async def test_소프트_삭제_청크와_캐시_제거_row_보존(client, tenant_id, fake_queue, blob_tmp, fake_embed):
    body = await _upload(client, '환불정책.md', MD)
    await ingest(body['document_id'])
    async with AsyncSessionLocal() as session:
        await cache.save_answer(session, tenant_id, '반품 기간', '14일', [], [body['document_id']])
        await session.commit()

    res = await client.delete(f"/kms/documents/{body['document_id']}")
    assert res.status_code == 204

    doc = await _get_doc(body['document_id'])
    assert doc is not None                                       # row 보존 (과거 인용 다운로드용)
    assert doc.status == 'deleted' and doc.is_active is False
    assert await _chunk_texts(body['document_id']) != []                  # drain 전 — 가이드대로 아직 색인에 있다 (E2E #4)
    await ingest(body['document_id'])                                     # DROP_DOCUMENTS 행 처리
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
    await ingest(v1['document_id'])

    body = (await client.get('/kms/documents/exists', params={'filename': '환불정책.md'})).json()
    assert body['exists'] is True
    assert body['document_id'] == v1['document_id']
    assert body['version'] == 1                      # 업로드하면 v2가 된다는 안내용
    assert body['status'] == 'ready'


@pytest.mark.asyncio
async def test_exists_는_최신_버전을_본다(client, tenant_id, fake_queue, blob_tmp):
    v1 = await _upload(client, '환불정책.md', MD)
    await ingest(v1['document_id'])
    v2 = await _upload(client, '환불정책.md', MD.replace(b'14', b'30'))
    await ingest(v2['document_id'])

    body = (await client.get('/kms/documents/exists', params={'filename': '환불정책.md'})).json()
    assert body['document_id'] == v2['document_id']  # supersede된 v1이 아니라 살아 있는 v2
    assert body['version'] == 2


@pytest.mark.asyncio
async def test_exists_는_완전일치만_본다(client, tenant_id, fake_queue, blob_tmp):
    """정책: 대소문자·공백 구분. supersede 기준과 동일해야 한다."""
    v1 = await _upload(client, '환불정책.md', MD)
    await ingest(v1['document_id'])

    for other in ('환불정책.MD', '환불정책 .md', '환불정책_최종.md'):
        body = (await client.get('/kms/documents/exists', params={'filename': other})).json()
        assert body['exists'] is False, other


@pytest.mark.asyncio
async def test_exists_는_테넌트_격리(client, tenant_id, fake_queue, blob_tmp):
    v1 = await _upload(client, '환불정책.md', MD)
    await ingest(v1['document_id'])

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
    await ingest(v1['document_id'])

    res = await _post(client, '환불정책(1).md', MD, expect_version=0)
    assert res.status_code == 409
    body = res.json()
    assert body['filename'] == '환불정책(1).md'    # 어느 이름이 걸렸는지
    assert body['current_version'] == 1            # FE가 다음 번호를 추천하는 근거
    assert isinstance(body['detail'], str)         # 공용 에러 토스트가 그대로 쓴다


@pytest.mark.asyncio
async def test_expect_version_일치하면_대체된다(client, tenant_id, fake_queue, blob_tmp, fake_embed):
    v1 = await _upload(client, '환불정책.md', MD)
    await ingest(v1['document_id'])

    res = await _post(client, '환불정책.md', MD.replace(b'14', b'30'), expect_version=1)
    assert res.status_code == 200 and res.json()['version'] == 2


@pytest.mark.asyncio
async def test_expect_version_어긋나면_409(client, tenant_id, fake_queue, blob_tmp, fake_embed):
    """화면에서 v1을 봤지만 그 사이 남이 v2를 올렸다 → 조용히 대체하지 않고 409."""
    v1 = await _upload(client, '환불정책.md', MD)
    await ingest(v1['document_id'])
    v2 = await _upload(client, '환불정책.md', MD.replace(b'14', b'30'))
    await ingest(v2['document_id'])

    res = await _post(client, '환불정책.md', MD, expect_version=1)
    assert res.status_code == 409
    assert res.json()['current_version'] == 2      # 다시 물어볼 때 쓸 값


@pytest.mark.asyncio
async def test_expect_version_미전송이면_검사하지_않는다(client, tenant_id, fake_queue, blob_tmp, fake_embed):
    """하위호환 — 구버전 FE는 파라미터 없이 부르고, 그 경우 기존처럼 대체된다."""
    v1 = await _upload(client, '환불정책.md', MD)
    await ingest(v1['document_id'])

    res = await _post(client, '환불정책.md', MD)
    assert res.status_code == 200 and res.json()['version'] == 2


@pytest.mark.asyncio
async def test_expect_version_판정은_exists와_같은_기준(client, tenant_id, fake_queue, blob_tmp, fake_embed):
    """소프트 삭제된 이름은 '없음'(0)으로 본다 — exists가 그렇게 답하므로 기준이 같아야 한다."""
    v1 = await _upload(client, '환불정책.md', MD)
    await ingest(v1['document_id'])
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
    await ingest(v1['document_id'])
    before = sorted((blob_tmp / tenant_id).iterdir())

    assert (await _post(client, '환불정책.md', MD, expect_version=0)).status_code == 409
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
    v1 = await _upload(client, '환불정책.md', MD)
    await ingest(v1['document_id'])

    import routers.documents as rd
    real = rd.handle_upload

    # 인자는 그대로 흘려보낸다 — handle_upload에 파라미터가 하나 늘 때마다 이 대역이 깨져
    # 무관한 테스트가 실패하던 것을 끊는다(#164·#165에서 연달아 겪었다).
    async def _collide(session, t, filename, blob_path, **kw):
        doc = await real(session, t, filename, blob_path, **kw)
        doc.version = 1          # 남이 방금 v1을 넣은 것과 같은 결과
        return doc

    monkeypatch.setattr(rd, 'handle_upload', _collide)

    res = await _post(client, '환불정책.md', MD)
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
    first = await _upload(client, '가정책.md', MD, headers={'X-User-Id': 'agent-a'})
    second = await _upload(client, '나정책.md', MD, headers={'X-User-Id': 'agent-b'})

    items = (await client.get('/kms/documents')).json()
    assert [d['document_id'] for d in items] == [second['document_id'], first['document_id']]
    assert items[0]['uploaded_by'] == 'agent-b'
    assert items[0]['uploaded_at']                      # datetime 직렬화 확인

    # 먼저 올린 문서의 등록일시만 미래로 옮긴다 → id 순서와 반대가 되어야 한다.
    # (aware 값을 그대로 대입한다 — 시각 컬럼 매핑이 TIMESTAMPTZ에 맞춰져 있다, rag/models.py의 Base)
    async with AsyncSessionLocal() as s:
        doc = await s.get(Document, first['document_id'])
        doc.uploaded_at = doc.uploaded_at + timedelta(days=1)
        await s.commit()

    items = (await client.get('/kms/documents')).json()
    assert [d['document_id'] for d in items] == [first['document_id'], second['document_id']]


@pytest.mark.asyncio
async def test_등록자는_X_User_Id로_기록되고_미전송이면_null(client, tenant_id, fake_queue, blob_tmp):
    """#164: 헤더가 없으면 NULL로 남긴다 — conversation의 'test-user' 폴백을 쓰지 않는다.
    그 가짜 값이 문서 관리 화면의 등록자 칸에 그대로 뜨기 때문."""
    with_user = await _upload(client, '가정책.md', MD, headers={'X-User-Id': 'agent-a'})
    without = await _upload(client, '나정책.md', MD)

    assert with_user['uploaded_by'] == 'agent-a'
    assert without['uploaded_by'] is None
    assert (await _get_doc(with_user['document_id'])).uploaded_by == 'agent-a'
    assert (await _get_doc(without['document_id'])).uploaded_by is None


@pytest.mark.asyncio
async def test_재업로드_등록자는_계승하지_않는다(client, tenant_id, fake_queue, blob_tmp):
    """#164: 문서에 건 설정(is_searchable)은 개정판에 계승되지만, 등록자는 그 버전을 올린
    사람으로 갱신된다 — 계승하면 최신 개정을 누가 했는지 화면에서 알 수 없다."""
    v1 = await _upload(client, '환불정책.md', MD, headers={'X-User-Id': 'agent-a'})
    await ingest(v1['document_id'])
    assert (await client.patch(f"/kms/documents/{v1['document_id']}",
                               json={'is_searchable': False})).status_code == 200

    v2 = await _upload(client, '환불정책.md', MD, headers={'X-User-Id': 'agent-b'})
    assert v2['version'] == 2
    assert v2['is_searchable'] is False       # 설정은 계승
    assert v2['uploaded_by'] == 'agent-b'     # 등록자는 갱신

    # 헤더 없이 올린 버전이 직전 등록자를 물려받지 않는지 — "계승하지 않는다"가 깨지는
    # 가장 흔한 형태(prev 폴백)를 직접 찍는다.
    v3 = await _upload(client, '환불정책.md', MD)
    assert v3['version'] == 3
    assert v3['uploaded_by'] is None


# ── #165 업로드 시 폴더 지정 (document-data 파트) ─────────────

async def _folder(client, name: str) -> int:
    res = await client.post('/kms/folders', json={'name': name})
    assert res.status_code == 200, res.text
    return res.json()['id']


@pytest.mark.asyncio
async def test_업로드_시_폴더_지정(client, tenant_id, fake_queue, blob_tmp):
    fid = await _folder(client, '규정집')
    body = await _upload(client, '환불정책.md', MD, extra_parts=_doc_data(folder_id=fid))
    assert body['folder_id'] == fid
    assert (await _get_doc(body['document_id'])).folder_id == fid


@pytest.mark.asyncio
async def test_document_data를_문자열_파트로_보내도_동작(client, tenant_id, fake_queue, blob_tmp):
    """브라우저는 Blob(=UploadFile)로, 다른 클라이언트는 문자열 필드로 보낼 수 있다 — 둘 다 받는다."""
    fid = await _folder(client, '규정집')
    body = await _upload(client, '환불정책.md', MD,
                         extra_parts={'document-data': (None, json.dumps({'folder_id': fid}))})
    assert body['folder_id'] == fid


@pytest.mark.asyncio
async def test_folder_id_null이면_미분류로_뗀다(client, tenant_id, fake_queue, blob_tmp):
    """null 전송 = 미분류. 값만 보면 '안 보냄'과 같아서, 보냈는지 여부로 갈린다."""
    fid = await _folder(client, '규정집')
    v1 = await _upload(client, '환불정책.md', MD, extra_parts=_doc_data(folder_id=fid))
    await ingest(v1['document_id'])

    v2 = await _upload(client, '환불정책.md', MD, extra_parts=_doc_data(folder_id=None))
    assert v2['version'] == 2
    assert v2['folder_id'] is None


@pytest.mark.asyncio
async def test_folder_id_미전송이면_직전_버전에서_계승(client, tenant_id, fake_queue, blob_tmp):
    fid = await _folder(client, '규정집')
    v1 = await _upload(client, '환불정책.md', MD, extra_parts=_doc_data(folder_id=fid))
    await ingest(v1['document_id'])

    # document-data는 보내되 folder_id 키만 없다 → 계승
    v2 = await _upload(client, '환불정책.md', MD, extra_parts=_doc_data(description='표 설명'))
    assert v2['folder_id'] == fid
    assert (await _get_doc(v2['document_id'])).description == '표 설명'   # JSON 경로도 값이 산다
    # 파트 자체를 안 보내도 계승 (옛 방식 호출)
    v3 = await _upload(client, '환불정책.md', MD)
    assert v3['folder_id'] == fid


@pytest.mark.asyncio
async def test_없는_폴더는_404_문서도_blob도_안_남는다(client, tenant_id, fake_queue, blob_tmp):
    res = await client.post('/kms/documents',
                            files={'file': ('환불정책.md', MD, 'text/markdown'),
                                   **_doc_data(folder_id=999999)})
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
                                   **_doc_data(folder_id=other_fid)})
    assert res.status_code == 404


@pytest.mark.asyncio
async def test_document_data와_평면필드_동시_전송은_400(client, tenant_id, fake_queue, blob_tmp):
    """한쪽을 조용히 무시하면 '보냈는데 안 먹는' 버그가 된다 — 명시 거절."""
    res = await client.post('/kms/documents',
                            files={'file': ('환불정책.md', MD, 'text/markdown'),
                                   **_doc_data(description='새 방식')},
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
    assert (await _get_doc(res.json()['document_id'])).description == '표 설명'


@pytest.mark.asyncio
async def test_신규_문서에_폴더를_안_주면_미분류(client, tenant_id, fake_queue, blob_tmp):
    """계승할 직전 버전이 없을 때의 기본값 — 파트를 안 보내도, 보내되 folder_id만 빼도 미분류."""
    a = await _upload(client, '환불정책.md', MD)
    b = await _upload(client, '배송정책.md', MD, extra_parts=_doc_data(description='표 설명'))
    assert a['folder_id'] is None and b['folder_id'] is None


# ── #166 문서 일괄 변경 (PATCH /kms/documents) ────────────────

async def _bulk(client, **body):
    return await client.patch('/kms/documents', json=body)


@pytest.mark.asyncio
async def test_일괄_폴더이동과_참조토글(client, tenant_id, fake_queue, blob_tmp):
    fid = await _folder(client, '규정집')
    a = await _upload(client, '환불정책.md', MD)
    b = await _upload(client, '배송정책.md', MD)

    res = await _bulk(client, document_ids=[a['document_id'], b['document_id']],
                      folder_id=fid, is_searchable=False)
    assert res.status_code == 200, res.text
    body = res.json()
    assert {d['document_id'] for d in body} == {a['document_id'], b['document_id']}
    assert all(d['folder_id'] == fid and d['is_searchable'] is False for d in body)
    assert all(d['ref_count'] is None for d in body)      # 목록 API 전용 집계
    for d in (a, b):
        doc = await _get_doc(d['document_id'])
        assert doc.folder_id == fid and doc.is_searchable is False


@pytest.mark.asyncio
async def test_일괄_folder_id_null이면_미분류(client, tenant_id, fake_queue, blob_tmp):
    fid = await _folder(client, '규정집')
    a = await _upload(client, '환불정책.md', MD, extra_parts=_doc_data(folder_id=fid))
    res = await _bulk(client, document_ids=[a['document_id']], folder_id=None)
    assert res.status_code == 200, res.text
    assert res.json()[0]['folder_id'] is None


@pytest.mark.asyncio
async def test_일괄_대상이_하나라도_어긋나면_아무것도_안_바뀐다(
        client, tenant_id, other_tenant_id, fake_queue, blob_tmp):
    """없는 id·다른 테넌트 id·삭제된 문서 — 셋 다 전체 거절이고, 성공분도 반영되지 않는다."""
    fid = await _folder(client, '규정집')
    mine = await _upload(client, '환불정책.md', MD)

    # 다른 테넌트의 문서
    other = await client.post('/kms/documents', files={'file': ('남의문서.md', MD, 'text/markdown')},
                              headers={'X-Tenant-Id': other_tenant_id})
    # 삭제된 문서
    gone = await _upload(client, '지울문서.md', MD)
    assert (await client.delete(f"/kms/documents/{gone['document_id']}")).status_code == 204

    for label, bad in (('없는 id', 999999),
                       ('다른 테넌트', other.json()['document_id']),
                       ('삭제된 문서', gone['document_id'])):
        res = await _bulk(client, document_ids=[mine['document_id'], bad], folder_id=fid)
        assert res.status_code == 404, f'{label} → {res.status_code}'
        assert (await _get_doc(mine['document_id'])).folder_id is None, f'{label}: 성공분이 반영됐다'


@pytest.mark.asyncio
async def test_일괄_실효참조가_꺼지는_문서만_캐시_무효화(client, tenant_id, fake_queue, blob_tmp):
    """문서마다 현재 폴더·스위치가 달라 on→off 전이도 제각각이다 — 일괄 판정하면 안 된다.

    off 폴더에 있던 문서는 이미 실효 off라 지울 캐시가 없고, on이던 문서만 무효화 대상이다.
    """
    off_folder = await _folder(client, '대외비')
    assert (await client.patch(f'/kms/folders/{off_folder}',
                               json={'is_searchable': False})).status_code == 200

    on_doc = await _upload(client, '환불정책.md', MD)                                   # 미분류 = 실효 on
    off_doc = await _upload(client, '배송정책.md', MD, extra_parts=_doc_data(folder_id=off_folder))

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
    a = await _upload(client, '환불정책.md', MD)
    b = await _upload(client, '배송정책.md', MD)
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
    doc = await _upload(client, '환불정책.md', MD)
    assert (await client.delete(f"/kms/documents/{doc['document_id']}")).status_code == 204
    res = await client.patch(f"/kms/documents/{doc['document_id']}", json={'is_searchable': False})
    assert res.status_code == 404


@pytest.mark.asyncio
async def test_일괄_폴더가_없거나_남의_것이면_404(client, tenant_id, other_tenant_id,
                                                fake_queue, blob_tmp):
    """대상 문서는 멀쩡해도 옮길 폴더가 어긋나면 전체 거절 — 문서도 안 바뀐다."""
    doc = await _upload(client, '환불정책.md', MD)
    other_folder = (await client.post('/kms/folders', json={'name': '남의폴더'},
                                      headers={'X-Tenant-Id': other_tenant_id})).json()['id']

    for label, fid in (('없는 폴더', 999999), ('다른 테넌트 폴더', other_folder)):
        res = await _bulk(client, document_ids=[doc['document_id']], folder_id=fid)
        assert res.status_code == 404, f'{label} → {res.status_code}'
    assert (await _get_doc(doc['document_id'])).folder_id is None


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
    doc = await _upload(client, '환불정책.md', MD)
    res = await _bulk(client, document_ids=[doc['document_id'], 999999], is_searchable=False)
    assert res.status_code == 404
    assert '999999' in res.json()['detail']
    assert str(doc['document_id']) not in res.json()['detail']   # 멀쩡한 id는 안 싣는다
