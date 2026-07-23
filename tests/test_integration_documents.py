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


class _FakePool:
    def __init__(self, jobs: list):
        self._jobs = jobs

    async def enqueue_job(self, name: str, *args) -> None:
        self._jobs.append((name, *args))

    async def aclose(self) -> None:
        pass


@pytest.fixture
def fake_queue(monkeypatch):
    """arq enqueue를 기록만 하는 가짜로 — 실제 Redis 큐에 잡이 쌓이지 않게."""
    jobs: list = []

    async def _create_pool(*a, **kw):
        return _FakePool(jobs)

    import routers.documents as rd
    monkeypatch.setattr(rd, 'create_pool', _create_pool)
    return jobs


@pytest.fixture
def blob_tmp(monkeypatch, tmp_path):
    """blob 저장소를 테스트 임시 디렉터리로 — 실제 blob 디렉터리 오염 방지."""
    from config import settings
    monkeypatch.setattr(settings, 'blob_storage_dir', str(tmp_path))
    return tmp_path


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
async def test_같은_내용_재업로드는_dedupe(client, tenant_id, fake_queue, blob_tmp):
    v1 = await _upload(client, '환불정책.md', MD)
    await index_pending_document(v1['document_id'])
    n_jobs = len(fake_queue)

    again = await _upload(client, '환불정책.md', MD)             # 동일 sha
    assert again['document_id'] == v1['document_id']                               # 기존 ready 재사용
    assert len(fake_queue) == n_jobs                             # enqueue 안 함


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
