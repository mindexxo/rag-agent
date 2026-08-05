"""폴더 '참조 설명'이 리랭커 입력에만 들어가는지 검증 (2026-08-05).

설계 전제:
- 임베딩 입력(chunks.dense)에는 폴더가 **안 들어간다** → 설명을 고쳐도 재색인 불필요.
  이 전제가 깨지면 저장 벡터와 질의 벡터의 형태가 어긋나 검색이 조용히 나빠진다.
- 리랭커 입력에만 들어간다 → 폴더 '사이' 순위에 개입.
"""
import inspect

import pytest

from rag.index_text import build_index_text
from rag.reranker import _rerank_text
from rag.retriever import RetrievedChunk


def _chunk(**kw) -> RetrievedChunk:
    base = dict(chunk_id=1, document_id=1, text='본문 내용', heading_path=['3. 배송비'],
                page=None, rrf_score=0.0, branches=['dense'],
                filename='환불반품정책.pdf', version=1)
    return RetrievedChunk(**{**base, **kw})


class TestBuildIndexText:
    def test_폴더_미전달이_기본값(self):
        """임베딩 호출부(documents.py·ingestion.py)는 folder를 안 넘긴다 — 기본값이 None이어야
        3인자 호출이 그대로 유지되고, 저장 벡터에 폴더가 섞이지 않는다."""
        sig = inspect.signature(build_index_text)
        assert sig.parameters['folder'].default is None

    def test_폴더_없으면_기존_형태_유지(self):
        assert build_index_text('본문', '환불반품정책.pdf', ['3. 배송비']) == '환불반품정책 > 3. 배송비\n본문'

    def test_폴더는_맨_앞에_붙는다(self):
        out = build_index_text('본문', '환불반품정책.pdf', ['3. 배송비'], folder='환불·반품 규정')
        assert out == '환불·반품 규정 > 환불반품정책 > 3. 배송비\n본문'

    def test_폴더만_있어도_붙는다(self):
        assert build_index_text('본문', None, [], folder='환불·반품 규정') == '환불·반품 규정\n본문'

    def test_붙일_게_없으면_본문_그대로(self):
        assert build_index_text('본문', None, [], folder=None) == '본문'

    def test_본문_첫줄과_같은_폴더명은_중복_제거(self):
        """기존 중복 제거 규칙이 폴더에도 적용되는지 (같은 말이 두 번 들어가면 헤딩 비중이 부풀음)."""
        assert build_index_text('환불·반품 규정\n본문', None, [], folder='환불·반품 규정') == '환불·반품 규정\n본문'


class TestRerankText:
    def test_폴더명과_설명이_합쳐진다(self):
        out = _rerank_text(_chunk(folder_name='환불·반품 규정',
                                  folder_description='반품 조건·비용을 묻는 상황'))
        assert out.startswith('환불·반품 규정 — 반품 조건·비용을 묻는 상황 > 환불반품정책 > 3. 배송비')

    def test_설명만_없으면_폴더명만(self):
        out = _rerank_text(_chunk(folder_name='환불·반품 규정'))
        assert out.startswith('환불·반품 규정 > 환불반품정책')

    def test_미분류_문서는_기존_형태(self):
        """folder_id NULL — outerjoin으로 None이 와도 형태가 변하지 않아야 한다."""
        assert _rerank_text(_chunk()) == '환불반품정책 > 3. 배송비\n본문 내용'

    def test_FAQ_청크는_폴더도_안_붙는다(self):
        """FAQ는 본문이 자기설명적이라 prefix 자체를 안 붙인다 (인제스션과 형태 일치)."""
        c = _chunk(faq_id=7, document_id=None, filename='FAQ',
                   folder_name='있어도무시', folder_description='설명')
        assert _rerank_text(c) == '본문 내용'


class TestFolderApi:
    @pytest.mark.asyncio
    async def test_설명_생성_조회_수정(self, client):
        created = (await client.post('/kms/folders',
                                     json={'name': '환불·반품', 'description': '반품 조건을 묻는 상황'})).json()
        assert created['description'] == '반품 조건을 묻는 상황'

        listed = (await client.get('/kms/folders')).json()
        assert any(f['id'] == created['id'] and f['description'] == '반품 조건을 묻는 상황' for f in listed)

        patched = (await client.patch(f"/kms/folders/{created['id']}",
                                      json={'description': '  바뀐 설명  '})).json()
        assert patched['description'] == '바뀐 설명'          # 앞뒤 공백 정리

        cleared = (await client.patch(f"/kms/folders/{created['id']}",
                                      json={'description': '   '})).json()
        assert cleared['description'] is None                 # 공백만 보내면 해제

        await client.delete(f"/kms/folders/{created['id']}")

    @pytest.mark.asyncio
    async def test_설명_길이_상한(self, client):
        """리랭커 입력에 후보 수만큼 누적되므로 길이를 막는다."""
        res = await client.post('/kms/folders', json={'name': '긴설명', 'description': 'x' * 201})
        assert res.status_code == 422


class TestFolderCacheInvalidation:
    """폴더 설정이 바뀌면 그 폴더 문서를 근거로 만든 답변 캐시를 버려야 한다.

    설정 변경은 검색 결과(참조 on/off)나 순위(설명 → 리랭커 입력)를 바꾸므로,
    옛 설정으로 만든 답변이 캐시에서 그대로 나오면 변경이 사용자에게 반영되지 않는다.
    """

    @staticmethod
    async def _setup(client, tenant_id, desc=None):
        from database import AsyncSessionLocal
        from rag.cache import AnswerCache
        from rag.models import Document
        from schemas.kms import SourceCitation

        body = {'name': f'캐시검증{desc or ""}'}
        if desc:
            body['description'] = desc
        folder = (await client.post('/kms/folders', json=body)).json()

        async with AsyncSessionLocal() as s:
            doc = Document(tenant_id=tenant_id, filename='정책.pdf', mime='application/pdf',
                           blob_path='blob://x.pdf', version=1, is_active=True, status='ready',
                           folder_id=folder['id'])
            s.add(doc)
            await s.flush()
            doc_id = doc.id
            await AnswerCache().set(s, tenant_id, '배송비 얼마예요', '3천원입니다',
                                    [SourceCitation(document_id=doc_id, filename='정책.pdf', version=1)],
                                    [doc_id])
            await s.commit()
        return folder, doc_id

    @staticmethod
    async def _cache_rows(tenant_id) -> int:
        from database import AsyncSessionLocal
        from rag.models import AnswerCache as Row
        from sqlalchemy import func, select
        async with AsyncSessionLocal() as s:
            return (await s.execute(
                select(func.count()).select_from(Row).where(Row.tenant_id == tenant_id))).scalar()

    @pytest.mark.asyncio
    async def test_참조_off로_바꾸면_캐시_무효화(self, client, tenant_id, fake_embed):
        folder, _ = await self._setup(client, tenant_id)
        assert await self._cache_rows(tenant_id) == 1

        await client.patch(f"/kms/folders/{folder['id']}", json={'is_searchable': False})
        assert await self._cache_rows(tenant_id) == 0

    @pytest.mark.asyncio
    async def test_설명을_바꾸면_캐시_무효화(self, client, tenant_id, fake_embed):
        """설명은 리랭커 입력에 들어가 순위를 바꾼다 → 옛 순위로 만든 답변은 버린다."""
        folder, _ = await self._setup(client, tenant_id, desc='기존 설명')
        assert await self._cache_rows(tenant_id) == 1

        await client.patch(f"/kms/folders/{folder['id']}", json={'description': '바뀐 설명'})
        assert await self._cache_rows(tenant_id) == 0

    @pytest.mark.asyncio
    async def test_같은_설명_재전송은_캐시_유지(self, client, tenant_id, fake_embed):
        """값이 안 바뀌었는데 캐시를 날리면 무효화가 과해진다 (FE가 폼 전체를 PATCH하는 경우)."""
        folder, _ = await self._setup(client, tenant_id, desc='그대로')
        assert await self._cache_rows(tenant_id) == 1

        await client.patch(f"/kms/folders/{folder['id']}",
                           json={'name': '이름만변경', 'description': '그대로'})
        assert await self._cache_rows(tenant_id) == 1
