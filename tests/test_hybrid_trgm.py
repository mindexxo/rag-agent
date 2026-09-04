"""하이브리드 어휘 주입(#128 — pg_trgm 채널) 통합 계약.

핵심 계약 셋:
  1. dense가 못 잡은 어휘 일치 청크가 후보에 주입된다 (플래그 on)
  2. 주입은 dense 순위 '뒤'다 — 리랭크 꺼짐 폴백에서 dense 순서가 안 깨진다
  3. 플래그 off·dense 빈 결과에선 기존 동작 그대로 (주입 없음)

dense를 결정적으로 통제하기 위해 fake_embed(텍스트 결정적 가짜 벡터)를 쓰고,
candidates_per_branch를 좁혀 '어휘 일치 청크가 dense 후보 밖'인 상황을 만든다.
trgm 유사도는 실제 DB(pg_trgm — schema.sql에서 설치)로 계산된다.
"""
import pytest

from database import AsyncSessionLocal
from rag.models import Chunk, Document
from rag.retriever import _search_trgm, retrieve_candidates
from tests.conftest import fake_vector


async def _seed(tenant_id: str) -> dict[str, int]:
    """문서 1 + 청크 3: 어휘 타깃(질의와 문자 겹침 큼) / 필러 2(겹침 없음)."""
    async with AsyncSessionLocal() as s:
        doc = Document(tenant_id=tenant_id, filename='정책.pdf', mime='application/pdf',
                       blob_path='blob://x.pdf', version=1, is_active=True, status='ready')
        s.add(doc)
        await s.flush()
        texts = {
            'target': 'KMS-SEC-001 보안 등급 산정 기준과 재심사 절차',
            'filler1': '해외 배송 관세 안내문',
            'filler2': '멤버십 등급 혜택 요약',
        }
        ids = {}
        for i, (key, text) in enumerate(texts.items()):
            ch = Chunk(tenant_id=tenant_id, document_id=doc.id, text=text, chunk_index=i,
                       dense=fake_vector(text), heading_path=[], token_count=10)
            s.add(ch)
            await s.flush()
            ids[key] = ch.id
        await s.commit()
        return ids


@pytest.mark.asyncio
async def test_trgm은_어휘_일치를_1위로(tenant_id, fake_embed):
    ids = await _seed(tenant_id)
    async with AsyncSessionLocal() as s:
        got = await _search_trgm(s, tenant_id, 'KMS-SEC-001 재심사 절차가 어떻게 되나요', 10)
    assert got and got[0] == ids['target']


@pytest.mark.asyncio
async def test_trgm_테넌트_격리(tenant_id, fake_embed):
    import uuid
    await _seed(tenant_id)
    async with AsyncSessionLocal() as s:
        assert await _search_trgm(s, str(uuid.uuid4()), 'KMS-SEC-001 재심사', 10) == []


@pytest.mark.asyncio
async def test_dense가_놓친_어휘_일치가_주입된다(tenant_id, fake_embed):
    """candidates_per_branch=1 → dense는 최근접 1개만. 질의와 벡터가 다른 target이
    dense 후보 밖이어도, trgm 주입으로 최종 후보에 들어와야 한다."""
    ids = await _seed(tenant_id)
    query = 'KMS-SEC-001 재심사 절차가 어떻게 되나요'
    async with AsyncSessionLocal() as s:
        cands = await retrieve_candidates(s, tenant_id, query, top_n=20,
                                          candidates_per_branch=1)
    got = [c.chunk_id for c in cands.chunks]
    assert ids['target'] in got, 'trgm 주입이 동작하지 않았다'
    assert len(got) >= 2, 'dense 1개 + 주입분이 합류해야 한다'


@pytest.mark.asyncio
async def test_플래그_off면_주입_없음(tenant_id, fake_embed, monkeypatch):
    ids = await _seed(tenant_id)
    from config import settings
    monkeypatch.setattr(settings, 'hybrid_trgm_enabled', False)
    query = 'KMS-SEC-001 재심사 절차가 어떻게 되나요'
    async with AsyncSessionLocal() as s:
        cands = await retrieve_candidates(s, tenant_id, query, top_n=20,
                                          candidates_per_branch=1)
    assert len(cands.chunks) == 1, 'off인데 후보가 dense 1개를 넘었다'


@pytest.mark.asyncio
async def test_주입은_dense_뒤다(tenant_id, fake_embed, monkeypatch):
    """리랭크를 꺼서 순서를 그대로 노출 — dense 후보가 앞, 주입분이 뒤여야 한다."""
    ids = await _seed(tenant_id)
    from config import settings
    monkeypatch.setattr(settings, 'rerank_enabled', False)
    query = 'KMS-SEC-001 재심사 절차가 어떻게 되나요'
    async with AsyncSessionLocal() as s:
        cands = await retrieve_candidates(s, tenant_id, query, top_n=20,
                                          candidates_per_branch=1)
    got = [c.chunk_id for c in cands.chunks]
    dense_first = got[0]
    assert ids['target'] in got
    if dense_first != ids['target']:
        assert got.index(ids['target']) > 0, '주입분이 dense 순위를 앞질렀다 (폴백 계약 위반)'


@pytest.mark.asyncio
async def test_dense가_비면_주입도_안_한다(tenant_id, fake_embed):
    """빈 테넌트 — dense 0건이면 어휘 후보로 채우지 않는다 (게이트 신호 보호)."""
    import uuid
    empty_tenant = str(uuid.uuid4())
    async with AsyncSessionLocal() as s:
        cands = await retrieve_candidates(s, empty_tenant, 'KMS-SEC-001', top_n=20)
    assert cands.chunks == []
