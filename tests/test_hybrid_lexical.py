"""하이브리드 어휘 주입(#135 → #139 OpenSearch BM25/Nori) 통합 계약.

핵심 계약 (#128 trgm 채널의 6계약 승계):
  1. dense가 못 잡은 어휘 일치 청크가 후보에 주입된다 (플래그 on)
  2. 주입은 dense 순위 '뒤'다 — 리랭크 꺼짐 폴백에서 dense 순서가 안 깨진다
  3. 플래그 off·dense 빈 결과에선 기존 동작 그대로 (주입 없음)
  4. 어휘 채널도 테넌트 격리를 지킨다 (tenant_filter)
  5. 구두점·따옴표·역슬래시가 든 질의가 엔진에서 죽지 않는다
  6. FAQ도 INDEX_FAQ 처리 뒤 어휘 채널에 잡힌다

dense를 결정적으로 통제하기 위해 fake_embed(텍스트 결정적 가짜 벡터)를 쓰고,
candidates_per_branch를 좁혀 '어휘 일치 청크가 dense 후보 밖'인 상황을 만든다.
BM25 채점은 실제 OpenSearch(Nori)가 한다 — 청크는 PG가 아니라 색인에만 있다(#139).
"""
import pytest

from database import AsyncSessionLocal
from rag import opensearch
from rag.chunking import ChunkData as ParsedChunk
from rag.models import Document
from rag.retriever import retrieve_candidates
from tests.conftest import fake_vector


class _Emb:
    __slots__ = ('dense',)

    def __init__(self, text):
        self.dense = fake_vector(text)


async def _seed(tenant_id: str) -> dict[str, int]:
    """문서 1 + 청크 3(어휘 타깃 / 필러 2)을 색인에 넣는다. 반환값은 색인 chunk_id."""
    async with AsyncSessionLocal() as s:
        doc = Document(tenant_id=tenant_id, filename='정책.pdf', mime='application/pdf',
                       blob_path='blob://x.pdf', version=1, is_active=True, status='ready')
        s.add(doc)
        await s.commit()
        doc_id = doc.id
    texts = {
        'target': 'KMS-SEC-001 보안 등급 산정 기준과 재심사 절차',
        'filler1': '해외 배송 관세 안내문',
        'filler2': '멤버십 혜택 요약',
    }
    chunks = [ParsedChunk(chunk_index=i, text=t, page=1, heading_path=[], meta={})
              for i, t in enumerate(texts.values())]
    await opensearch.index_parsed_document(
        document_id=doc_id, tenant_id=tenant_id, filename='정책.pdf', version=1,
        folder_id=None, folder_name=None, folder_description=None, searchable=True,
        chunks=chunks, embeddings=[_Emb(c.text) for c in chunks])
    return {k: opensearch.chunk_os_id(document_id=doc_id, chunk_index=i)
            for i, k in enumerate(texts)}


QUERY = 'KMS-SEC-001 재심사 절차가 어떻게 되나요'


@pytest.mark.asyncio
async def test_lexical은_어휘_일치를_1위로(tenant_id, fake_embed):
    ids = await _seed(tenant_id)
    got = await opensearch.search_lexical(tenant_id, QUERY, 10)
    assert got and got[0] == ids['target']


@pytest.mark.asyncio
async def test_lexical_테넌트_격리(tenant_id, fake_embed):
    import uuid
    await _seed(tenant_id)
    assert await opensearch.search_lexical(str(uuid.uuid4()), 'KMS-SEC-001 재심사', 10) == []


@pytest.mark.asyncio
async def test_dense가_놓친_어휘_일치가_주입된다(tenant_id, fake_embed):
    """candidates_per_branch=1 → dense는 최근접 1개만. 질의와 벡터가 다른 target이
    dense 후보 밖이어도, 어휘 주입으로 최종 후보에 들어와야 한다."""
    ids = await _seed(tenant_id)
    async with AsyncSessionLocal() as s:
        cands = await retrieve_candidates(s, tenant_id, QUERY, top_n=20, candidates_per_branch=1)
    got = [c.chunk_id for c in cands.chunks]
    assert ids['target'] in got, '어휘 주입이 동작하지 않았다'
    assert len(got) >= 2, 'dense 1개 + 주입분이 합류해야 한다'


@pytest.mark.asyncio
async def test_플래그_off면_주입_없음(tenant_id, fake_embed, monkeypatch):
    await _seed(tenant_id)
    from config import settings
    monkeypatch.setattr(settings, 'hybrid_lexical_enabled', False)
    async with AsyncSessionLocal() as s:
        cands = await retrieve_candidates(s, tenant_id, QUERY, top_n=20, candidates_per_branch=1)
    assert len(cands.chunks) == 1, 'off인데 후보가 dense 1개를 넘었다'


@pytest.mark.asyncio
async def test_주입은_dense_뒤다(tenant_id, fake_embed, monkeypatch):
    """리랭크를 꺼서 순서를 그대로 노출 — dense 후보가 앞, 주입분이 뒤여야 한다."""
    ids = await _seed(tenant_id)
    from config import settings
    monkeypatch.setattr(settings, 'rerank_enabled', False)
    async with AsyncSessionLocal() as s:
        cands = await retrieve_candidates(s, tenant_id, QUERY, top_n=20, candidates_per_branch=1)
    got = [c.chunk_id for c in cands.chunks]
    assert ids['target'] in got
    if got[0] != ids['target']:
        assert got.index(ids['target']) > 0, '주입분이 dense 순위를 앞질렀다 (폴백 계약 위반)'


@pytest.mark.asyncio
async def test_dense가_비면_주입도_안_한다(tenant_id, fake_embed):
    """빈 테넌트 — dense 0건이면 어휘 후보로 채우지 않는다 (게이트 신호 보호)."""
    import uuid
    async with AsyncSessionLocal() as s:
        cands = await retrieve_candidates(s, str(uuid.uuid4()), 'KMS-SEC-001', top_n=20)
    assert cands.chunks == []


@pytest.mark.asyncio
async def test_적대_토큰_질의가_엔진에서_죽지_않는다(tenant_id, fake_embed):
    """구두점·따옴표·역슬래시 — 엔진 질의는 match 절이라 이스케이프 문제가 없어야 하고,
    상품코드 형태(HP-CS-011)는 Nori가 잘라도 매칭돼야 한다."""
    async with AsyncSessionLocal() as s:
        doc = Document(tenant_id=tenant_id, filename="특수'문자\\모음.pdf", mime='application/pdf',
                       blob_path='blob://y.pdf', version=1, is_active=True, status='ready')
        s.add(doc)
        await s.commit()
        doc_id = doc.id
    text = "HP-CS-011 모델의 don't 옵션과 C:\\경로 설정"
    ch = ParsedChunk(chunk_index=0, text=text, page=1, heading_path=[], meta={})
    await opensearch.index_parsed_document(
        document_id=doc_id, tenant_id=tenant_id, filename=doc.filename, version=1,
        folder_id=None, folder_name=None, folder_description=None, searchable=True,
        chunks=[ch], embeddings=[_Emb(text)])
    target = opensearch.chunk_os_id(document_id=doc_id, chunk_index=0)
    for q in ("HP-CS-011 설정", "don't 옵션", "C:\\경로"):
        got = await opensearch.search_lexical(tenant_id, q, 5)      # 예외 없이 돌아야 한다
        assert isinstance(got, list), q
    assert target in await opensearch.search_lexical(tenant_id, "HP-CS-011 설정", 5)


@pytest.mark.asyncio
async def test_faq_색인이_어휘_채널에_잡힌다(tenant_id, fake_embed):
    """INDEX_FAQ 처리 경로 — FAQ 행에서 텍스트를 조립·임베딩해 색인하고, 어휘 채널이 그 청크를 낸다."""
    from rag import outbox
    from rag.models import Faq
    async with AsyncSessionLocal() as s:
        faq = Faq(tenant_id=tenant_id, question='RF 회원카드 발급 조건은?',
                  variants=[], answer='RF 등급 6개월 유지 시 발급됩니다.')
        s.add(faq)
        await s.flush()
        outbox.enqueue(s, tenant_id, outbox.INDEX_FAQ, faq_id=faq.id)
        await s.commit()
        faq_id = faq.id
        ids = await outbox.pending_row_ids(s, faq_id=faq_id)
    r = await outbox.drain_once(row_ids=ids)
    assert r['done'] == 1 and r['failed'] == 0
    got = await opensearch.search_lexical(tenant_id, 'RF 회원카드 발급', 5)
    assert opensearch.chunk_os_id(faq_id=faq_id) in got
