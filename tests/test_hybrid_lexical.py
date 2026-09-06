"""하이브리드 어휘 주입(#135 — FTS+BM25 채널) 통합 계약.

핵심 계약 (#128 trgm 채널의 6계약 승계 + 어휘 채널 고유 3개):
  1. dense가 못 잡은 어휘 일치 청크가 후보에 주입된다 (플래그 on)
  2. 주입은 dense 순위 '뒤'다 — 리랭크 꺼짐 폴백에서 dense 순서가 안 깨진다
  3. 플래그 off·dense 빈 결과에선 기존 동작 그대로 (주입 없음)
  4. BM25 산식이 측정 정본(eval/_hybrid_ablation.Bm25)과 순위 단위로 같다
  5. 구두점·따옴표·역슬래시 토큰이 저장→매칭 왕복에서 왜곡되지 않는다 (이스케이프 계약)
  6. 백필 전 청크(lex NULL)가 섞여 있어도 채널이 죽지 않는다

dense를 결정적으로 통제하기 위해 fake_embed(텍스트 결정적 가짜 벡터)를 쓰고,
candidates_per_branch를 좁혀 '어휘 일치 청크가 dense 후보 밖'인 상황을 만든다.
FTS 매칭·집계는 실제 DB로 계산된다.
"""
import pytest
from sqlalchemy import ARRAY, Text, cast, func

from database import AsyncSessionLocal
from rag import lexical
from rag.index_text import build_index_text
from rag.models import Chunk, Document
from rag.retriever import _search_lexical, retrieve_candidates
from tests.conftest import fake_vector


def _lex_values(text: str, filename: str | None = None,
                heading_path: list[str] | None = None) -> dict:
    """운영 인제스션과 동일한 조립(문서=프리픽스, FAQ/None=원문)으로 lex 컬럼 값 생성."""
    idx_text = text if filename is None else build_index_text(text, filename, heading_path or [])
    toks = lexical.bigrams(idx_text)
    return {
        'lex_tsv': func.array_to_tsvector(cast(lexical.tsvector_lexemes(toks), ARRAY(Text))),
        'lex_len': len(toks),
    }


async def _seed(tenant_id: str) -> dict[str, int]:
    """문서 1 + 청크 4: 어휘 타깃 / 필러 2 / 백필 전(lex NULL) 1."""
    async with AsyncSessionLocal() as s:
        doc = Document(tenant_id=tenant_id, filename='정책.pdf', mime='application/pdf',
                       blob_path='blob://x.pdf', version=1, is_active=True, status='ready')
        s.add(doc)
        await s.flush()
        texts = {
            'target': 'KMS-SEC-001 보안 등급 산정 기준과 재심사 절차',
            'filler1': '해외 배송 관세 안내문',
            'filler2': '멤버십 혜택 요약',
        }
        ids = {}
        for i, (key, text) in enumerate(texts.items()):
            ch = Chunk(tenant_id=tenant_id, document_id=doc.id, text=text, chunk_index=i,
                       dense=fake_vector(text), heading_path=[],
                       **_lex_values(text, doc.filename))
            s.add(ch)
            await s.flush()
            ids[key] = ch.id
        # 백필 전 상태 재현 — lex 컬럼 NULL. 채널은 이 청크를 못 보되 죽지 않아야 한다.
        ch = Chunk(tenant_id=tenant_id, document_id=doc.id, text='백필 전 안내문 초안',
                   chunk_index=3, dense=fake_vector('백필 전'), heading_path=[])
        s.add(ch)
        await s.flush()
        ids['unfilled'] = ch.id
        await s.commit()
        return ids


@pytest.mark.asyncio
async def test_lexical은_어휘_일치를_1위로(tenant_id, fake_embed):
    ids = await _seed(tenant_id)
    async with AsyncSessionLocal() as s:
        got = await _search_lexical(s, tenant_id, 'KMS-SEC-001 재심사 절차가 어떻게 되나요', 10)
    assert got and got[0] == ids['target']


@pytest.mark.asyncio
async def test_lexical_테넌트_격리(tenant_id, fake_embed):
    import uuid
    await _seed(tenant_id)
    async with AsyncSessionLocal() as s:
        assert await _search_lexical(s, str(uuid.uuid4()), 'KMS-SEC-001 재심사', 10) == []


@pytest.mark.asyncio
async def test_dense가_놓친_어휘_일치가_주입된다(tenant_id, fake_embed):
    """candidates_per_branch=1 → dense는 최근접 1개만. 질의와 벡터가 다른 target이
    dense 후보 밖이어도, 어휘 주입으로 최종 후보에 들어와야 한다."""
    ids = await _seed(tenant_id)
    query = 'KMS-SEC-001 재심사 절차가 어떻게 되나요'
    async with AsyncSessionLocal() as s:
        cands = await retrieve_candidates(s, tenant_id, query, top_n=20,
                                          candidates_per_branch=1)
    got = [c.chunk_id for c in cands.chunks]
    assert ids['target'] in got, '어휘 주입이 동작하지 않았다'
    assert len(got) >= 2, 'dense 1개 + 주입분이 합류해야 한다'


@pytest.mark.asyncio
async def test_플래그_off면_주입_없음(tenant_id, fake_embed, monkeypatch):
    await _seed(tenant_id)
    from config import settings
    monkeypatch.setattr(settings, 'hybrid_lexical_enabled', False)
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
    assert ids['target'] in got
    if got[0] != ids['target']:
        assert got.index(ids['target']) > 0, '주입분이 dense 순위를 앞질렀다 (폴백 계약 위반)'


@pytest.mark.asyncio
async def test_dense가_비면_주입도_안_한다(tenant_id, fake_embed):
    """빈 테넌트 — dense 0건이면 어휘 후보로 채우지 않는다 (게이트 신호 보호)."""
    import uuid
    empty_tenant = str(uuid.uuid4())
    async with AsyncSessionLocal() as s:
        cands = await retrieve_candidates(s, empty_tenant, 'KMS-SEC-001', top_n=20)
    assert cands.chunks == []


def test_bm25_산식은_측정_정본과_같다():
    """rag.lexical.bm25_rank ↔ eval/_hybrid_ablation.Bm25 — 같은 코퍼스·질의에서
    순위 동일. 산식이 갈라지면 어블레이션 눈금이 운영을 예언하지 못한다(#128 규율).
    DB 불필요한 순수 로직 대조다 (운영 SQL 경로의 값 대조는 #135 구현 시 실데이터
    5질의 완전일치로 검증 — PR 기록)."""
    from eval._hybrid_ablation import Bm25
    docs = {
        1: '정책.pdf > 보안 KMS-SEC-001 보안 등급 산정 기준과 재심사 절차',
        2: '정책.pdf > 배송 해외 배송 관세 안내문과 반품 기간 규정',
        3: '정책.pdf > 회원 멤버십 등급 혜택 요약 및 등급 산정',
        4: 'FAQ Q: 반품 기간이 며칠인가요 A: 수령 후 7일 이내입니다',
    }
    ref = Bm25(docs)
    n = len(docs)
    avgdl = sum(len(lexical.bigrams(t)) for t in docs.values()) / n
    cand_tokens = {cid: lexical.bigrams(t) for cid, t in docs.items()}
    for q in ('KMS-SEC-001 재심사 절차', '반품 기간 며칠', '등급 산정 기준', '멤버십 혜택'):
        assert lexical.bm25_rank(q, cand_tokens, n, avgdl) == ref.top(q, n), q


@pytest.mark.asyncio
async def test_적대_토큰_저장_매칭_왕복(tenant_id, fake_embed):
    """구두점·따옴표·역슬래시가 든 텍스트 — 저장(array_to_tsvector)과 질의(tsquery
    이스케이프)가 왜곡 없이 왕복해야 한다. rag/lexical.tsquery_or의 이스케이프 계약."""
    async with AsyncSessionLocal() as s:
        doc = Document(tenant_id=tenant_id, filename="특수'문자\\모음.pdf", mime='application/pdf',
                       blob_path='blob://y.pdf', version=1, is_active=True, status='ready')
        s.add(doc)
        await s.flush()
        text = "HP-CS-011 모델의 don't 옵션과 C:\\경로 설정"
        ch = Chunk(tenant_id=tenant_id, document_id=doc.id, text=text, chunk_index=0,
                   dense=fake_vector(text), heading_path=[],
                   **_lex_values(text, doc.filename))
        s.add(ch)
        await s.flush()
        target = ch.id
        await s.commit()
    async with AsyncSessionLocal() as s:
        for q in ("HP-CS-011 설정", "don't 옵션", "C:\\경로"):
            got = await _search_lexical(s, tenant_id, q, 5)
            assert target in got, f'적대 토큰 질의 실패: {q!r}'


@pytest.mark.asyncio
async def test_faq_색인이_lex를_채운다(tenant_id, fake_embed):
    """reindex_faq 경로 — FAQ 청크에 lex_tsv·lex_len이 채워지고 채널에 잡힌다."""
    from types import SimpleNamespace
    from sqlalchemy import select
    from rag.faq_indexing import reindex_faq
    from rag.models import Faq
    async with AsyncSessionLocal() as s:
        faq = Faq(tenant_id=tenant_id, question='RF 회원카드 발급 조건은?',
                  variants=[], answer='RF 등급 6개월 유지 시 발급됩니다.')
        s.add(faq)
        await s.flush()
        await reindex_faq(s, faq, SimpleNamespace(dense=fake_vector('RF 회원카드')))
        await s.commit()
        row = (await s.execute(
            select(Chunk).where(Chunk.faq_id == faq.id))).scalar_one()
        assert row.lex_len and row.lex_len > 0
        got = await _search_lexical(s, tenant_id, 'RF 회원카드 발급', 5)
        assert row.id in got
