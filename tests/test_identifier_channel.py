"""식별자 채널(#238) 통합 계약 — 실제 OpenSearch에 색인하고 운영 경로(retrieve_candidates → prepare_evidence)를 탄다.

  1. 색인 경로가 식별자 사전(`identifiers`)과 헤딩 줄(`headings`)을 채운다 — 묶음 헤딩(meta) + 본문 헤딩 꼴 줄
  2. 정확 일치 질의(search_identifiers)는 '11조'에 묶음 조각을 돌려주고 '111조'·'11장' 조각은 돌려주지 않는다
  3. dense가 놓친 정확 일치 조각이 후보에 주입된다 — 리랭크 꺼짐이면 주입만, 뒤에 붙는다
  4. 리랭크가 "제11장" 조각을 위에 두어도 가산이 정답을 1위로 올리고 matched_ident를 남긴다 →
     근거 준비가 묶음에서 그 조 구간만 오린다(라벨은 그대로)
  5. 식별자 없는 질문·채널 off에선 아무것도 바뀌지 않는다 (주입·가산·오리기 전부 미발동)
  6. 테넌트 격리

문서 텍스트는 가짜다(저장소 공개). 실문서에서 관측된 **표기 꼴**("제 11 조(…)", "제 11 장 …")만 옮겼다.
dense는 fake_embed(결정적 가짜 벡터)로 통제하고 candidates_per_branch를 좁혀 '정답이 dense 후보 밖'인 상황을 만든다.
"""
import uuid

import pytest

from config import settings
from database import AsyncSessionLocal
from rag import os_index, os_search
from rag.chunking import ChunkData as ParsedChunk
from rag.models import Document
from rag.retriever import prepare_evidence, retrieve_candidates
from tests.conftest import fake_vector


class _Emb:
    __slots__ = ('dense',)

    def __init__(self, text):
        self.dense = fake_vector(text)


_PACKED = "제 9 조(고용관계)\n고용관계의 개시는 근로계약에 의한다.\n제 10 조(인사발령)\n인사발령은 공고로 행한다.\n제 11 조(사무인계 의무)\n퇴직하는 사원은 업무를 후임자에게 인계한다."
_CHAPTER = "제 111 조(고객 등에 의한 괴롭힘 방지)\n회사는 고객의 괴롭힘으로부터 사원을 보호한다."
_FILLER = "해외 배송 관세 안내문"
QUERY = "가나규칙 11조 뭐야"


async def _seed(tenant_id: str) -> dict[str, int]:
    async with AsyncSessionLocal() as s:
        doc = Document(tenant_id=tenant_id, filename='가나규칙.pdf', mime='application/pdf',
                       blob_path='blob://x.pdf', version=1, is_active=True, status='ready')
        s.add(doc)
        await s.commit()
        doc_id = doc.id
    chunks = [
        # 묶음 — 제9조만 청킹이 기록했고(meta), 제10·11조는 레이아웃 모델이 본문으로 분류한 꼴(결정 15-3)
        ParsedChunk(chunk_index=0, text=_PACKED, page=3, heading_path=['가나규칙', '제 2 장 인 사'],
                    meta={'headings': ['제 9 조(고용관계)']}),
        ParsedChunk(chunk_index=1, text=_CHAPTER, page=25, heading_path=['가나규칙', '제 11 장 괴롭힘의 예방'], meta={}),
        ParsedChunk(chunk_index=2, text=_FILLER, page=1, heading_path=[], meta={}),
    ]
    await os_index.index_parsed_document(
        document_id=doc_id, tenant_id=tenant_id, filename='가나규칙.pdf', version=1,
        folder_id=None, folder_name=None, folder_description=None, searchable=True,
        chunks=chunks, embeddings=[_Emb(c.text) for c in chunks])
    return {k: os_index.chunk_os_id(document_id=doc_id, chunk_index=i)
            for i, k in enumerate(('packed', 'chapter', 'filler'))}


@pytest.mark.asyncio
async def test_색인이_사전과_헤딩_줄을_채운다(tenant_id, fake_embed):
    ids = await _seed(tenant_id)
    cm = await os_search.fetch_chunk_map(list(ids.values()))
    packed, chapter = cm[ids['packed']], cm[ids['chapter']]
    assert packed.identifiers == ['2장', '9조', '10조', '11조']          # 제목 경로의 장 + 기록 헤딩 + 본문 헤딩 꼴 줄
    assert packed.headings == ['제 9 조(고용관계)', '제 10 조(인사발령)', '제 11 조(사무인계 의무)']
    assert chapter.identifiers == ['11장', '111조']                     # '11조'는 없다
    assert cm[ids['filler']].identifiers == [] and cm[ids['filler']].headings == []


@pytest.mark.asyncio
async def test_정확_일치_질의와_테넌트_격리(tenant_id, fake_embed):
    ids = await _seed(tenant_id)
    assert await os_search.search_identifiers(tenant_id, ['11조'], 10) == [ids['packed']]
    assert await os_search.search_identifiers(tenant_id, ['111조'], 10) == [ids['chapter']]
    assert await os_search.search_identifiers(tenant_id, ['99조'], 10) == []
    assert await os_search.search_identifiers(str(uuid.uuid4()), ['11조'], 10) == []


@pytest.mark.asyncio
async def test_dense가_놓친_정확_일치_조각이_주입된다(tenant_id, fake_embed):
    """candidates_per_branch=1 → dense는 최근접 1개. 리랭크 꺼짐(fake_embed)이라 가산은 없고 주입만 — 뒤에 붙는다."""
    ids = await _seed(tenant_id)
    async with AsyncSessionLocal() as s:
        cands = await retrieve_candidates(s, tenant_id, QUERY, top_n=20, candidates_per_branch=1)
    got = [c.chunk_id for c in cands.chunks]
    assert ids['packed'] in got, '식별자 주입이 동작하지 않았다'
    assert all(c.matched_ident is None for c in cands.chunks)        # 리랭크 점수가 없으면 가산·표시 없음


@pytest.mark.asyncio
async def test_가산이_장조_혼동을_넘고_근거는_그_조만_오린다(tenant_id, fake_embed, monkeypatch):
    """리랭커가 '제11장' 조각을 0.97, 묶음을 0.72로 채점하는 상황(실측 재현) — +0.3이면 묶음이 1위."""
    ids = await _seed(tenant_id)
    monkeypatch.setattr(settings, 'rerank_enabled', True)
    import rag.reranker
    fake_scores = {ids['chapter']: 0.97, ids['packed']: 0.72, ids['filler']: 0.05}

    async def _fake_maxpool(queries, chunks):
        rows = sorted(((c, fake_scores[c.chunk_id], 0) for c in chunks), key=lambda r: -r[1])
        return [r[0] for r in rows], [r[1] for r in rows], [r[2] for r in rows]
    monkeypatch.setattr(rag.reranker, 'rerank_maxpool', _fake_maxpool)

    async with AsyncSessionLocal() as s:
        cands = await retrieve_candidates(s, tenant_id, QUERY, top_n=20, candidates_per_branch=3)
    assert cands.chunks[0].chunk_id == ids['packed'] and cands.chunks[0].matched_ident == '11조'
    assert cands.chunks[1].chunk_id == ids['chapter'] and cands.chunks[1].matched_ident is None
    ev = prepare_evidence(cands.chunks, 5)
    assert ev[0].text == "제 11 조(사무인계 의무)\n퇴직하는 사원은 업무를 후임자에게 인계한다."   # 묶음에서 그 조만
    assert ev[0].heading_path == ['가나규칙', '제 2 장 인 사'] and ev[0].page == 3         # 라벨·쪽 불변(결정 18)
    assert ev[1].text == _CHAPTER                                                          # 다른 조각은 그대로


@pytest.mark.asyncio
async def test_식별자_없는_질문과_채널_off는_아무것도_안_바꾼다(tenant_id, fake_embed, monkeypatch):
    ids = await _seed(tenant_id)
    async with AsyncSessionLocal() as s:
        plain = await retrieve_candidates(s, tenant_id, '해외 배송 관세가 얼마인가요', top_n=20, candidates_per_branch=1)
        assert [c.chunk_id for c in plain.chunks] == [ids['filler']]            # 주입 없음 — dense 1개 그대로
        monkeypatch.setattr(settings, 'ident_channel_enabled', False)
        off = await retrieve_candidates(s, tenant_id, QUERY, top_n=20, candidates_per_branch=1)
    assert ids['packed'] not in [c.chunk_id for c in off.chunks]
    assert all(c.matched_ident is None for c in off.chunks)
