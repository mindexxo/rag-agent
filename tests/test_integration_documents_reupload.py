"""#239 — 같은 파일명 재업로드 뒤 **실제 질의 경로**가 구버전 답을 재생하지 않는가.

기존 재업로드 테스트(test_integration_documents.py::test_같은_파일명_재업로드는_엎어치기)는
`cache.save_answer`로 행을 직접 심고 "행이 사라졌다"와 PG·색인 상태를 본다. 여기서는 그 상태 단언을
반복하지 않고, `/kms/query`로 캐시를 채우고(semantic 히트까지 확인) 재업로드 뒤 같은 질문이 어느
시점에 miss가 되고 무엇을 근거로 다시 저장되는지를 사용자 경로 그대로 본다.

시점별 기대(코드 근거: rag/ingestion.py index_pending_document ③, rag/cache.py get_semantic 6번):
- ③ 커밋 직후(DROP 전, ≤1분 공존 창): 구판 근거 캐시는 ③의 invalidate_source로 이미 없다 → miss.
  이때 생성된 답은 구·신 청크를 함께 근거({v1,v2})로 저장된다 — 공존 창의 제품 결정 범위이며
  #239 결정 1 = 손대지 않음. 이 행이 실제로 {v1,v2}인지 단언해야 아래 miss가 "행이 없어서"가 아니라
  "집합이 달라서"임이 보장된다.
- DROP 뒤: 현재 근거 집합이 {v2}뿐이라 {v1,v2} 행은 집합 불일치로 영구 miss → 신판으로 새로
  생성되고 {v2} 근거로 저장돼 다음 같은 질문부터 다시 히트한다.

인용 단언의 뜻: FakeLlm은 항상 1번 후보를 인용하므로 `citations == [(v2, 2)]`는 "후보 집합에 v1이
없다"의 귀결이다. 신판 인용의 실제 증거는 구판 청크 0건(DROP)과 캐시 miss 쪽이다.
"""
import pytest
from sqlalchemy import select

from database import AsyncSessionLocal
from rag.models import AnswerCache as AnswerCacheRow
from tests.conftest import ingest, sse_done, sse_meta
from tests.helpers_documents import MD, chunk_texts, upload

Q = '반품 기간은?'


async def _cache_rows(tenant_id: str) -> list[AnswerCacheRow]:
    async with AsyncSessionLocal() as session:
        return (await session.execute(
            select(AnswerCacheRow).where(AnswerCacheRow.tenant_id == tenant_id))).scalars().all()


def _cited(res) -> list[tuple[int, int]]:
    """done.citations → [(document_id, version)]."""
    return [(c['document_id'], c['version']) for c in sse_done(res)['citations']]


@pytest.mark.asyncio
async def test_재업로드_뒤_같은_질문은_캐시를_재생하지_않고_신판을_인용한다(
        client, tenant_id, fake_queue, blob_tmp, fake_embed, fake_llm, pass_gate, relaxed_quota):
    v1 = await upload(client, '환불정책.md', MD)
    assert await ingest(v1['document_id']) == {'done': 1, 'failed': 0}

    # 1) 질의 2회 — 둘째가 semantic 히트. 캐시 행(질의당 1행, cache_key upsert)의 근거는 v1.
    first = await client.post('/kms/query', json={'query': Q})
    assert sse_meta(first)['cached'] is False
    assert _cited(first) == [(v1['document_id'], 1)]
    second = await client.post('/kms/query', json={'query': Q})
    assert sse_meta(second)['cached'] is True and sse_meta(second)['cache_kind'] == 'semantic'
    assert _cited(second) == [(v1['document_id'], 1)]
    assert [r.source_doc_ids for r in await _cache_rows(tenant_id)] == [[v1['document_id']]]

    # 2) 같은 파일명으로 신판(14일→30일) 업로드·색인(③ 커밋) — 구판 근거 캐시는 이 커밋이 지운다.
    v2 = await upload(client, '환불정책.md', MD.replace(b'14', b'30'))
    assert await ingest(v2['document_id']) == {'done': 1, 'failed': 0}
    assert await _cache_rows(tenant_id) == []

    # 3) 공존 창(DROP 전) — 구판 청크가 아직 색인에 있어 miss로 새로 생성되고, 그 답은 {v1,v2}를 근거로 저장된다.
    assert await chunk_texts(v1['document_id']) != []
    window = await client.post('/kms/query', json={'query': Q})
    assert sse_meta(window)['cached'] is False
    assert set(_cited(window)) <= {(v1['document_id'], 1), (v2['document_id'], 2)}   # 후보 순서는 비결정
    (row,) = await _cache_rows(tenant_id)
    assert set(row.source_doc_ids) == {v1['document_id'], v2['document_id']}

    # 4) DROP(다음 회차) — 구판 청크 0건.
    assert await ingest(v1['document_id']) == {'done': 1, 'failed': 0}
    assert await chunk_texts(v1['document_id']) == []

    # 5) 같은 질문 — 창에 저장된 {v1,v2} 행은 현재 근거 {v2}와 달라 miss, 신판만 인용.
    after = await client.post('/kms/query', json={'query': Q})
    assert sse_meta(after)['cached'] is False
    assert _cited(after) == [(v2['document_id'], 2)]
    # 6) 그 답이 {v2} 근거로 덮어써져(같은 cache_key upsert) 다음부터 다시 히트하고, 재생 인용도 신판.
    again = await client.post('/kms/query', json={'query': Q})
    assert sse_meta(again)['cached'] is True
    assert _cited(again) == [(v2['document_id'], 2)]
    assert [r.source_doc_ids for r in await _cache_rows(tenant_id)] == [[v2['document_id']]]
