"""외부 색인 백엔드의 읽기 권위 필터 계약 (#139 정합 1층) — DB만 필요, OpenSearch 불필요.

## 여기서 지키는 것

외부 검색 엔진을 쓰면 청크와 색인이 더 이상 한 트랜잭션이 아니다. 그때 불일치의 두 방향은
위험도가 다르다:

  색인에 있고 PG엔 없음/비검색  → 삭제·비공개 문서가 답변에 인용되는 **오답**
  PG에 있고 색인엔 없음         → 못 찾는 **누락**(no_evidence). 오답은 아니다

오답 방향은 **구조적으로 불가능해야** 한다. 그 방어선이 `_fetch_chunk_map(searchable_only=True)`
다 — 색인이 준 id를 PG로 되물을 때 `_searchable_condition()`을 걸어, PG를 통과하지 못하는
것은 결과에 못 들어오게 한다. 덕분에 문서 삭제·검색토글·폴더 off·FAQ off가 **색인을 건드리지
않아도 즉시 반영된다**(설계 전문은 rag/opensearch.py 상단).

그래서 이 필터가 조용히 빠지면 "삭제한 문서가 계속 인용된다"가 된다 — 테스트로 묶는다.
pg 경로는 searchable_only=False라 이전과 동일해야 하고, 그것도 함께 고정한다.
"""
import pytest
from database import AsyncSessionLocal
from rag.models import Chunk, Document, Faq, Folder
from rag.retriever import _fetch_chunk_map
from tests.conftest import fake_vector


async def _seed(tenant_id: str) -> dict:
    """문서 2개(검색가능/비검색) + 폴더 off 문서 1 + FAQ 2개(활성/비활성)."""
    async with AsyncSessionLocal() as s:
        folder_off = Folder(tenant_id=tenant_id, name='보관함', is_searchable=False)
        s.add(folder_off)
        await s.flush()

        ids = {}
        specs = [
            ('ok', dict(filename='활성.pdf', is_active=True, status='ready',
                        is_searchable=True, folder_id=None)),
            ('unsearchable', dict(filename='비검색.pdf', is_active=True, status='ready',
                                  is_searchable=False, folder_id=None)),
            ('deleted', dict(filename='삭제됨.pdf', is_active=False, status='deleted',
                             is_searchable=True, folder_id=None)),
            ('folder_off', dict(filename='보관.pdf', is_active=True, status='ready',
                                is_searchable=True, folder_id=folder_off.id)),
        ]
        for key, kw in specs:
            doc = Document(tenant_id=tenant_id, mime='application/pdf',
                           blob_path=f'blob://{key}.pdf', version=1, **kw)
            s.add(doc)
            await s.flush()
            ch = Chunk(tenant_id=tenant_id, document_id=doc.id, text=f'{key} 본문',
                       chunk_index=0, dense=fake_vector(key), heading_path=[])
            s.add(ch)
            await s.flush()
            ids[key] = ch.id

        for key, active in (('faq_on', True), ('faq_off', False)):
            faq = Faq(tenant_id=tenant_id, question=f'{key}?', answer='답', is_active=active)
            s.add(faq)
            await s.flush()
            ch = Chunk(tenant_id=tenant_id, faq_id=faq.id, text=f'Q: {key}\nA: 답',
                       chunk_index=0, dense=fake_vector(key), heading_path=[])
            s.add(ch)
            await s.flush()
            ids[key] = ch.id

        await s.commit()
        return ids


@pytest.mark.asyncio
async def test_권위필터가_비검색_청크를_결과에서_뺀다(tenant_id):
    """색인이 낡아 이 id들을 돌려줘도 답변에 못 들어와야 한다 — 오답 차단."""
    ids = await _seed(tenant_id)
    async with AsyncSessionLocal() as s:
        got = await _fetch_chunk_map(s, list(ids.values()), searchable_only=True)

    assert ids['ok'] in got, '검색 가능한 문서 청크는 통과해야 한다'
    assert ids['faq_on'] in got, '활성 FAQ 청크는 통과해야 한다'
    for key in ('unsearchable', 'deleted', 'folder_off', 'faq_off'):
        assert ids[key] not in got, f'{key} 청크가 새어 나왔다 — 삭제·비공개 문서가 인용된다'


@pytest.mark.asyncio
async def test_pg_경로는_필터를_걸지_않는다(tenant_id):
    """기본값(False)에서는 넘긴 id를 그대로 돌려준다 — pg 경로 동작 보존.

    pg 경로의 id는 이미 같은 조건을 통과해 나오므로 여기서 다시 걸 이유가 없고,
    걸면 조인만 늘어난다(_fetch_chunk_map docstring).
    """
    ids = await _seed(tenant_id)
    async with AsyncSessionLocal() as s:
        got = await _fetch_chunk_map(s, list(ids.values()))
    assert set(got) == set(ids.values()), 'pg 경로가 조용히 후보를 깎으면 안 된다'
