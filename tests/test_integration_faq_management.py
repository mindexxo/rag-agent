"""FAQ 관리 통합 테스트 (#278, FE 요청 4~6) — 카테고리·응답 필드(등록/변경자·시각·ref_count)·목록 검색/필터/페이징.

항목별 FAQ 인용(저장·대화 이력·ref_count)이 여기 들어간다 — 번호·표시명 단위는 tests/test_prompts.py,
SSE done.citations는 tests/test_integration_query_sse.py, 통계 집계는 tests/test_integration_stats.py.
"""
import pytest
from httpx import ASGITransport, AsyncClient

from database import AsyncSessionLocal
from rag import outbox
from rag.citation_labels import citation_tail
from schemas.kms import FAQ_CATEGORY_NAME_MAX
from tests.conftest import sse_done, sse_meta, sync_faq


def _client_for(tenant: str) -> AsyncClient:
    from main import app
    return AsyncClient(transport=ASGITransport(app=app), base_url='http://testserver',
                       headers={'X-Tenant-Id': tenant})


async def _faq(client, question: str, answer: str = '답변', variants: list[str] | None = None,
               headers: dict | None = None, **extra) -> dict:
    res = await client.post('/kms/faqs', json={'question': question, 'variants': variants or [], 'answer': answer, **extra},
                            headers=headers or {})
    assert res.status_code == 200, res.text
    await sync_faq(res.json()['id'])
    return res.json()


async def _category(client, name: str) -> dict:
    res = await client.post('/kms/faq-categories', json={'name': name})
    assert res.status_code == 200, res.text
    return res.json()


async def _items(client, query: str = '') -> list[dict]:
    res = await client.get(f'/kms/faqs{query}')
    assert res.status_code == 200, res.text
    return res.json()['items']


# ───────────────────────── 6. 목록 — 모양·정렬·페이징·검색 ─────────────────────────

@pytest.mark.asyncio
async def test_목록_모양_최신순_페이징(client, tenant_id):
    ids = [(await _faq(client, f'질문 {i}'))['id'] for i in range(3)]

    body = (await client.get('/kms/faqs')).json()
    assert set(body) == {'items', 'total', 'has_more'}                 # #278: 배열 → 문서 목록과 같은 모양
    assert [f['id'] for f in body['items']] == ids[::-1]               # 최신 등록이 위
    assert body['total'] == 3 and body['has_more'] is False

    page1 = (await client.get('/kms/faqs?limit=2')).json()
    assert [f['id'] for f in page1['items']] == ids[::-1][:2] and page1['has_more'] is True
    page2 = (await client.get('/kms/faqs?limit=2&offset=2')).json()
    assert [f['id'] for f in page2['items']] == [ids[0]] and page2['has_more'] is False   # 마지막 부분 페이지
    assert page2['total'] == 3
    # 파라미터 위생 — 범위 밖 limit·음수 offset은 에러가 아니라 보정(문서 목록과 같다)
    assert len((await client.get('/kms/faqs?limit=0&offset=-5')).json()['items']) == 3


@pytest.mark.asyncio
async def test_q_검색은_질문_유사질문_답변_중_하나라도_대소문자_무시(client, tenant_id):
    a = await _faq(client, 'Refund policy?', answer='7 days', variants=['money back'])
    b = await _faq(client, '배송은 얼마나 걸리나요?', answer='이틀이면 도착')
    c = await _faq(client, '100% 환불이 되나요?', answer='조건부')

    assert [f['id'] for f in await _items(client, '?q=refund')] == [a['id']]       # 질문·대소문자 무시
    assert [f['id'] for f in await _items(client, '?q=MONEY')] == [a['id']]        # 유사 질문
    assert [f['id'] for f in await _items(client, '?q=이틀')] == [b['id']]         # 답변
    assert [f['id'] for f in await _items(client, '?q=100%25')] == [c['id']]       # %는 와일드카드가 아니라 글자
    assert await _items(client, '?q=없는말') == []
    assert len(await _items(client, '?q=%20%20')) == 3                              # 공백뿐인 q = 미전송
    # 유사 질문은 원소별로 본다(리뷰 지적 2) — JSON 배열의 따옴표·쉼표·대괄호는 검색 대상이 아니다
    assert await _items(client, '?q=%22') == [] and await _items(client, '?q=,') == [] and await _items(client, '?q=[') == []
    d = await _faq(client, '쉼표, 들어간 질문?', variants=['따옴표"가 든 유사 질문'])
    assert [f['id'] for f in await _items(client, '?q=,')] == [d['id']]            # 진짜 쉼표가 든 질문만
    assert [f['id'] for f in await _items(client, '?q=%22가')] == [d['id']]        # 원소 안의 따옴표도 글자로 검색된다
    body = (await client.get('/kms/faqs?q=refund')).json()
    assert body['total'] == 1                                                       # total도 같은 조건


# ───────────────────────── 4. 카테고리 ─────────────────────────

@pytest.mark.asyncio
async def test_카테고리_규칙은_폴더와_같다(client, tenant_id):
    cat = await _category(client, '환불')
    assert cat == {'id': cat['id'], 'name': '환불', 'faq_count': 0}

    assert (await client.post('/kms/faq-categories', json={'name': '   '})).status_code == 422          # 빈 이름
    assert (await client.post('/kms/faq-categories', json={'name': 'x' * (FAQ_CATEGORY_NAME_MAX + 1)})).status_code == 422
    assert (await client.post('/kms/faq-categories', json={'name': ' 환불 '})).status_code == 409        # 같은 이름(공백 제거 후)

    other = await _category(client, '배송')
    await _faq(client, '환불 기간은?', category_id=cat['id'])
    await _faq(client, '환불 수수료는?', category_id=cat['id'])
    listing = (await client.get('/kms/faq-categories')).json()
    assert [(c['name'], c['faq_count']) for c in listing] == [('배송', 0), ('환불', 2)]               # 이름순 + 소속 수

    res = await client.patch(f"/kms/faq-categories/{cat['id']}", json={'name': '환불·취소'})
    assert res.status_code == 200 and res.json() == {'id': cat['id'], 'name': '환불·취소', 'faq_count': 2}
    assert (await client.patch(f"/kms/faq-categories/{cat['id']}", json={'name': '배송'})).status_code == 409   # 다른 것과 충돌
    assert (await client.patch(f"/kms/faq-categories/{cat['id']}", json={'name': ''})).status_code == 422

    res = await client.delete(f"/kms/faq-categories/{cat['id']}")
    assert res.status_code == 409 and 'FAQ 2개' in res.json()['detail']                                  # 소속 FAQ가 있으면 삭제 불가
    for f in await _items(client, f"?category_id={cat['id']}"):
        assert (await client.patch(f"/kms/faqs/{f['id']}", json={'category_id': None})).status_code == 200
    assert (await client.delete(f"/kms/faq-categories/{cat['id']}")).status_code == 204
    assert (await client.delete(f"/kms/faq-categories/{cat['id']}")).status_code == 404
    assert (await client.delete(f"/kms/faq-categories/{other['id']}")).status_code == 204


@pytest.mark.asyncio
async def test_카테고리_격리(client, tenant_id, other_tenant_id):
    cat = await _category(client, 'A사 전용')
    async with _client_for(other_tenant_id) as other:
        assert all(c['id'] != cat['id'] for c in (await other.get('/kms/faq-categories')).json())
        assert (await other.patch(f"/kms/faq-categories/{cat['id']}", json={'name': '탈취'})).status_code == 404
        assert (await other.delete(f"/kms/faq-categories/{cat['id']}")).status_code == 404
        # 남의 카테고리에 FAQ를 넣을 수 없다 — 문서 업로드가 남의 폴더를 404로 막는 것과 같다
        res = await other.post('/kms/faqs', json={'question': 'B사', 'variants': [], 'answer': 'b', 'category_id': cat['id']})
        assert res.status_code == 404
        b = (await other.post('/kms/faqs', json={'question': 'B사 2', 'variants': [], 'answer': 'b'})).json()
        assert (await other.patch(f"/kms/faqs/{b['id']}", json={'category_id': cat['id']})).status_code == 404
    assert (await client.get('/kms/faq-categories')).json()[0]['faq_count'] == 0


@pytest.mark.asyncio
async def test_FAQ_카테고리_지정과_필터_null은_미분류_미전송은_유지(client, tenant_id):
    cat = await _category(client, '환불')
    f1 = await _faq(client, '환불 기간은?', category_id=cat['id'])
    f2 = await _faq(client, '배송은?')
    assert f1['category_id'] == cat['id'] and f2['category_id'] is None

    assert [f['id'] for f in await _items(client, f"?category_id={cat['id']}")] == [f1['id']]
    assert [f['id'] for f in await _items(client, '?category_id=0')] == [f2['id']]          # 0 = 미분류만(문서 folder_id=0 관례)
    assert len(await _items(client)) == 2                                                   # 미전송 = 전체

    assert (await client.patch(f"/kms/faqs/{f2['id']}", json={'category_id': cat['id']})).json()['category_id'] == cat['id']
    assert (await client.patch(f"/kms/faqs/{f1['id']}", json={'answer': '7일'})).json()['category_id'] == cat['id']   # 미전송 = 유지
    assert (await client.patch(f"/kms/faqs/{f1['id']}", json={'category_id': None})).json()['category_id'] is None   # null = 미분류
    assert (await client.patch(f"/kms/faqs/{f1['id']}", json={'category_id': 999999})).status_code == 404

    # is_active 필터
    await client.patch(f"/kms/faqs/{f2['id']}", json={'is_active': False})
    assert [f['id'] for f in await _items(client, '?is_active=false')] == [f2['id']]
    assert [f['id'] for f in await _items(client, '?is_active=true')] == [f1['id']]
    assert [f['id'] for f in await _items(client, f"?is_active=false&category_id={cat['id']}")] == [f2['id']]


@pytest.mark.asyncio
async def test_카테고리만_바꾸면_재색인_없음(client, tenant_id):
    """관리용 분류라 검색·생성 입력이 아니다 — 대기열에 아무 행도 남기지 않는다(재임베딩은 GPU를 태운다)."""
    cat = await _category(client, '환불')
    f = await _faq(client, '환불 기간은?')
    assert (await client.patch(f"/kms/faqs/{f['id']}", json={'category_id': cat['id']})).status_code == 200
    async with AsyncSessionLocal() as s:
        assert await outbox.pending_row_ids(s, faq_id=f['id']) == []


# ───────────────────────── 5. 응답 필드 ─────────────────────────

@pytest.mark.asyncio
async def test_등록자_변경자_시각(client, tenant_id):
    f = await _faq(client, '환불 기간은?', headers={'X-User-Id': 'agent-kim'})
    assert f['created_by'] == 'agent-kim' and f['updated_by'] == 'agent-kim'
    assert f['created_at'].endswith('+09:00') and f['updated_at'].endswith('+09:00')   # 문서와 같은 KST 오프셋 ISO
    assert f['ref_count'] == 0

    res = await client.patch(f"/kms/faqs/{f['id']}", json={'is_active': False}, headers={'X-User-Id': 'agent-lee'})
    body = res.json()
    assert body['created_by'] == 'agent-kim' and body['updated_by'] == 'agent-lee'     # 켜고 끄기도 '손댄 것'
    assert body['updated_at'] >= f['updated_at']

    res = await client.patch(f"/kms/faqs/{f['id']}", json={'answer': '7일'})            # 헤더 없음 → null로 덮는다(결정 5)
    assert res.json()['updated_by'] is None and res.json()['created_by'] == 'agent-kim'

    g = await _faq(client, '배송은?')                                                    # 헤더 없이 등록
    assert g['created_by'] is None and g['updated_by'] is None
    assert {k for k in (await _items(client))[0]} >= {'category_id', 'created_at', 'updated_at', 'created_by', 'updated_by', 'ref_count'}


@pytest.mark.asyncio
async def test_항목별_인용_저장_ref_count_대화이력(client, tenant_id, fake_llm, pass_gate, relaxed_quota):
    """인용된 FAQ 항목이 cited_faq_ids로 저장되고, 목록 ref_count·PATCH 응답·대화 이력 인용에 같은 사실이 보인다."""
    f = await _faq(client, '환불 기간은?', answer='7일')
    fake_llm.answer = f'7일 이내 처리됩니다. {citation_tail([1])}'
    headers = {'X-User-Id': 'agent-kim'}

    res = await client.post('/kms/query', json={'query': '환불 기간 알려줘'}, headers=headers)
    assert res.status_code == 200, res.text
    assert sse_done(res)['citations'] == [{'document_id': None, 'filename': 'FAQ: 환불 기간은?', 'version': 1, 'faq_id': f['id']}]
    conv_id = sse_meta(res)['conversation_id']

    assert (await _items(client))[0]['ref_count'] == 1

    msgs = (await client.get(f'/kms/conversations/{conv_id}/messages', headers=headers)).json()
    assistant = [m for m in msgs if m['role'] == 'assistant'][0]
    assert assistant['citations'] == [{'document_id': None, 'filename': 'FAQ: 환불 기간은?', 'version': 1, 'faq_id': f['id']}]

    # 캐시 재생(즉시 경로)도 같은 항목을 센다 — 두 번째 같은 질문
    res = await client.post('/kms/query', json={'query': '환불 기간 알려줘'}, headers=headers)
    assert sse_meta(res)['cached'] is True
    assert (await _items(client))[0]['ref_count'] == 2
    # PATCH 응답에도 실린다(FE 요청 5). 내용 수정은 캐시를 지우므로 캐시 재생 확인 뒤에 한다
    assert (await client.patch(f"/kms/faqs/{f['id']}", json={'answer': '7일!'})).json()['ref_count'] == 2


@pytest.mark.asyncio
async def test_여러_FAQ_중_인용된_항목만_센다(client, tenant_id, fake_llm, pass_gate, relaxed_quota):
    """항목마다 번호가 다르다 — 2번만 인용하면 2번 항목만 ref_count가 오른다(#278 전엔 'FAQ' 하나로 접혀 구분 불가)."""
    a = await _faq(client, '환불 기간은?', answer='7일')
    b = await _faq(client, '환불 수수료는?', answer='없음')
    fake_llm.answer = f'수수료는 없습니다. {citation_tail([2])}'

    res = await client.post('/kms/query', json={'query': '환불 수수료 있나요'})
    cited = sse_done(res)['citations']
    assert len(cited) == 1 and cited[0]['faq_id'] in {a['id'], b['id']}              # 가짜 임베딩이라 순위는 비결정 — 인용된 쪽만 센다
    counts = {f['id']: f['ref_count'] for f in await _items(client)}
    assert counts == {cited[0]['faq_id']: 1, ({a['id'], b['id']} - {cited[0]['faq_id']}).pop(): 0}
