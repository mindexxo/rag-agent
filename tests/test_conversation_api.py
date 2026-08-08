"""대화 목록 API(#10) 통합 테스트 — 페이지네이션·소프트 삭제·제목 변경·user 스코핑.

실제 앱 + DB (conftest client 패턴). LLM 불필요 — 소유 검증은 prepare의
ensure_conversation 단계(LLM 호출 전)에서 끝나므로 fake_llm 없이 404 경로 검증 가능.
"""
from datetime import datetime, timedelta

import pytest
from sqlalchemy import select

from database import AsyncSessionLocal
from rag.models import Conversation, Message

USER_A = {'X-User-Id': 'agent-a'}
USER_B = {'X-User-Id': 'agent-b'}



async def _seed(tenant_id: str, n: int, created_by: str = 'agent-a') -> list[int]:
    """대화 n개 삽입 — last_used_at을 1분 간격으로 명시해 정렬이 결정적이게."""
    base = datetime.now()   # 모델 매핑이 naive(datetime) — aware를 넣으면 asyncpg가 거부
    async with AsyncSessionLocal() as s:
        convs = [Conversation(tenant_id=tenant_id, created_by=created_by, title=f'대화{i}',
                              last_used_at=base + timedelta(minutes=i)) for i in range(n)]
        s.add_all(convs)
        await s.flush()
        ids = [c.id for c in convs]
        await s.commit()
    return ids


@pytest.mark.asyncio
async def test_목록_응답형태_정렬_updated_at(client, tenant_id):
    ids = await _seed(tenant_id, 3)
    res = await client.get('/kms/conversations', headers=USER_A)
    assert res.status_code == 200
    body = res.json()
    assert set(body.keys()) == {'items', 'has_more'}          # breaking: 배열 → 객체
    assert [i['conversation_id'] for i in body['items']] == list(reversed(ids))   # 최근 사용순
    assert body['has_more'] is False
    assert all(i['updated_at'] for i in body['items'])


@pytest.mark.asyncio
async def test_페이지네이션_has_more와_offset(client, tenant_id):
    ids = await _seed(tenant_id, 5)
    page1 = (await client.get('/kms/conversations?limit=2', headers=USER_A)).json()
    assert len(page1['items']) == 2 and page1['has_more'] is True
    page2 = (await client.get('/kms/conversations?limit=2&offset=2', headers=USER_A)).json()
    assert len(page2['items']) == 2 and page2['has_more'] is True
    page3 = (await client.get('/kms/conversations?limit=2&offset=4', headers=USER_A)).json()
    assert len(page3['items']) == 1 and page3['has_more'] is False
    got = [i['conversation_id'] for i in page1['items'] + page2['items'] + page3['items']]
    assert got == list(reversed(ids))                          # 페이지 이어붙이면 전체 최근순


@pytest.mark.asyncio
async def test_user_스코핑_남의_대화_안보임(client, tenant_id):
    await _seed(tenant_id, 2, created_by='agent-a')
    res_b = await client.get('/kms/conversations', headers=USER_B)
    assert res_b.json()['items'] == []                         # B에겐 빈 목록
    # 헤더 미전송 → test-user 폴백 스코프
    await _seed(tenant_id, 1, created_by='test-user')
    res_none = await client.get('/kms/conversations')
    assert len(res_none.json()['items']) == 1


@pytest.mark.asyncio
async def test_메시지_조회도_소유_검증(client, tenant_id):
    (cid,) = await _seed(tenant_id, 1, created_by='agent-a')
    assert (await client.get(f'/kms/conversations/{cid}/messages', headers=USER_A)).status_code == 200
    assert (await client.get(f'/kms/conversations/{cid}/messages', headers=USER_B)).status_code == 404


@pytest.mark.asyncio
async def test_소프트삭제_흐름(client, tenant_id):
    (cid,) = await _seed(tenant_id, 1)
    async with AsyncSessionLocal() as s:                       # 이력 보존 검증용 메시지
        s.add(Message(conversation_id=cid, tenant_id=tenant_id, role='user', content='q'))
        await s.commit()

    assert (await client.delete(f'/kms/conversations/{cid}', headers=USER_A)).status_code == 204
    # 목록·조회·재질의 전부 미노출/404
    assert (await client.get('/kms/conversations', headers=USER_A)).json()['items'] == []
    assert (await client.get(f'/kms/conversations/{cid}/messages', headers=USER_A)).status_code == 404
    res_q = await client.post('/kms/query?stream=false', headers=USER_A,
                              json={'query': '계속 질문', 'conversation_id': cid})
    assert res_q.status_code == 404                            # 삭제된 대화로 질의 계속 차단
    # 재삭제도 404 (이미 안 보이는 대상)
    assert (await client.delete(f'/kms/conversations/{cid}', headers=USER_A)).status_code == 404
    # 소프트 삭제 — row·메시지는 감사 목적 보존
    async with AsyncSessionLocal() as s:
        conv = await s.get(Conversation, cid)
        assert conv is not None and conv.deleted_at is not None
        assert (await s.execute(select(Message).where(Message.conversation_id == cid))).scalars().all()


@pytest.mark.asyncio
async def test_제목변경(client, tenant_id):
    (cid,) = await _seed(tenant_id, 1)
    res = await client.patch(f'/kms/conversations/{cid}', headers=USER_A, json={'title': '새 제목'})
    assert res.status_code == 200 and res.json()['title'] == '새 제목'
    assert (await client.patch(f'/kms/conversations/{cid}', headers=USER_B,
                               json={'title': 'x'})).status_code == 404   # 남의 대화
    assert (await client.patch(f'/kms/conversations/{cid}', headers=USER_A,
                               json={'title': ''})).status_code == 422    # 빈 제목


@pytest.mark.asyncio
async def test_query_신규대화에_created_by_저장(client, tenant_id, fake_llm, pass_gate):
    from tests.test_integration_query_sse import _register_faq
    await _register_faq(client)
    res = await client.post('/kms/query?stream=false', headers=USER_A, json={'query': '환불 기간 알려줘'})
    assert res.status_code == 200
    cid = res.json()['conversation_id']
    async with AsyncSessionLocal() as s:
        conv = await s.get(Conversation, cid)
        assert conv.created_by == 'agent-a'
    # 만든 사람에겐 이어가기 허용, 남에겐 404
    assert (await client.post('/kms/query?stream=false', headers=USER_B,
                              json={'query': '그럼 교환은?', 'conversation_id': cid})).status_code == 404
