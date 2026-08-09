"""답변 피드백 API(#8) 통합 테스트 — 👍/👎+태그 저장·취소·검증·소유/테넌트 격리.

실제 앱 + DB (conftest client 패턴). LLM 불필요 — 검증은 전부 DB 조회 전 단계.
"""
import pytest
from sqlalchemy import select

from database import AsyncSessionLocal
from rag.models import Conversation, Message

USER_A = {'X-User-Id': 'agent-a'}
USER_B = {'X-User-Id': 'agent-b'}


async def _seed_turn(tenant_id: str, created_by: str = 'agent-a') -> tuple[int, int, int]:
    """대화 1개 + user/assistant 한 턴 삽입 → (conversation_id, assistant_id, user_id)."""
    async with AsyncSessionLocal() as s:
        conv = Conversation(tenant_id=tenant_id, created_by=created_by)
        s.add(conv)
        await s.flush()
        u = Message(tenant_id=tenant_id, conversation_id=conv.id, role='user', content='반품 기간?')
        s.add(u)
        await s.flush()
        a = Message(tenant_id=tenant_id, conversation_id=conv.id, role='assistant',
                    content='14일입니다', question_message_id=u.id)
        s.add(a)
        await s.flush()
        ids = (conv.id, a.id, u.id)
        await s.commit()
    return ids


async def _db_state(assistant_id: int) -> tuple[bool | None, str | None]:
    async with AsyncSessionLocal() as s:
        m = (await s.execute(select(Message).where(Message.id == assistant_id))).scalar_one()
        return m.feedback, m.feedback_tag


@pytest.mark.asyncio
async def test_싫어요와_태그_저장(client, tenant_id):
    _, aid, _ = await _seed_turn(tenant_id)
    res = await client.patch(f'/kms/messages/{aid}/feedback',
                             json={'feedback': False, 'tag': 'wrong_info'}, headers=USER_A)
    assert res.status_code == 200
    assert res.json() == {'message_id': aid, 'feedback': False, 'feedback_tag': 'wrong_info'}
    assert await _db_state(aid) == (False, 'wrong_info')


@pytest.mark.asyncio
async def test_싫어요_태그없이도_기록(client, tenant_id):
    """태그는 옵셔널 — 스킵해도 👎 자체는 저장 (필수화하면 👎가 줄어드는 UX 함정)."""
    _, aid, _ = await _seed_turn(tenant_id)
    res = await client.patch(f'/kms/messages/{aid}/feedback',
                             json={'feedback': False}, headers=USER_A)
    assert res.status_code == 200
    assert await _db_state(aid) == (False, None)


@pytest.mark.asyncio
async def test_좋아요로_전환하면_태그_강제_NULL(client, tenant_id):
    _, aid, _ = await _seed_turn(tenant_id)
    await client.patch(f'/kms/messages/{aid}/feedback',
                       json={'feedback': False, 'tag': 'outdated_doc'}, headers=USER_A)
    res = await client.patch(f'/kms/messages/{aid}/feedback',
                             json={'feedback': True}, headers=USER_A)
    assert res.status_code == 200
    assert await _db_state(aid) == (True, None)


@pytest.mark.asyncio
async def test_취소는_둘다_NULL(client, tenant_id):
    _, aid, _ = await _seed_turn(tenant_id)
    await client.patch(f'/kms/messages/{aid}/feedback',
                       json={'feedback': False, 'tag': 'insufficient'}, headers=USER_A)
    res = await client.patch(f'/kms/messages/{aid}/feedback',
                             json={'feedback': None}, headers=USER_A)
    assert res.status_code == 200
    assert await _db_state(aid) == (None, None)


@pytest.mark.asyncio
async def test_좋아요에_태그_동봉은_422(client, tenant_id):
    _, aid, _ = await _seed_turn(tenant_id)
    res = await client.patch(f'/kms/messages/{aid}/feedback',
                             json={'feedback': True, 'tag': 'wrong_info'}, headers=USER_A)
    assert res.status_code == 422


@pytest.mark.asyncio
async def test_미정의_태그는_422(client, tenant_id):
    _, aid, _ = await _seed_turn(tenant_id)
    res = await client.patch(f'/kms/messages/{aid}/feedback',
                             json={'feedback': False, 'tag': 'too_slow'}, headers=USER_A)
    assert res.status_code == 422


@pytest.mark.asyncio
async def test_user_메시지에는_404(client, tenant_id):
    _, _, uid = await _seed_turn(tenant_id)
    res = await client.patch(f'/kms/messages/{uid}/feedback',
                             json={'feedback': True}, headers=USER_A)
    assert res.status_code == 404


@pytest.mark.asyncio
async def test_남의_대화_메시지는_404(client, tenant_id):
    _, aid, _ = await _seed_turn(tenant_id, created_by='agent-a')
    res = await client.patch(f'/kms/messages/{aid}/feedback',
                             json={'feedback': True}, headers=USER_B)
    assert res.status_code == 404
    assert await _db_state(aid) == (None, None)


@pytest.mark.asyncio
async def test_타_테넌트_메시지는_404(client, tenant_id, other_tenant_id):
    """client는 tenant_id 헤더 고정 — 타 테넌트 메시지 id를 알아도 접근 불가 (WHERE 격리 계약)."""
    _, aid, _ = await _seed_turn(other_tenant_id)
    res = await client.patch(f'/kms/messages/{aid}/feedback',
                             json={'feedback': True}, headers=USER_A)
    assert res.status_code == 404


@pytest.mark.asyncio
async def test_대화조회에_message_id와_피드백_상태_노출(client, tenant_id):
    """FE 연동 계약: 히스토리 재진입 시 버튼 상태 복원 재료가 응답에 있어야 한다."""
    cid, aid, _ = await _seed_turn(tenant_id)
    await client.patch(f'/kms/messages/{aid}/feedback',
                       json={'feedback': False, 'tag': 'wrong_source'}, headers=USER_A)
    res = await client.get(f'/kms/conversations/{cid}/messages', headers=USER_A)
    assert res.status_code == 200
    assistant = [m for m in res.json() if m['role'] == 'assistant'][0]
    assert assistant['message_id'] == aid
    assert assistant['feedback'] is False
    assert assistant['feedback_tag'] == 'wrong_source'


@pytest.mark.asyncio
async def test_done_아닌_메시지는_404(client, tenant_id):
    """실패/생성중/차단 턴엔 평가할 답변이 없다 — 집계 오염 방지."""
    async with AsyncSessionLocal() as s:
        conv = Conversation(tenant_id=tenant_id, created_by='agent-a')
        s.add(conv)
        await s.flush()
        a = Message(tenant_id=tenant_id, conversation_id=conv.id, role='assistant',
                    content='', status='failed')
        s.add(a)
        await s.flush()
        aid = a.id
        await s.commit()
    res = await client.patch(f'/kms/messages/{aid}/feedback',
                             json={'feedback': False}, headers=USER_A)
    assert res.status_code == 404
