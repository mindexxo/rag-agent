"""대화 목록 '최근 사용순' 정렬 통합 테스트 — last_used_at 갱신 (ChatGPT식 UX).

옛 대화에 턴이 추가되면 목록 첫 번째로 부상해야 한다.
실 DB 사용 (tenant_isolation과 같은 패턴: 랜덤 tenant + finally 정리).
"""
import uuid

import pytest
import pytest_asyncio
from sqlalchemy import delete, select

from database import AsyncSessionLocal
from rag.turn_state import add_pending_turn
from rag.models import Conversation, Message


@pytest_asyncio.fixture(autouse=True)
async def _reset_db_pool():
    """pytest-asyncio는 테스트마다 새 이벤트 루프를 만든다 — 풀에 남은 커넥션이 이전 루프에
    묶여 'attached to a different loop'가 나므로, 테스트 끝날 때 풀을 비운다.
    (통합 테스트 확장 시 conftest 공용 fixture로 승격 예정)"""
    yield
    from database import engine
    await engine.dispose()


@pytest_asyncio.fixture
async def tenant_id():
    t = str(uuid.uuid4())
    yield t
    async with AsyncSessionLocal() as session:
        await session.execute(delete(Message).where(Message.tenant_id == t))
        await session.execute(delete(Conversation).where(Conversation.tenant_id == t))
        await session.commit()


async def _create_two_conversations(tenant_id: str) -> tuple[int, int]:
    """c1(먼저) → c2(나중) 순으로 생성. 반환: (c1.id, c2.id)"""
    async with AsyncSessionLocal() as session:
        c1, c2 = Conversation(tenant_id=tenant_id), Conversation(tenant_id=tenant_id)
        session.add_all([c1, c2])
        await session.flush()
        ids = (c1.id, c2.id)
        await session.commit()
    return ids


async def _ordered_ids(tenant_id: str) -> list[int]:
    async with AsyncSessionLocal() as session:
        return list((await session.execute(
            select(Conversation.id)
            .where(Conversation.tenant_id == tenant_id)
            .order_by(Conversation.last_used_at.desc())
        )).scalars().all())


@pytest.mark.asyncio
async def test_턴_시작_저장시_목록_첫번째로_부상(tenant_id):
    # #72로 모든 경로(차단·캐시히트·근거없음·생성)가 add_pending_turn으로 턴을 연다 —
    # 예전엔 즉시 경로만 save_exchange로 부상시켰고 그 경로를 따로 검증했다.
    c1_id, c2_id = await _create_two_conversations(tenant_id)
    # PG now()는 트랜잭션 시각 고정 — 별도 세션(트랜잭션)이라 시간차가 생긴다
    async with AsyncSessionLocal() as session:
        await add_pending_turn(session, tenant_id, c1_id, 'q')
        await session.commit()
    assert (await _ordered_ids(tenant_id))[0] == c1_id


@pytest.mark.asyncio
async def test_타_테넌트_대화는_건드리지_않음(tenant_id):
    # _touch_conversation의 tenant WHERE 검증 — 같은 id라도 남의 테넌트면 갱신 금지
    c1_id, _ = await _create_two_conversations(tenant_id)
    other = str(uuid.uuid4())
    try:
        async with AsyncSessionLocal() as session:
            before = (await session.execute(
                select(Conversation.last_used_at).where(Conversation.id == c1_id)
            )).scalar_one()
            # 다른 tenant_id로 같은 conversation_id에 저장 시도 — 메시지는 남의 대화에 붙지만
            # (ensure_conversation을 우회한 직접 호출), last_used_at은 갱신되면 안 된다
            await add_pending_turn(session, other, c1_id, 'q')
            await session.commit()
        async with AsyncSessionLocal() as session:
            after = (await session.execute(
                select(Conversation.last_used_at).where(Conversation.id == c1_id)
            )).scalar_one()
        assert after == before
    finally:
        async with AsyncSessionLocal() as session:
            await session.execute(delete(Message).where(Message.tenant_id == other))
            await session.commit()
