"""백그라운드 태스크의 늦은 정리를 고정 sleep 없이 기다리는 테스트 도우미."""

import asyncio

from database import AsyncSessionLocal
from rag import limiter
from rag.models import Message


async def inflight_count(tenant_id: str) -> int:
    """리미터 ZSET에 남아 있는 in-flight 항목 수."""
    return await limiter._redis().zcard(f'kms:inflight:t:{tenant_id}')


async def wait_inflight(tenant_id: str, expected: int = 0, tries: int = 100) -> int:
    """리미터 슬롯 수가 expected가 될 때까지 기다린다 — 반납은 상태 커밋 뒤 finally에서 온다."""
    for _ in range(tries):
        if (n := await inflight_count(tenant_id)) == expected:
            return n
        await asyncio.sleep(0.05)
    return await inflight_count(tenant_id)


async def wait_message_status(message_id: int, expected: str, tries: int = 100) -> str:
    """Message.status가 expected가 될 때까지 기다린다 — 백그라운드 태스크가 별도 세션으로 커밋한다. 없으면 'MISSING'."""
    for _ in range(tries):
        async with AsyncSessionLocal() as session:
            msg = await session.get(Message, message_id)
            if msg is not None and msg.status == expected:
                return msg.status
        await asyncio.sleep(0.05)
    async with AsyncSessionLocal() as session:
        msg = await session.get(Message, message_id)
        return msg.status if msg else 'MISSING'
