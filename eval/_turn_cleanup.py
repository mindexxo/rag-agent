"""eval 전용 — 운영 서비스를 태워 만든 대화·메시지를 즉시 되돌린다 (#72).

왜 필요한가: `RagService.prepare()`는 턴 **시작** 시점에 대화(신규면)와 이번 턴
(user 메시지 + assistant 자리표시)을 INSERT하고 **커밋**한다. 사용자의 질문이 파이프라인
중간에 죽어도 유실되지 않게 하려는 설계다(#72).

그런데 eval 하네스는 라우팅·생성 **결과만** 보고 `finalize()`를 부르지 않는다. 치우지 않으면
실행할 때마다 운영 테넌트에 대화와 `generating` 메시지가 그대로 쌓인다 — 실제로 빈 대화가
4,505건 누적된 적이 있고(2026-08), 원인이 정확히 이것이었다.

`prepare(persist=False)` 같은 플래그를 운영 코드에 두지 않는 이유: 그건 "메시지를 저장하지
않는 경로"를 파이프라인에 새로 만드는 것이고, 우리가 #72에서 세운 보장(어떤 실패에도 질문은
남는다)을 정면으로 되돌리는 스위치가 된다. 어지른 쪽이 치우는 편이 안전하다.
"""
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession

from rag.models import Conversation, Message


async def discard_turn(session: AsyncSession, tenant_id: str, conversation_id: int) -> None:
    """이 eval 호출이 만든 대화와 그 메시지를 삭제하고 커밋한다.

    messages를 먼저 지운다 — conversations로의 FK가 걸려 있다. 두 DELETE 모두 tenant를
    명시한다(쓰기 경로는 예외 없이 테넌트를 WHERE에 넣는다는 프로젝트 원칙).
    """
    await session.execute(
        delete(Message)
        .where(Message.tenant_id == tenant_id)
        .where(Message.conversation_id == conversation_id)
    )
    await session.execute(
        delete(Conversation)
        .where(Conversation.tenant_id == tenant_id)
        .where(Conversation.id == conversation_id)
    )
    await session.commit()
