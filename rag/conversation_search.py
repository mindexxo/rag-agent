"""대화 목록 검색 — 제목·내용 부분일치 필터와 매칭 발췌(snippet).

호출부는 사용자 원문 q만 넘기고 LIKE 패턴 조립은 이 모듈 안에서 끝난다.
호출부 계약: 빈 문자열 q는 None으로 접어서 넘길 것 — '%%'는 전건 매칭이라 필터를 건 척만 한다.
"""

from sqlalchemy import or_, select, true
from sqlalchemy.ext.asyncio import AsyncSession

from rag.models import Conversation, Message
from text_norm import LIKE_ESCAPE_CHAR, like_pattern

# 매칭 지점 전후 문자 수 — 목록에서 한 줄로 보이므로 한글 기준 30자 남짓이 상한이다.
SNIPPET_RADIUS = 15


def _build_snippet(content: str, q: str) -> str | None:
    """매칭 지점 전후 SNIPPET_RADIUS자 발췌(잘린 쪽엔 말줄임).

    위치는 SQL position()이 아니라 파이썬에서 소문자로 찾는다 — position()은 대소문자를 구분해
    ILIKE로 매칭된 영문 검색어를 못 찾는다.
    """
    idx = content.lower().find(q.lower())
    if idx < 0:
        # ILIKE와 케이스폴딩이 달라 못 찾는 경우 — 대화는 결과에 남고 snippet만 빈다.
        return None
    start = max(0, idx - SNIPPET_RADIUS)
    end = min(len(content), idx + len(q) + SNIPPET_RADIUS)
    prefix = '…' if start > 0 else ''
    suffix = '…' if end < len(content) else ''
    return f'{prefix}{content[start:end]}{suffix}'


def search_filter(tenant_id: str, q: str | None):
    """제목 또는 대화 내용 부분일치 필터. q=None이면 필터 없음.

    메시지 매칭은 JOIN이 아니라 EXISTS다 — 한 대화에 여러 건이 걸려도 대화 행이 하나라 total이 부풀지 않는다.
    status로 거르지 않는다(차단 턴도 검색된다). 첨부 본문은 고객 개인 문서라 검색하지 않는다.
    """
    if q is None:
        return true()
    pattern = like_pattern(q)
    return or_(
        Conversation.title.ilike(pattern, escape=LIKE_ESCAPE_CHAR),
        select(Message.id)
        .where(Message.conversation_id == Conversation.id)
        .where(Message.tenant_id == tenant_id)  # 격리 — 상관 서브쿼리에도 WHERE 명시
        .where(Message.content.ilike(pattern, escape=LIKE_ESCAPE_CHAR))
        .exists(),
    )


async def snippets_for(
    session: AsyncSession, tenant_id: str, conversation_ids: list[int], q: str
) -> dict[int, str | None]:
    """페이지에 실린 대화마다 가장 최근 매칭 메시지의 발췌를 한 번의 쿼리로 모은다.

    한계: 대화당 LIMIT이 없어 매칭이 많은 대화는 행을 전부 실어온다 — 느려지면 row_number·LATERAL로 바꿀 것.
    """
    pattern = like_pattern(q)
    rows = (
        await session.execute(
            select(Message.conversation_id, Message.content)
            .where(Message.conversation_id.in_(conversation_ids))
            .where(Message.tenant_id == tenant_id)  # 격리 — 메시지에도 WHERE 명시
            .where(Message.content.ilike(pattern, escape=LIKE_ESCAPE_CHAR))
            .order_by(Message.conversation_id, Message.created_at.desc(), Message.id.desc())
        )
    ).all()

    latest: dict[int, str] = {}
    for conversation_id, content in rows:
        latest.setdefault(conversation_id, content)
    return {cid: _build_snippet(content, q) for cid, content in latest.items()}
