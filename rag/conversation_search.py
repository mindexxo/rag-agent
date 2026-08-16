"""대화 히스토리 검색 (#28) — 제목·내용 부분일치 필터 + 매칭 발췌(snippet).

routers/conversations.py에 살던 검색 로직을 분리했다(#46) — 라우터는 HTTP 경계만
담당한다는 9번 축 원칙. rag/conversation.py에 합치지 않은 이유: 그쪽은 프롬프트
재료(답변 품질과 함께 움직임)고 여기는 검색 UI(#28, FE 요구와 함께 움직임)라
변경 이유가 다르다.

공개 표면은 search_filter / snippets_for 둘뿐이다 — 호출부는 사용자 원문 q만 넘기고,
LIKE 패턴 조립(이스케이프·와일드카드)은 이 모듈 안에서 끝난다. 라우터는 pattern을 모른다.

**호출부 계약**: q=''(빈/공백 문자열)는 이 모듈의 정규화 대상이 아니다 — '%%'는 전건
매칭이라 필터를 건 척만 하게 된다. 빈 문자열을 None으로 접는 건 HTTP 파라미터 위생이라
라우터 책임(limit·offset 클램프와 같은 자리).
"""
from sqlalchemy import or_, select, true
from sqlalchemy.ext.asyncio import AsyncSession

from rag.models import Conversation, Message

# 스니펫: 매칭 지점 전후 문자 수 (#28). 목록에서 한 줄로 표시되므로 총 30자 남짓이 상한이다.
# 처음엔 40이었는데 한글 기준으로 너무 길었다 — 영문 40자는 예닐곱 단어지만 한글 40자는
# 정보량이 훨씬 많아 제목보다 길어지고 시선이 분산된다. 반대로 10까지 줄이면 문장이 잘려
# 뜻이 안 잡히는 경우가 생겨 그 사이로 잡았다.
SNIPPET_RADIUS = 15

# LIKE 패턴에서 특수 의미를 갖는 문자. ESCAPE 절과 짝이며, 백슬래시가 먼저 나와야 한다
# (뒤에 두면 %·_를 이스케이프하며 넣은 백슬래시를 다시 이스케이프한다).
_LIKE_ESCAPE_CHAR = '\\'
_LIKE_SPECIALS = (_LIKE_ESCAPE_CHAR, '%', '_')


def _escape_like(raw: str) -> str:
    """사용자 입력을 LIKE 패턴 리터럴로 만든다 (#28).

    이스케이프하지 않으면 검색어의 %·_가 와일드카드로 동작한다 — 'a_b'로 검색하면
    'axb'까지 걸린다(실측). SQLAlchemy .ilike()는 자동 이스케이프를 하지 않으므로
    호출부가 escape=_LIKE_ESCAPE_CHAR를 함께 넘겨야 이 치환이 의미를 갖는다.
    """
    for ch in _LIKE_SPECIALS:
        raw = raw.replace(ch, _LIKE_ESCAPE_CHAR + ch)
    return raw


def _pattern(q: str) -> str:
    """search_filter·snippets_for가 공유하는 LIKE 패턴 조립 — 정의점 하나."""
    return f'%{_escape_like(q)}%'


def _build_snippet(content: str, q: str) -> str | None:
    """매칭 지점 전후 SNIPPET_RADIUS자를 잘라낸 발췌 (#28). 잘린 쪽엔 말줄임을 붙인다.

    위치 탐색을 SQL의 position()이 아니라 파이썬에서 하는 이유: position()은 대소문자를
    구분해서 ILIKE로 매칭된 영문 검색어를 못 찾고 0을 반환한다 — 그러면 조용히 '항상 앞
    80자'를 반환하는 오동작이 된다(실측). 여기선 양쪽 다 소문자로 맞춰 그 함정이 없다.
    또 이스케이프한 SQL 패턴이 아니라 원본 q를 쓴다 — 두 경로가 아예 분리돼 섞일 수 없다.

    파이썬 str 슬라이싱은 코드포인트 단위라 한글도 글자 수 그대로 맞는다.
    """
    idx = content.lower().find(q.lower())
    if idx < 0:
        # ILIKE는 매칭했는데 여기선 못 찾는 경우(유니코드 케이스폴딩 차이 등) — 발췌를 포기한다.
        # 대화 자체는 결과에 남고 snippet만 비므로, 목록이 깨지지 않는다.
        return None
    start = max(0, idx - SNIPPET_RADIUS)
    end = min(len(content), idx + len(q) + SNIPPET_RADIUS)
    prefix = '…' if start > 0 else ''
    suffix = '…' if end < len(content) else ''
    return f'{prefix}{content[start:end]}{suffix}'


def search_filter(tenant_id: str, q: str | None):
    """제목 또는 대화 내용 부분일치 필터 (#28). q=None이면 필터 없음(기존 목록 동작).

    메시지 매칭을 JOIN이 아니라 EXISTS로 거는 게 핵심이다 — JOIN이면 한 대화에서 두 건이
    매칭될 때 그 대화가 결과에 두 번 나오고 total도 부풀어 오른다(실측: EXISTS 2 vs JOIN 3).
    EXISTS는 세미조인이라 매칭 개수와 무관하게 대화 행이 정확히 하나다. 덕분에 DISTINCT나
    윈도우 함수 없이 "대화당 1행"이 저절로 성립한다.

    status로 거르지 않는다 — 차단 턴(blocked)도 검색에 걸려야 한다. 프롬프트 재료에서
    비정상 턴을 빼는 load_recent_messages와는 목적이 다르다(그건 맥락 오염 방지).
    attachments(첨부 추출 텍스트)는 건드리지 않는다 — 조회 API가 파일명만 노출하는 정책과
    같은 이유로, 고객 개인 문서 본문이 검색으로 새면 안 된다.
    """
    if q is None:
        return true()
    pattern = _pattern(q)
    return or_(
        Conversation.title.ilike(pattern, escape=_LIKE_ESCAPE_CHAR),
        select(Message.id)
        .where(Message.conversation_id == Conversation.id)
        .where(Message.tenant_id == tenant_id)      # 격리 — 상관 서브쿼리에도 WHERE 명시
        .where(Message.content.ilike(pattern, escape=_LIKE_ESCAPE_CHAR))
        .exists(),
    )


async def snippets_for(session: AsyncSession, tenant_id: str, conversation_ids: list[int],
                       q: str) -> dict[int, str | None]:
    """페이지에 실린 대화들의 발췌를 한 번의 쿼리로 모아온다 (#28).

    대화당 '가장 최근 매칭 메시지'를 골라야 하는데, (conversation_id, 최신순) 정렬로 받아
    각 대화의 첫 등장만 취하면 된다 — setdefault가 그 역할. created_at 동률일 때를 위해
    id 보조정렬을 함께 쓴다(한 턴의 user·assistant가 같은 커밋이라 시각이 같다).

    알려진 한계: 대화당 LIMIT을 걸지 않아 매칭이 많은 대화는 그 행을 전부 실어온다.
    윈도우 함수(row_number)나 LATERAL이면 DB에서 1건으로 줄일 수 있지만, 리포에 전례가
    없는 구문이라 현 데이터 규모에서는 단순함을 택했다 — 느려지면 그때 교체할 것.
    """
    pattern = _pattern(q)
    rows = (await session.execute(
        select(Message.conversation_id, Message.content)
        .where(Message.conversation_id.in_(conversation_ids))
        .where(Message.tenant_id == tenant_id)      # 격리 — 메시지에도 WHERE 명시
        .where(Message.content.ilike(pattern, escape=_LIKE_ESCAPE_CHAR))
        .order_by(Message.conversation_id, Message.created_at.desc(), Message.id.desc())
    )).all()

    latest: dict[int, str] = {}
    for conversation_id, content in rows:
        latest.setdefault(conversation_id, content)
    return {cid: _build_snippet(content, q) for cid, content in latest.items()}
