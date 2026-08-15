"""대화 히스토리 검색(#28) — q 부분일치·스니펫·total.

실제 앱 + DB (conftest client 패턴). LLM 불필요 — 검색은 조회 경로라 생성이 끼지 않는다.

핵심 불변식 두 개를 특히 지킨다:
  1. 한 대화에서 여러 메시지가 매칭돼도 결과·total은 대화 1건 (EXISTS 세미조인)
  2. 첨부 본문(attachments)은 검색 대상이 아니다 — 조회 API가 파일명만 주는 정책과 같은 이유
"""
from datetime import datetime, timedelta

import pytest
from httpx import ASGITransport, AsyncClient

from database import AsyncSessionLocal
from rag.models import Conversation, Message
from routers.conversations import SNIPPET_RADIUS, _build_snippet, _escape_like

USER_A = {'X-User-Id': 'agent-a'}
USER_B = {'X-User-Id': 'agent-b'}


async def _seed_conv(tenant_id: str, title: str | None, contents: list[str],
                     created_by: str = 'agent-a', minutes: int = 0) -> int:
    """대화 1건 + 메시지 여러 건. last_used_at을 분 단위로 벌려 정렬을 결정적이게.

    메시지의 created_at은 **전부 같은 값**을 준다 — 운영에서 한 턴의 user·assistant가
    같은 커밋이라 now()가 동률이 되는 걸 그대로 재현한다(rag/conversation.py:81 참조).
    시간을 인위적으로 벌리면 '최근 매칭' 선택이 created_at만으로 결정돼, 실제로 동작하는
    id 보조정렬이 검증되지 않는다.
    """
    base = datetime.now()   # 모델 매핑이 naive — aware를 넣으면 asyncpg가 거부
    async with AsyncSessionLocal() as s:
        conv = Conversation(tenant_id=tenant_id, created_by=created_by, title=title,
                            last_used_at=base + timedelta(minutes=minutes))
        s.add(conv)
        await s.flush()
        for i, content in enumerate(contents):
            s.add(Message(conversation_id=conv.id, tenant_id=tenant_id,
                          role='user' if i % 2 == 0 else 'assistant', content=content,
                          created_at=base + timedelta(minutes=minutes)))
        cid = conv.id
        await s.commit()
    return cid


async def _search(client, q: str, **params) -> dict:
    query = '&'.join(f'{k}={v}' for k, v in {'q': q, **params}.items())
    res = await client.get(f'/kms/conversations?{query}', headers=USER_A)
    assert res.status_code == 200, res.text
    return res.json()


# ── 순수 함수 (DB 불필요) ────────────────────────────────────

def test_스니펫_중간매칭은_양쪽_말줄임():
    content = '가' * 100 + '배송' + '나' * 100
    snippet = _build_snippet(content, '배송')
    assert snippet.startswith('…') and snippet.endswith('…')
    assert '배송' in snippet
    # 전후 SNIPPET_RADIUS자 + 검색어 + 말줄임 2개. 상수를 참조해 값이 바뀌어도 계약만 검증한다.
    assert len(snippet) == SNIPPET_RADIUS * 2 + len('배송') + 2


def test_스니펫_경계에서는_해당쪽_말줄임_없음():
    assert _build_snippet('배송 문의', '배송') == '배송 문의'          # 원문이 창보다 짧음
    assert _build_snippet('배송' + '가' * 200, '배송').startswith('배송')   # 맨 앞 매칭
    assert _build_snippet('가' * 200 + '배송', '배송').endswith('배송')     # 맨 끝 매칭


def test_스니펫_대소문자_무시():
    """position()을 SQL에서 쓰면 대소문자를 구분해 조용히 앞부분만 잘라준다 — 그 함정 회귀."""
    content = '앞부분 설명이 충분히 길어야 함' * 5 + ' RefundPolicy 문서를 참조'
    snippet = _build_snippet(content, 'refundpolicy')
    assert 'RefundPolicy' in snippet


def test_스니펫_매칭_못찾으면_None():
    assert _build_snippet('내용', '없는말') is None


def test_이스케이프_와일드카드():
    assert _escape_like('a_b') == r'a\_b'
    assert _escape_like('50%') == r'50\%'
    assert _escape_like(r'a\b') == r'a\\b'      # 백슬래시가 먼저 — 이후 치환분을 재이스케이프하지 않는다


# ── 검색 동작 ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_제목_매칭은_스니펫_없음(client, tenant_id):
    cid = await _seed_conv(tenant_id, '배송 문의', ['언제 오나요'])
    body = await _search(client, '배송')
    assert [i['conversation_id'] for i in body['items']] == [cid]
    assert body['items'][0]['snippet'] is None    # 제목은 이미 title로 보인다
    assert body['total'] == 1


@pytest.mark.asyncio
async def test_동률_created_at이면_나중에_저장된_매칭이_이긴다(client, tenant_id):
    """운영의 정상 데이터 모양 — 한 턴의 user·assistant는 커밋이 같아 created_at이 동률이다.

    그래서 '가장 최근 매칭'을 시각만으로는 못 고르고 id 보조정렬이 실제로 일한다.
    (_seed_conv가 created_at을 일부러 같게 넣는 이유이기도 하다.)
    """
    cid = await _seed_conv(tenant_id, '반품 규정', [
        '교환은 7일 이내 가능하며 배송은 3~5영업일이 걸립니다',
        '배송 지연 시 고객센터로 문의해 주세요',      # 같은 시각, 더 큰 id — 이쪽이 뽑혀야
        '추가 문의는 없습니다',
    ])
    body = await _search(client, '배송')
    assert [i['conversation_id'] for i in body['items']] == [cid]
    assert '고객센터' in body['items'][0]['snippet']


@pytest.mark.asyncio
async def test_턴이_다르면_created_at_최신이_이긴다(client, tenant_id):
    """턴이 다르면 커밋이 달라 created_at도 다르다 — 이땐 시각이 먼저 판단한다."""
    cid = await _seed_conv(tenant_id, '반품 규정', ['배송 지연 시 고객센터로 문의해 주세요'])
    async with AsyncSessionLocal() as s:                       # 뒤이은 턴 (시각이 더 나중)
        s.add(Message(conversation_id=cid, tenant_id=tenant_id, role='user',
                      content='배송비는 누가 부담하나요',
                      created_at=datetime.now() + timedelta(minutes=10)))
        await s.commit()
    body = await _search(client, '배송')
    assert '배송비' in body['items'][0]['snippet']


@pytest.mark.asyncio
async def test_한_대화에_여러_매칭이어도_total은_1(client, tenant_id):
    """JOIN으로 짜면 조용히 부풀어 오르는 지점 — EXISTS 세미조인의 존재 이유."""
    await _seed_conv(tenant_id, '문의', ['배송 질문', '배송 답변', '배송 재질문'])
    body = await _search(client, '배송')
    assert body['total'] == 1
    assert len(body['items']) == 1


@pytest.mark.asyncio
async def test_매칭_없으면_빈_결과(client, tenant_id):
    await _seed_conv(tenant_id, '급여 문의', ['명세서 발급 방법'])
    body = await _search(client, '배송')
    assert body == {'items': [], 'has_more': False, 'total': 0}


@pytest.mark.asyncio
async def test_첨부_본문은_검색되지_않는다(client, tenant_id):
    """첨부 추출 텍스트는 고객 개인 문서 — 검색으로 우회 노출되면 조회 정책이 깨진다."""
    async with AsyncSessionLocal() as s:
        conv = Conversation(tenant_id=tenant_id, created_by='agent-a', title='첨부 대화')
        s.add(conv)
        await s.flush()
        s.add(Message(conversation_id=conv.id, tenant_id=tenant_id, role='user', content='이것 좀 봐줘',
                      attachments=[{'filename': '계약서.pdf', 'text': '위약금 조항 상세'}]))
        await s.commit()
    assert (await _search(client, '위약금'))['items'] == []


@pytest.mark.asyncio
async def test_차단된_턴의_본문도_검색된다(client, tenant_id):
    """blocked는 프롬프트 재료에서만 빠진다(load_recent_messages) — 조회·검색에선 보인다.

    검색어를 차단 메시지에만 두어 '그 메시지 본문이 걸렸는지'를 직접 확인한다
    (같은 대화의 user 메시지로 걸리면 status 필터 유무를 검증하지 못한다).
    """
    cid = await _seed_conv(tenant_id, '문의', ['민감한 질문입니다'])
    async with AsyncSessionLocal() as s:
        s.add(Message(conversation_id=cid, tenant_id=tenant_id, role='assistant',
                      content='개인정보라 답변할수없습니다', status='blocked'))
        await s.commit()
    body = await _search(client, '개인정보')
    assert [i['conversation_id'] for i in body['items']] == [cid]
    assert '개인정보' in body['items'][0]['snippet']    # 발췌도 차단 메시지에서 나온다


@pytest.mark.asyncio
async def test_와일드카드_이스케이프(client, tenant_id):
    """이스케이프가 없으면 'a_b'가 'axb'까지 잡는다 (실측 확인된 오탐)."""
    literal = await _seed_conv(tenant_id, 'a_b 파일명 규칙', ['내용'], minutes=1)
    await _seed_conv(tenant_id, 'axb 파일명 규칙', ['내용'], minutes=2)
    body = await _search(client, 'a_b')
    assert [i['conversation_id'] for i in body['items']] == [literal]


@pytest.mark.asyncio
async def test_대소문자_무시_매칭(client, tenant_id):
    cid = await _seed_conv(tenant_id, 'RefundPolicy 문서', ['본문'])
    assert [i['conversation_id'] for i in (await _search(client, 'refundpolicy'))['items']] == [cid]


@pytest.mark.asyncio
async def test_한글자_검색도_동작(client, tenant_id):
    """최소 2자는 FE의 UX 기준 — 서버는 특수 분기 없이 그대로 검색한다."""
    cid = await _seed_conv(tenant_id, '배송 문의', ['내용'])
    assert [i['conversation_id'] for i in (await _search(client, '배'))['items']] == [cid]


@pytest.mark.asyncio
async def test_공백_q는_미전송과_동일(client, tenant_id):
    """'%%'는 전건 매칭이라 필터를 건 척만 하게 된다 — 전체 목록으로 취급."""
    await _seed_conv(tenant_id, '배송', ['내용'], minutes=1)
    await _seed_conv(tenant_id, '급여', ['내용'], minutes=2)
    assert (await _search(client, '%20'))['total'] == 2       # q=' ' (공백)


@pytest.mark.asyncio
async def test_검색_페이지네이션_total과_has_more(client, tenant_id):
    """검색은 소유 대화 전체를 훑고 자르기만 한다 — total은 페이지가 아니라 전체 기준."""
    ids = [await _seed_conv(tenant_id, f'배송 문의 {i}', ['내용'], minutes=i) for i in range(3)]
    page1 = await _search(client, '배송', limit=2)
    assert len(page1['items']) == 2 and page1['total'] == 3 and page1['has_more'] is True
    page2 = await _search(client, '배송', limit=2, offset=2)
    assert len(page2['items']) == 1 and page2['total'] == 3 and page2['has_more'] is False
    got = [i['conversation_id'] for i in page1['items'] + page2['items']]
    assert got == list(reversed(ids))                          # 이어붙이면 전체 최근순

    # offset이 total을 넘어도 total은 실제 값 그대로 (빈 페이지 ≠ 결과 없음)
    beyond = await _search(client, '배송', limit=2, offset=10)
    assert beyond['items'] == [] and beyond['total'] == 3 and beyond['has_more'] is False


@pytest.mark.asyncio
async def test_남의_대화는_검색에_안_걸린다(client, tenant_id):
    await _seed_conv(tenant_id, '배송 문의', ['배송 내용'], created_by='agent-b')
    assert (await _search(client, '배송')) == {'items': [], 'has_more': False, 'total': 0}


@pytest.mark.asyncio
async def test_소프트삭제된_대화는_검색에_안_걸린다(client, tenant_id):
    cid = await _seed_conv(tenant_id, '배송 문의', ['배송 내용'])
    assert (await client.delete(f'/kms/conversations/{cid}', headers=USER_A)).status_code == 204
    assert (await _search(client, '배송'))['items'] == []


@pytest.mark.asyncio
async def test_타_테넌트_대화는_검색에_안_걸린다(client, tenant_id, other_tenant_id):
    """격리의 마지막 표면 — 검색이 messages 조인으로 확장돼도 WHERE 절이 유지되는지."""
    from main import app

    cid = await _seed_conv(tenant_id, '배송 문의', ['배송 내용'])
    async with AsyncClient(transport=ASGITransport(app=app), base_url='http://testserver',
                           headers={'X-Tenant-Id': other_tenant_id}) as other:
        res = await other.get('/kms/conversations?q=배송', headers=USER_A)
        assert all(c['conversation_id'] != cid for c in res.json()['items'])
        assert res.json()['total'] == 0
