"""대화 API 경계·예외 — 메시지 조회·삭제·피드백·취소·재접속의 404 갈래, 응답 필드 필터, 목록 파라미터 보정.

HTTP로만 확인한다. 턴은 DB에 직접 심거나(상태 조합), 가짜 LLM을 생성 도중 멈춰 진행 중 턴을 만든다.
"""

import asyncio

import pytest
from sqlalchemy import select

from database import AsyncSessionLocal
from rag import clients
from rag.models import Conversation, Message
from tests.conftest import register_faq, sse_events, sse_meta
from tests.polling import wait_inflight

NO_CONV = {'detail': '대화를 찾을 수 없습니다.'}
NO_MSG = {'detail': '메시지를 찾을 수 없습니다.'}
NOT_RUNNING = {'detail': '진행 중인 생성이 아닙니다.'}


async def _seed(
    tenant: str,
    *,
    status: str = 'done',
    created_by: str = 'test-user',
    sources: list | None = None,
    cited_docs: list | None = None,
    attachments: list | None = None,
) -> tuple[int, int, int]:
    """대화 1개 + user/assistant 한 턴 → (conversation_id, user_id, assistant_id)."""
    async with AsyncSessionLocal() as s:
        conv = Conversation(tenant_id=tenant, created_by=created_by)
        s.add(conv)
        await s.flush()
        u = Message(
            tenant_id=tenant, conversation_id=conv.id, role='user', content='반품 기간?', attachments=attachments
        )
        s.add(u)
        await s.flush()
        a = Message(
            tenant_id=tenant,
            conversation_id=conv.id,
            role='assistant',
            content='14일입니다',
            status=status,
            question_message_id=u.id,
            sources=sources,
            cited_docs=cited_docs,
        )
        s.add(a)
        await s.flush()
        ids = (conv.id, u.id, a.id)
        await s.commit()
    return ids


async def _start_paused(client, fake_llm) -> tuple[asyncio.Task, int, int]:
    """생성 도중 멈춘 질의 — (task, conversation_id, assistant_message_id)."""
    fake_llm.pause_after_tokens = 2
    task = asyncio.create_task(client.post('/kms/query', json={'query': '환불 기간 알려줘'}))
    await asyncio.wait_for(fake_llm.paused.wait(), timeout=10)
    async with AsyncSessionLocal() as s:
        a = (
            (await s.execute(select(Message).where(Message.status == 'generating').order_by(Message.id.desc())))
            .scalars()
            .first()
        )
    return task, a.conversation_id, a.id


async def _resume(fake_llm, task: asyncio.Task, tenant: str):
    fake_llm.pause_after_tokens = None
    fake_llm.resume.set()
    res = await task
    assert await wait_inflight(tenant, 0) == 0
    return res


# ── 조회·삭제 ──


@pytest.mark.asyncio
async def test_메시지_조회는_id_오름차순이고_없는_대화는_404(client, tenant_id):
    """CNV-15 — 남·삭제는 기존 테스트가, 여기서는 순서와 없는 id."""
    cid, uid, aid = await _seed(tenant_id)
    msgs = (await client.get(f'/kms/conversations/{cid}/messages')).json()
    assert [m['message_id'] for m in msgs] == [uid, aid]
    res = await client.get('/kms/conversations/999999999/messages')
    assert res.status_code == 404 and res.json() == NO_CONV


@pytest.mark.asyncio
async def test_메시지_응답은_실인용만_싣고_첨부는_파일명만_내린다(client, tenant_id):
    """CNV-20 — sources 중 cited_docs에 없는 문서는 걸러지고, 걸러져 비면 None. 첨부 본문은 응답에 없다."""
    src = [
        {'document_id': 1, 'filename': '환불정책.md', 'version': 1},
        {'document_id': 2, 'filename': '배송정책.md', 'version': 1},
    ]
    cid, _, _ = await _seed(
        tenant_id, sources=src, cited_docs=['환불정책.md'], attachments=[{'filename': '약관.md', 'text': '비밀 본문'}]
    )
    res = await client.get(f'/kms/conversations/{cid}/messages')
    user, assistant = res.json()
    assert user['attachments'] == ['약관.md'] and '비밀 본문' not in res.text
    assert [c['filename'] for c in assistant['citations']] == ['환불정책.md']

    cid2, _, _ = await _seed(tenant_id, sources=src, cited_docs=[])
    assert (await client.get(f'/kms/conversations/{cid2}/messages')).json()[1]['citations'] is None


@pytest.mark.asyncio
async def test_메시지_조회는_진행_중_차단_상태를_그대로_보이고_바꾸지_않는다(client, tenant_id):
    """CNV-20 — GET은 읽기 전용이다(스윕은 워커만 한다)."""
    for status in ('generating', 'blocked'):
        cid, _, aid = await _seed(tenant_id, status=status)
        assert (await client.get(f'/kms/conversations/{cid}/messages')).json()[1]['status'] == status
        async with AsyncSessionLocal() as s:
            assert (await s.get(Message, aid)).status == status


@pytest.mark.asyncio
async def test_대화_삭제는_메시지를_보존하고_재삭제_남의_없는_id는_404(client, tenant_id):
    """CNV-30."""
    cid, _, _ = await _seed(tenant_id)
    assert (await client.delete(f'/kms/conversations/{cid}')).status_code == 204
    async with AsyncSessionLocal() as s:
        assert (await s.get(Conversation, cid)).deleted_at is not None
        count = len((await s.execute(select(Message).where(Message.conversation_id == cid))).scalars().all())
    assert count == 2

    other, _, _ = await _seed(tenant_id, created_by='agent-b')
    for target in (cid, other, 999999999):
        res = await client.delete(f'/kms/conversations/{target}')
        assert res.status_code == 404 and res.json() == NO_CONV, target


@pytest.mark.asyncio
async def test_목록_파라미터는_보정되고_비정수는_422(client, tenant_id):
    """CNV-03 — limit 0·음수·51은 50으로, offset 음수는 0으로, 범위 밖 offset은 빈 목록. 비정수는 422."""
    await _seed(tenant_id)
    for limit in (0, -1, 51):
        res = await client.get('/kms/conversations', params={'limit': limit})
        assert res.status_code == 200 and len(res.json()['items']) == 1, limit
    assert len((await client.get('/kms/conversations', params={'offset': -5})).json()['items']) == 1
    page = (await client.get('/kms/conversations', params={'offset': 100})).json()
    assert page['items'] == [] and page['has_more'] is False and page['total'] == 1
    for bad in ({'limit': 'abc'}, {'offset': '1.5'}):
        assert (await client.get('/kms/conversations', params=bad)).status_code == 422, bad


# ── 피드백 ──


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'body',
    [
        {'feedback': None, 'tag': 'wrong_info'},
        {},
        {'feedback': False, 'text': 'ㄱ' * 501},
    ],
    ids=['null-with-tag', 'missing', 'text-501'],
)
async def test_피드백_바디_검증은_422(client, tenant_id, body):
    """CNV-37 — 미정의 태그·👍에 태그는 기존 테스트가 고정."""
    _, _, aid = await _seed(tenant_id)
    assert (await client.patch(f'/kms/messages/{aid}/feedback', json=body)).status_code == 422


@pytest.mark.asyncio
async def test_피드백_텍스트_500자는_저장된다(client, tenant_id):
    _, _, aid = await _seed(tenant_id)
    res = await client.patch(f'/kms/messages/{aid}/feedback', json={'feedback': False, 'text': 'ㄱ' * 500})
    assert res.status_code == 200 and res.json()['feedback_text'] == 'ㄱ' * 500


@pytest.mark.asyncio
async def test_피드백_대상이_아니면_404(client, tenant_id):
    """CNV-45 — done이 아닌 턴(생성 중·차단·취소)과 삭제된 대화의 메시지. user·failed는 기존 테스트가 고정."""
    for status in ('generating', 'blocked', 'cancelled'):
        _, _, aid = await _seed(tenant_id, status=status)
        res = await client.patch(f'/kms/messages/{aid}/feedback', json={'feedback': True})
        assert res.status_code == 404 and res.json() == NO_MSG, status

    cid, _, aid = await _seed(tenant_id)
    assert (await client.delete(f'/kms/conversations/{cid}')).status_code == 204
    res = await client.patch(f'/kms/messages/{aid}/feedback', json={'feedback': True})
    assert res.status_code == 404 and res.json() == NO_MSG


# ── 취소·재접속 ──


@pytest.mark.asyncio
async def test_취소_대상이_아니면_갈래별_404(client, tenant_id):
    """CNV-52 — 끝난·차단 턴은 '진행 중인 생성이 아닙니다', user 메시지·삭제된 대화는 '메시지를 찾을 수 없습니다'."""
    for status in ('done', 'blocked', 'failed'):
        _, _, aid = await _seed(tenant_id, status=status)
        res = await client.post(f'/kms/messages/{aid}/cancel')
        assert res.status_code == 404 and res.json() == NOT_RUNNING, status

    cid, uid, aid = await _seed(tenant_id, status='generating')
    res = await client.post(f'/kms/messages/{uid}/cancel')
    assert res.status_code == 404 and res.json() == NO_MSG
    assert (await client.delete(f'/kms/conversations/{cid}')).status_code == 204
    res = await client.post(f'/kms/messages/{aid}/cancel')
    assert res.status_code == 404 and res.json() == NO_MSG


@pytest.mark.asyncio
async def test_재접속_대상이_없으면_404이고_Redis가_죽으면_503(client, tenant_id, monkeypatch):
    """CNV-68 — 없는 id·user 메시지·삭제된 대화의 메시지는 소유 검증에서 404. 스트림 조회 실패는 503."""
    cid, uid, aid = await _seed(tenant_id, status='generating')
    for target in (999999999, uid):
        res = await client.get(f'/kms/messages/{target}/events')
        assert res.status_code == 404 and res.json() == NO_MSG, target

    class _DownRedis:
        async def exists(self, *a: object) -> int:
            raise ConnectionError('redis down')

    monkeypatch.setattr(clients, 'shared_redis', _DownRedis())
    res = await client.get(f'/kms/messages/{aid}/events')
    assert res.status_code == 503
    assert res.json() == {'detail': '일시적으로 처리할 수 없습니다. 잠시 후 다시 시도해 주세요.'}

    monkeypatch.undo()
    assert (await client.delete(f'/kms/conversations/{cid}')).status_code == 204
    res = await client.get(f'/kms/messages/{aid}/events')
    assert res.status_code == 404 and res.json() == NO_MSG


@pytest.mark.asyncio
async def test_생성_중_대화를_삭제하면_생성은_완주하고_취소_재접속은_404(client, tenant_id, fake_llm, pass_gate):
    """CNV-33 — 스펙으로 확정한 현재 동작: 삭제는 진행 중 생성을 멈추지 않는다."""
    await register_faq(client)
    task, cid, aid = await _start_paused(client, fake_llm)
    try:
        assert (await client.delete(f'/kms/conversations/{cid}')).status_code == 204
        assert (await client.post(f'/kms/messages/{aid}/cancel')).json() == NO_MSG
        assert (await client.get(f'/kms/messages/{aid}/events')).json() == NO_MSG
    finally:
        res = await _resume(fake_llm, task, tenant_id)
    assert [e for e, _ in sse_events(res)][-1] == 'done'
    async with AsyncSessionLocal() as s:
        assert (await s.get(Message, aid)).status == 'done'


@pytest.mark.asyncio
async def test_재접속은_Last_Event_ID를_무시하고_처음부터_재생한다(client, tenant_id, fake_llm, pass_gate):
    """CNV-72 — 스펙으로 확정한 현재 동작. 헤더가 있어도 meta부터 전체가 온다."""
    await register_faq(client)
    res = await client.post('/kms/query', json={'query': '환불 기간 알려줘'})
    aid = sse_meta(res)['assistant_message_id']
    assert await wait_inflight(tenant_id, 0) == 0

    plain = await client.get(f'/kms/messages/{aid}/events')
    with_header = await client.get(f'/kms/messages/{aid}/events', headers={'Last-Event-ID': '3'})
    assert plain.status_code == with_header.status_code == 200
    names = [e for e, _ in sse_events(with_header)]
    assert names[0] == 'meta' and names[-1] == 'done'
    assert names == [e for e, _ in sse_events(plain)]
