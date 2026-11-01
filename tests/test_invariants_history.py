"""불변식 대장(INVARIANTS.md)의 부분 칸 보강 — 턴 저장·이력 축(상태 어휘·화면=저장·이력 격리·표식·RETRY).

질의는 HTTP 경로(/kms/query)로 끝까지 태우고, 저장 결과는 새 세션으로 다시 읽어 본다(백그라운드 태스크가 자기 세션으로 커밋한다).
"""

import pytest
from sqlalchemy import select

from database import AsyncSessionLocal
from rag.models import Message, TurnStatus
from rag.turn.citation_labels import TAIL_START, citation_tail
from rag.turn.conversation import build_prior_turns, condense_query, load_recent_messages
from rag.turn.prompt_texts import CANCELLED_TURN_SUFFIX, FAILED_TURN_EMPTY
from rag.turn.turn_state import finalize_turn
from tests.conftest import register_faq, seed_turn, sse_answer, sse_meta
from tests.polling import wait_message_status

USER_A = {'X-User-Id': 'agent-a'}  # seed_turn이 만드는 대화의 소유자
RETRY_JSON = '{"safe": true, "intent": "RETRY"}'


async def _assistants(tenant_id: str, conversation_id: int | None = None) -> list[Message]:
    stmt = select(Message).where(Message.tenant_id == tenant_id).where(Message.role == 'assistant')
    if conversation_id is not None:
        stmt = stmt.where(Message.conversation_id == conversation_id)
    async with AsyncSessionLocal() as s:
        return list((await s.execute(stmt.order_by(Message.id))).scalars().all())


async def _ask(client, query: str, conversation_id: int | None = None):
    body = {'query': query} if conversation_id is None else {'query': query, 'conversation_id': conversation_id}
    res = await client.post('/kms/query', headers=USER_A, json=body)
    assert res.status_code == 200, res.text
    return res


# ── INV-32 상태 어휘 밖 값은 쓰기 시점에 거부 ─────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize('bad', ['parsing', 'embedding', 'DONE', 'ready', ''])
async def test_INV32_finalize는_다섯_값_밖의_상태를_저장하지_않는다(tenant_id, bad):
    """INV-32 TurnStatus 밖의 값은 쓰기 시점(finalize_turn — 모든 경로의 유일한 종료 지점)에 거부한다."""
    cid = await seed_turn(tenant_id, '질문', '', status='generating')
    (msg,) = await _assistants(tenant_id, cid)
    async with AsyncSessionLocal() as s:
        with pytest.raises(ValueError):
            await finalize_turn(s, tenant_id, msg.id, '답', [], status=bad)
        await s.commit()  # 거부가 늦어 속성이 이미 바뀌었다면 여기서 저장된다
    (after,) = await _assistants(tenant_id, cid)
    assert after.status == TurnStatus.GENERATING and after.content == ''


# ── INV-33 화면에 보낸 본문 = 저장되는 본문 ──────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'answer',
    [
        f'테스트 답변입니다. {citation_tail([1])}',
        f'붙은 꼬리 답변입니다.{citation_tail([1])}',
        f'잘린 꼬리 답변입니다. {TAIL_START}1',
        '꼬리 없는 답변입니다.',
    ],
    ids=['꼬리', '붙은_꼬리', '잘린_꼬리', '꼬리_없음'],
)
async def test_INV33_인용_꼬리_경로에서_SSE_누적_본문과_저장_본문이_같다(
    client, tenant_id, fake_llm, pass_gate, answer
):
    """INV-33 화면에 보낸 본문 = 저장되는 본문 — 인용 꼬리·잘린 버퍼는 어느 쪽에도 없다."""
    await register_faq(client)
    fake_llm.answer = answer
    res = await _ask(client, '환불 기간 알려줘')
    mid = sse_meta(res)['assistant_message_id']
    assert await wait_message_status(mid, 'done') == 'done'
    async with AsyncSessionLocal() as s:
        stored = (await s.get(Message, mid)).content
    shown = sse_answer(res)
    assert shown == stored, f'화면 {shown!r} ≠ 저장 {stored!r}'
    assert TAIL_START not in stored


# ── INV-44 generating 턴은 이력에서 짝째로 빠진다 ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_INV44_generating_턴은_짝째로_빠지고_앞뒤_짝짓기는_유지된다(tenant_id):
    """INV-44 프롬프트 이력에서 blocked·generating 턴은 짝째로 뺀다 — 진행 중(또는 고아) 자리표시가 이력에 끼면 안 된다."""
    cid = await seed_turn(tenant_id, '정상1', '답1', status='done')
    await seed_turn(tenant_id, '진행 중 질문', '', status='generating', conversation_id=cid)
    await seed_turn(tenant_id, '정상2', '답2', status='done', conversation_id=cid)
    async with AsyncSessionLocal() as s:
        msgs = await load_recent_messages(s, tenant_id, cid)
    assert [(m.role, m.content) for m in msgs] == [
        ('user', '정상1'),
        ('assistant', '답1'),
        ('user', '정상2'),
        ('assistant', '답2'),
    ]


# ── INV-45 표식은 조립 시점에만 — 저장본 불변 ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_INV45_취소_실패_표식은_모델_입력에만_붙고_저장본은_그대로다(client, tenant_id, fake_llm, pass_gate):
    """INV-45 취소·실패 표식은 조립 시점에만 붙이고 Message.content는 바꾸지 않는다 — 다음 턴을 끝까지 돌린 뒤 다시 읽는다."""
    await register_faq(client)
    cid = await seed_turn(tenant_id, '사이즈 가이드해줘', 'S(1): 평소 착용', status='cancelled')
    await seed_turn(tenant_id, '배송 기간?', '', status='failed', conversation_id=cid)

    seen: list[str] = []
    orig_complete, orig_stream = fake_llm.acomplete, fake_llm.astream

    async def complete_spy(messages, **kw):
        seen.extend(m['content'] for m in messages)
        return await orig_complete(messages, **kw)

    def stream_spy(messages, **kw):
        seen.extend(m['content'] for m in messages)
        return orig_stream(messages, **kw)

    fake_llm.acomplete, fake_llm.astream = complete_spy, stream_spy
    res = await _ask(client, '환불 기간 알려줘', cid)
    assert await wait_message_status(sse_meta(res)['assistant_message_id'], 'done') == 'done'

    assert any('S(1): 평소 착용' + CANCELLED_TURN_SUFFIX in c for c in seen), '취소 표식이 모델 입력에 없다'
    assert any(FAILED_TURN_EMPTY in c for c in seen), '실패 표식이 모델 입력에 없다'
    cancelled, failed, _new = await _assistants(tenant_id, cid)
    assert (cancelled.status, cancelled.content) == ('cancelled', 'S(1): 평소 착용')
    assert (failed.status, failed.content) == ('failed', '')


@pytest.mark.asyncio
async def test_INV45_이력_조립은_불러온_메시지_객체도_바꾸지_않는다(tenant_id, fake_llm):
    """같은 세션이 나중에 커밋해도 표식이 저장되지 않게 — 조립(prior_turns·condense) 뒤 세션에 바뀐 객체가 없어야 한다."""
    cid = await seed_turn(tenant_id, '사이즈 가이드해줘', 'S(1): 평소 착용', status='cancelled')
    await seed_turn(tenant_id, '배송 기간?', '', status='failed', conversation_id=cid)
    async with AsyncSessionLocal() as s:
        msgs = await load_recent_messages(s, tenant_id, cid)
        turns = build_prior_turns(msgs, budget_tokens=2000)
        await condense_query(fake_llm, '다시', msgs)
        assert turns[0]['a'] == 'S(1): 평소 착용' + CANCELLED_TURN_SUFFIX  # 표식은 조립 결과에만
        assert not s.dirty, f'조립이 메시지 객체를 바꿨다: {s.dirty}'
        await s.commit()
    cancelled, failed = await _assistants(tenant_id, cid)
    assert (cancelled.content, failed.content) == ('S(1): 평소 착용', '')


# ── INV-46 RETRY는 저장 계층에 닿지 않는다 ───────────────────────────────────────────────────────


async def _prev_cancelled_knowledge(tenant_id: str) -> int:
    return await seed_turn(tenant_id, '사이즈 가이드해줘', 'S(1)', status='cancelled', intent='KNOWLEDGE')


async def _prev_cancelled_other(tenant_id: str) -> int:
    return await seed_turn(tenant_id, '지금까지 요약해줘', '지금까지', status='cancelled', intent='OTHER')


async def _prev_done(tenant_id: str) -> int:
    return await seed_turn(tenant_id, '환불 기간?', '14일입니다', status='done', intent='KNOWLEDGE')


async def _no_history(tenant_id: str) -> None:
    return None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('seed', 'stored_intent'),
    [
        (_prev_cancelled_knowledge, 'KNOWLEDGE'),
        (_prev_cancelled_other, 'OTHER'),
        (_prev_done, 'OTHER'),
        (_no_history, 'OTHER'),
    ],
    ids=['직전_취소_KNOWLEDGE', '직전_취소_OTHER', '직전_완료', '이력_없음'],
)
async def test_INV46_RETRY_분류는_어느_갈래로_해소돼도_messages_intent에_남지_않는다(
    client, tenant_id, fake_llm, pass_gate, seed, stored_intent
):
    """INV-46 RETRY 인텐트는 저장 계층에 닿지 않는다 — 분류기가 RETRY를 내도 저장 라벨은 해소된 갈래다."""
    await register_faq(client)
    cid = await seed(tenant_id)
    fake_llm.intent_json = RETRY_JSON
    res = await _ask(client, '다시', cid)
    mid = sse_meta(res)['assistant_message_id']
    assert await wait_message_status(mid, 'done') == 'done'

    rows = await _assistants(tenant_id)
    assert [r.intent for r in rows if r.intent == 'RETRY'] == []
    assert next(r for r in rows if r.id == mid).intent == stored_intent
