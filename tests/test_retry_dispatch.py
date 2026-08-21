"""RETRY 디스패치 (#59) — "다시"류 재요청의 결정론 해소.

분류기는 "재요청 발화"라는 표면 사실만 인식하고(RouteDecision.intent='RETRY'),
무엇을 다시 할지는 prepare()의 디스패처가 직전 턴 상태로 결정한다:
  직전 cancelled → 그 턴의 실질 질문·standalone으로 원래 intent 경로 재실행
  그 외(done/이력 없음)   → OTHER (기존 회상·재설명 동작)
RETRY는 저장 계층에 도달하지 않는 전이 인텐트 — route·intent 라벨은 기존 어휘 그대로다.
"""
from types import SimpleNamespace

import pytest
from sqlalchemy import select

from database import AsyncSessionLocal
from rag.conversation import last_cancelled_turn
from rag.models import Conversation, Message
from rag.service import RagService
from tests.conftest import register_faq, seed_turn

RETRY_JSON = '{"safe": true, "intent": "RETRY"}'
USER_A = {'X-User-Id': 'agent-a'}


# ── 단위: last_cancelled_turn ────────────────────────────────

def _m(role, status='done', mid=1, qid=None):
    return SimpleNamespace(role=role, status=status, id=mid, question_message_id=qid,
                           content=f'{role}-{mid}', standalone_query=None, intent=None)


class TestLastCancelledTurn:
    def test_직전_취소_짝을_돌려준다(self):
        u, a = _m('user', mid=1), _m('assistant', status='cancelled', mid=2, qid=1)
        assert last_cancelled_turn([u, a]) == (u, a)

    def test_직전이_done이면_None(self):
        msgs = [_m('user', mid=1), _m('assistant', status='done', mid=2, qid=1)]
        assert last_cancelled_turn(msgs) is None

    def test_이력_없으면_None(self):
        assert last_cancelled_turn([]) is None
        assert last_cancelled_turn([_m('user')]) is None

    def test_페어링_불변식이_깨지면_None(self):
        # user 바로 다음이 아닌 assistant(question_message_id 불일치) — 크래시 대신 폴백
        u, a = _m('user', mid=1), _m('assistant', status='cancelled', mid=3, qid=99)
        assert last_cancelled_turn([u, a]) is None

    def test_같은_standalone_체인은_머리까지_되감는다(self):
        u1 = _m('user', mid=1); u1.standalone_query = 'X'
        a1 = _m('assistant', status='cancelled', mid=2, qid=1)
        u2 = _m('user', mid=3); u2.standalone_query = 'X'      # 재실행 턴("다시" 저장, standalone 물림)
        a2 = _m('assistant', status='cancelled', mid=4, qid=3)
        assert last_cancelled_turn([u1, a1, u2, a2]) == (u1, a2)

    def test_standalone이_다르면_되감기_정지(self):
        u1 = _m('user', mid=1); u1.standalone_query = 'X'
        a1 = _m('assistant', status='cancelled', mid=2, qid=1)
        u2 = _m('user', mid=3); u2.standalone_query = 'Y'      # 다른 질문의 연속 취소
        a2 = _m('assistant', status='cancelled', mid=4, qid=3)
        assert last_cancelled_turn([u1, a1, u2, a2]) == (u2, a2)


# ── 통합: 디스패처 ───────────────────────────────────────────

async def _seed_cancelled(tenant_id: str, question: str, partial: str,
                          standalone: str | None = None, intent: str = 'KNOWLEDGE',
                          conversation_id: int | None = None) -> int:
    """취소로 끝난 턴 시딩 — conftest.seed_turn 위임 (헬퍼 단일화, 리뷰 반영)."""
    return await seed_turn(tenant_id, question, partial, status='cancelled',
                           standalone=standalone, intent=intent, conversation_id=conversation_id)


@pytest.mark.asyncio
async def test_RETRY_직전취소는_그_질문으로_KNOWLEDGE_재실행(client, tenant_id, fake_llm, fake_embed, pass_gate):
    await register_faq(client)
    cid = await _seed_cancelled(tenant_id, '사이즈 가이드해줘', 'S(1): 평소',
                                standalone='티셔츠 사이즈별 착용 가이드')
    fake_llm.intent_json = RETRY_JSON

    async with AsyncSessionLocal() as s:
        svc = RagService(tenant_id=tenant_id, session=s, user_id='agent-a')
        prepared = await svc.prepare('다시얘기해줘야지', conversation_id=cid)

    assert prepared.route == 'knowledge'
    assert prepared.original_query == '사이즈 가이드해줘'            # 검색·프롬프트용 실질 질문
    assert prepared.display_query == '다시얘기해줘야지'              # 저장·화면용 원 발화
    assert prepared.standalone_query == '티셔츠 사이즈별 착용 가이드'  # 원본 검색 재현 (condense 스킵)
    assert 'condense' not in fake_llm.calls                          # LLM 재작성 미호출
    assert prepared.intent_label == 'KNOWLEDGE'                      # RETRY는 저장 어휘에 없다


@pytest.mark.asyncio
async def test_RETRY_재실행_턴의_user_메시지는_원_발화로_저장(client, tenant_id, fake_llm, fake_embed, pass_gate):
    """기록엔 사용자가 실제 친 말("다시")만 남는다 — 화면=기록 진실성 (#59 결정)."""
    await register_faq(client)
    cid = await _seed_cancelled(tenant_id, '사이즈 가이드해줘', 'S(1): 평소')
    fake_llm.intent_json = RETRY_JSON

    async with AsyncSessionLocal() as s:
        svc = RagService(tenant_id=tenant_id, session=s, user_id='agent-a')
        prepared = await svc.prepare('다시', conversation_id=cid)
        await svc.begin_turn(prepared)

    async with AsyncSessionLocal() as s:
        users = (await s.execute(
            select(Message).where(Message.conversation_id == cid)
            .where(Message.role == 'user').order_by(Message.id)
        )).scalars().all()
    assert [u.content for u in users] == ['사이즈 가이드해줘', '다시']


@pytest.mark.asyncio
async def test_RETRY_직전완료는_OTHER_회상(client, tenant_id, fake_llm):
    """완료된 턴 뒤의 "다시"는 재생성이 아니라 회상 — 기존 OTHER 동작 유지."""
    async with AsyncSessionLocal() as s:
        conv = Conversation(tenant_id=tenant_id, created_by='agent-a')
        s.add(conv)
        await s.flush()
        u = Message(tenant_id=tenant_id, conversation_id=conv.id, role='user', content='환불 기간?')
        s.add(u)
        await s.flush()
        s.add(Message(tenant_id=tenant_id, conversation_id=conv.id, role='assistant',
                      content='14일입니다', status='done', question_message_id=u.id, intent='KNOWLEDGE'))
        cid = conv.id
        await s.commit()
    fake_llm.intent_json = RETRY_JSON

    async with AsyncSessionLocal() as s:
        svc = RagService(tenant_id=tenant_id, session=s, user_id='agent-a')
        prepared = await svc.prepare('다시 말해줘', conversation_id=cid)
    assert prepared.route == 'other'
    assert prepared.original_query == '다시 말해줘'                  # 스왑 없음
    assert prepared.display_query is None


@pytest.mark.asyncio
async def test_RETRY_이력없음은_OTHER_폴백(client, tenant_id, fake_llm):
    fake_llm.intent_json = RETRY_JSON
    async with AsyncSessionLocal() as s:
        svc = RagService(tenant_id=tenant_id, session=s, user_id='agent-a')
        prepared = await svc.prepare('다시')
    assert prepared.route == 'other'


@pytest.mark.asyncio
async def test_RETRY_원래_OTHER턴_취소는_OTHER로_재실행(client, tenant_id, fake_llm):
    """'지금까지 요약해줘' 취소 후 "다시" — 검색이 아니라 원래 경로(OTHER)로, 실질 질문으로."""
    cid = await _seed_cancelled(tenant_id, '지금까지 요약해줘', '지금까지의 대화를', intent='OTHER')
    fake_llm.intent_json = RETRY_JSON

    async with AsyncSessionLocal() as s:
        svc = RagService(tenant_id=tenant_id, session=s, user_id='agent-a')
        prepared = await svc.prepare('다시', conversation_id=cid)
    assert prepared.route == 'other'
    assert prepared.original_query == '지금까지 요약해줘'            # OTHER 생성이 볼 실질 질문
    assert prepared.display_query == '다시'


@pytest.mark.asyncio
async def test_RETRY_체인은_실질_질문과_검색어를_모두_복원(client, tenant_id, fake_llm, fake_embed, pass_gate):
    """"다시" 연타 — 재실행 턴의 user 행엔 "다시"(display)만 남지만, 같은 standalone으로
    연결된 연속 취소 짝을 되감아 실질 질문까지 복원한다 (리뷰 발견: 질문 슬롯 퇴화 방지)."""
    await register_faq(client)
    # Turn N: 실질 질문 취소 → Turn N+1: 재실행("다시" 저장, standalone 물림)도 취소
    cid = await _seed_cancelled(tenant_id, '사이즈 가이드해줘', 'S(1): 평소',
                                standalone='티셔츠 사이즈별 착용 가이드')
    await _seed_cancelled(tenant_id, '다시', '',
                          standalone='티셔츠 사이즈별 착용 가이드', conversation_id=cid)
    fake_llm.intent_json = RETRY_JSON

    async with AsyncSessionLocal() as s:
        svc = RagService(tenant_id=tenant_id, session=s, user_id='agent-a')
        prepared = await svc.prepare('다시', conversation_id=cid)
    assert prepared.route == 'knowledge'
    assert prepared.original_query == '사이즈 가이드해줘'               # 질문 슬롯 퇴화 없음
    assert prepared.standalone_query == '티셔츠 사이즈별 착용 가이드'   # 검색도 원본 그대로
    assert prepared.display_query == '다시'


@pytest.mark.asyncio
async def test_RETRY_서로_다른_질문의_연속취소는_최신_질문만(client, tenant_id, fake_llm, fake_embed, pass_gate):
    """X 취소 → Y 취소 → "다시"는 Y 재실행 — standalone이 달라 체인 되감기가 멈춰야 한다."""
    await register_faq(client)
    cid = await _seed_cancelled(tenant_id, '환불 기간 알려줘', '', standalone='환불 처리 기간')
    await _seed_cancelled(tenant_id, '사이즈 가이드해줘', 'S(1)',
                          standalone='티셔츠 사이즈별 착용 가이드', conversation_id=cid)
    fake_llm.intent_json = RETRY_JSON

    async with AsyncSessionLocal() as s:
        svc = RagService(tenant_id=tenant_id, session=s, user_id='agent-a')
        prepared = await svc.prepare('다시', conversation_id=cid)
    assert prepared.original_query == '사이즈 가이드해줘'               # 최신 질문(Y)
    assert prepared.standalone_query == '티셔츠 사이즈별 착용 가이드'
