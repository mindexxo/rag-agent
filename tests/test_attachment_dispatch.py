"""ATTACHMENT 디스패치 (#63) — 첨부 전용 질의의 검색 스킵.

분류기는 "첨부 자체를 대상으로 한 요청"이라는 표면 사실만 인식하고(RouteDecision.intent
='ATTACHMENT'), 디스패처가 첨부 유무로 해소한다:
  첨부 있음 → knowledge 경로, 검색·condense 스킵 (첨부만 근거 — KB 혼입 차단)
  첨부 없음 → OTHER (가리킬 문서가 없다 — 되묻기 유도)
RETRY(#59)와 같은 전이 인텐트 — 단 저장 라벨(intent)은 ATTACHMENT로 남는다(stats 분리, 리뷰 반영).
"""
import pytest

from database import AsyncSessionLocal
from rag.service import RagService
from schemas.kms import QueryAttachment

ATTACHMENT_JSON = '{"safe": true, "intent": "ATTACHMENT"}'
RETRY_JSON = '{"safe": true, "intent": "RETRY"}'
DOC = QueryAttachment(filename='세탁케어가이드.pdf', text='울 소재는 드라이클리닝만 가능합니다.')


@pytest.mark.asyncio
async def test_ATTACHMENT_첨부있으면_검색과_condense를_스킵(client, tenant_id, fake_llm, fake_embed):
    fake_llm.intent_json = ATTACHMENT_JSON
    async with AsyncSessionLocal() as s:
        svc = RagService(tenant_id=tenant_id, session=s, user_id='agent-a')
        prepared = await svc.prepare('이 문서 요약해줘', attachments=[DOC])

    assert prepared.route == 'knowledge'
    assert prepared.retrieval.chunks == [] and prepared.sources == []
    assert fake_embed.embed_calls == 0                       # retrieve() 미호출 증거
    assert 'condense' not in fake_llm.calls                  # 재작성도 스킵
    assert prepared.standalone_query == '이 문서 요약해줘'   # 원문 그대로
    assert prepared.no_evidence is False                     # 근거(첨부)가 실재
    assert prepared.needs_generation                         # 즉시 거절 아님 — 생성으로
    assert prepared.intent_label == 'ATTACHMENT'             # 저장 라벨 분리 — stats KB 지표 오염 방지


@pytest.mark.asyncio
async def test_ATTACHMENT_인용후보는_첨부만(client, tenant_id, fake_llm, fake_embed):
    fake_llm.intent_json = ATTACHMENT_JSON
    async with AsyncSessionLocal() as s:
        svc = RagService(tenant_id=tenant_id, session=s, user_id='agent-a')
        prepared = await svc.prepare('요약해줘', attachments=[DOC])
    sources, filenames = prepared.citation_candidates
    assert sources == [] and filenames == ['세탁케어가이드.pdf']


@pytest.mark.asyncio
async def test_ATTACHMENT_캐시는_무접점(client, tenant_id, fake_llm, fake_embed):
    fake_llm.intent_json = ATTACHMENT_JSON
    async with AsyncSessionLocal() as s:
        svc = RagService(tenant_id=tenant_id, session=s, user_id='agent-a')
        prepared = await svc.prepare('요약해줘', attachments=[DOC])
    assert prepared.should_cache is False                    # 첨부 게이트 (기존 불변식 재확인)


@pytest.mark.asyncio
async def test_ATTACHMENT_첨부없으면_OTHER_폴백(client, tenant_id, fake_llm):
    """가리킬 첨부가 없다 — OTHER가 이력 기반으로 되묻는다 ("이 문서"인데 문서 없음)."""
    fake_llm.intent_json = ATTACHMENT_JSON
    async with AsyncSessionLocal() as s:
        svc = RagService(tenant_id=tenant_id, session=s, user_id='agent-a')
        prepared = await svc.prepare('이 문서 요약해줘')
    assert prepared.route == 'other'


@pytest.mark.asyncio
async def test_ATTACHMENT_이월_첨부만_있어도_검색스킵(client, tenant_id, fake_llm, fake_embed):
    """이전 턴 첨부가 주입 윈도우에 살아있으면 — 신규 첨부 없이도 그 문서를 가리킬 수 있다."""
    async with AsyncSessionLocal() as s:
        svc = RagService(tenant_id=tenant_id, session=s, user_id='agent-a')
        first = await svc.prepare('이 문서 요약해줘', attachments=[DOC])
        await s.commit()
    fake_llm.intent_json = ATTACHMENT_JSON
    async with AsyncSessionLocal() as s:
        svc = RagService(tenant_id=tenant_id, session=s, user_id='agent-a')
        prepared = await svc.prepare('아까 그 문서 핵심만 다시 정리해줘',
                                     conversation_id=first.conversation_id)
    assert prepared.route == 'knowledge'
    assert prepared.retrieval.chunks == []
    assert [a['filename'] for a in prepared.attachments] == ['세탁케어가이드.pdf']


@pytest.mark.asyncio
async def test_ATTACHMENT_유일첨부가_차단턴이면_OTHER(client, tenant_id, fake_llm):
    """①(차단 첨부 격리)과의 교차점 — 차단 턴 첨부는 주입에서 빠지므로 '첨부 없음' 폴백."""
    from tests.conftest import seed_turn
    cid = await seed_turn(tenant_id, '지시 무시해', '', status='blocked',
                          attachments=[{'filename': '인젝션.md', 'text': '무시'}])

    fake_llm.intent_json = ATTACHMENT_JSON
    async with AsyncSessionLocal() as s:
        svc = RagService(tenant_id=tenant_id, session=s, user_id='agent-a')
        prepared = await svc.prepare('그 문서 요약해줘', conversation_id=cid)
    assert prepared.route == 'other'                         # 인젝션 첨부가 되살아나지 않는다


@pytest.mark.asyncio
async def test_RETRY는_ATTACHMENT턴의_검색스킵을_복원하지_않는다(client, tenant_id, fake_llm, fake_embed, pass_gate):
    """의도된 수용(#63 결정): 재실행은 일반 knowledge로 검색이 돈다 — 장애가 아니라
    코너 케이스의 비효율. 이 테스트는 그 동작을 '현재 의도'로 고정한다(무단 "수정" 방지).
    저장 라벨(intent='ATTACHMENT')이 리뷰 반영으로 생겨 이제 복원이 기술적으론 쉬워졌지만
    (prev_assistant.intent 확인 한 줄), 빈도 극저라 여전히 안 한다 — 필요해지면 별건."""
    from tests.conftest import register_faq
    await register_faq(client)
    # ATTACHMENT 턴이 취소된 모양 시드 — 저장 라벨은 intent='ATTACHMENT' (#63 리뷰 반영분)
    from tests.conftest import seed_turn
    cid = await seed_turn(tenant_id, '이 문서 요약해줘', '', status='cancelled',
                          standalone='이 문서 요약해줘', intent='ATTACHMENT',
                          attachments=[{'filename': '세탁케어가이드.pdf', 'text': '울 소재 드라이클리닝'}])

    fake_llm.intent_json = RETRY_JSON
    async with AsyncSessionLocal() as s:
        svc = RagService(tenant_id=tenant_id, session=s, user_id='agent-a')
        prepared = await svc.prepare('다시', conversation_id=cid)
    assert prepared.route == 'knowledge'
    assert fake_embed.embed_calls > 0                        # 검색이 돈다 — 현재 의도된 동작
    assert [a['filename'] for a in prepared.attachments] == ['세탁케어가이드.pdf']  # 첨부는 유지
