"""멀티쿼리 통합(#5) 플래그 on 통합 테스트 — service.prepare → retriever multi 분기 관통.

리뷰 지적(테스트 커버리지 갭): 플래그를 켠 상태로 RagService 전 구간을 태우는 테스트가
없으면 on 전환 시점에 union+max-pool 경로가 미검증인 채 운영에 들어간다.
fake_embed 픽스처가 rerank_enabled=False로 두므로 여기서 multi 분기는 RRF 폴백 순서를
탄다 — union·게이트·standalone 흐름 검증이 목적 (max-pool 점수 자체는 eval이 담당).
"""
import pytest

from config import settings
from tests.conftest import register_faq, sse_answer, sse_meta



@pytest.fixture
def multi_query_on(monkeypatch):
    """기본값(True)과 무관하게 on을 명시 — 기본값이 바뀌어도 테스트 의도 유지."""
    monkeypatch.setattr(settings, 'condense_multi_query_enabled', True)


@pytest.fixture
def multi_query_off(monkeypatch):
    """원복 경로(off) 검증용 — 기본 on이므로 명시적으로 끈다."""
    monkeypatch.setattr(settings, 'condense_multi_query_enabled', False)




@pytest.mark.asyncio
async def test_멀티턴이면_condense_multi가_호출되고_응답_정상(
        client, tenant_id, fake_llm, pass_gate, multi_query_on):
    await register_faq(client)
    # 1턴 (단일턴 — 게이트에 의해 condense_multi 미호출이어야 함)
    res1 = await client.post('/kms/query', json={'query': '환불 기간 알려줘'})
    assert res1.status_code == 200
    assert 'condense_multi' not in fake_llm.calls          # 단일턴은 main 경로 그대로

    # 2턴 (멀티턴 — condense_multi 1콜로 재작성+변형, 검색은 union 경로)
    conv_id = sse_meta(res1)['conversation_id']
    res2 = await client.post('/kms/query',
                             json={'query': '그럼 교환은?', 'conversation_id': conv_id})
    assert res2.status_code == 200
    assert sse_meta(res2)['reason'] == 'ok'
    assert 'condense_multi' in fake_llm.calls              # 멀티턴에서만 호출
    assert fake_llm.calls.count('condense') == 0           # 기존 condense로 새지 않음
    assert '테스트 답변입니다.' in sse_answer(res2)


@pytest.mark.asyncio
async def test_플래그_off면_멀티턴도_기존_condense(
        client, tenant_id, fake_llm, pass_gate, multi_query_off):
    await register_faq(client)
    res1 = await client.post('/kms/query', json={'query': '환불 기간 알려줘'})
    conv_id = sse_meta(res1)['conversation_id']
    res2 = await client.post('/kms/query',
                             json={'query': '그럼 교환은?', 'conversation_id': conv_id})
    assert res2.status_code == 200
    assert 'condense_multi' not in fake_llm.calls
    assert 'condense' in fake_llm.calls                    # off = 현행 경로
