"""판정기(#113 — 2차 검증) 경로 통합 계약: 임계 아래 [floor, threshold) 대역의 재사용 심사.

유사도는 명시 벡터로 제어한다(코사인이 손으로 계산되는 2-성분 직교 조합) — fake_embed의
텍스트 결정적 벡터로는 대역 안 유사도를 못 만든다. 판정기 자체(_verify_reuse)는
몽키패치로 고정한다 — LLM 판정 품질은 오프라인 검증(prompt_texts 주석의 40쌍 실측)이
담당하고, 여기서는 배선(어느 조건에서 불리고, 결과가 hit/miss로 이어지는가)만 고정한다.
"""
import math

import pytest

from database import AsyncSessionLocal
from rag import cache
from schemas.kms import SourceCitation

DIM = 1024


def _unit(i: int) -> list[float]:
    v = [0.0] * DIM
    v[i] = 1.0
    return v


def _mix(sim: float) -> list[float]:
    """_unit(0)과 코사인 유사도가 정확히 sim인 단위 벡터."""
    v = [0.0] * DIM
    v[0] = sim
    v[1] = math.sqrt(1 - sim * sim)
    return v


class _StubLlm:
    """판정기까지 도달했는지만 세는 자리표시 — _verify_reuse가 패치되므로 호출되지 않는다."""


async def _seed(session, tenant_id: str) -> None:
    src = [SourceCitation(document_id=5, filename='정책.pdf', version=1)]
    await cache.save_answer(session, tenant_id, '단순변심 반품 며칠까지 돼요?',
                            '14일입니다', src, [5], query_embedding=_unit(0))
    await session.commit()


def _patch_verdict(monkeypatch, verdict: bool, calls: list) -> None:
    async def fake_verify(llm, cached_query, new_query):
        calls.append((cached_query, new_query))
        return verdict
    monkeypatch.setattr(cache, '_verify_reuse', fake_verify)


@pytest.mark.asyncio
async def test_대역_후보는_판정_승인_시_히트(tenant_id, fake_embed, monkeypatch):
    calls = []
    _patch_verdict(monkeypatch, True, calls)
    async with AsyncSessionLocal() as session:
        await _seed(session, tenant_id)
        hit = await cache.get_semantic(session, tenant_id, '단순변심 반품 기간 알려주세요',
                                       [5], query_embedding=_mix(0.90), llm=_StubLlm())
    assert hit is not None and hit.answer == '14일입니다'
    assert len(calls) == 1, '대역 후보인데 판정기가 안 불렸다'


@pytest.mark.asyncio
async def test_대역_후보는_판정_거절_시_미스(tenant_id, fake_embed, monkeypatch):
    calls = []
    _patch_verdict(monkeypatch, False, calls)
    async with AsyncSessionLocal() as session:
        await _seed(session, tenant_id)
        hit = await cache.get_semantic(session, tenant_id, '하자 반품 기간 알려주세요',
                                       [5], query_embedding=_mix(0.90), llm=_StubLlm())
    assert hit is None
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_llm이_없으면_대역_후보는_기존처럼_미스(tenant_id, fake_embed, monkeypatch):
    calls = []
    _patch_verdict(monkeypatch, True, calls)
    async with AsyncSessionLocal() as session:
        await _seed(session, tenant_id)
        hit = await cache.get_semantic(session, tenant_id, '단순변심 반품 기간 알려주세요',
                                       [5], query_embedding=_mix(0.90))
    assert hit is None, 'llm=None인데 대역 후보가 히트했다 — 기존 계약 파괴'
    assert calls == []


@pytest.mark.asyncio
async def test_floor_미만은_판정_없이_미스(tenant_id, fake_embed, monkeypatch):
    calls = []
    _patch_verdict(monkeypatch, True, calls)
    async with AsyncSessionLocal() as session:
        await _seed(session, tenant_id)
        hit = await cache.get_semantic(session, tenant_id, '전혀 다른 질문',
                                       [5], query_embedding=_mix(0.60), llm=_StubLlm())
    assert hit is None
    assert calls == [], 'floor 미만인데 판정 콜이 나갔다 — 비용 누수'


@pytest.mark.asyncio
async def test_임계_이상은_판정_없이_즉시_히트(tenant_id, fake_embed, monkeypatch):
    calls = []
    _patch_verdict(monkeypatch, False, calls)   # 거절로 패치해도 안 불려야 한다
    async with AsyncSessionLocal() as session:
        await _seed(session, tenant_id)
        hit = await cache.get_semantic(session, tenant_id, '단순변심 반품 며칠까지 돼요?',
                                       [5], query_embedding=_unit(0), llm=_StubLlm())
    assert hit is not None, '임계 이상 히트가 판정기에 막혔다 — 히트 지연 계약 파괴'
    assert calls == [], '임계 이상인데 판정 콜이 나갔다'


@pytest.mark.asyncio
async def test_대역_후보도_기계_가드가_먼저다(tenant_id, fake_embed, monkeypatch):
    calls = []
    _patch_verdict(monkeypatch, True, calls)   # 판정기가 승인해도 가드가 먼저 잘라야 한다
    async with AsyncSessionLocal() as session:
        await _seed(session, tenant_id)
        hit = await cache.get_semantic(session, tenant_id, '단순변심 반품 7일까지 돼요?',
                                       [5], query_embedding=_mix(0.90), llm=_StubLlm())
    assert hit is None, '수치 지문이 다른데(7일 추가) 재사용됐다'
    assert calls == [], '가드가 막을 후보에 판정 콜이 나갔다 — 순서 위반'
