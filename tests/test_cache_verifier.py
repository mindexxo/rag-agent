"""캐시 재사용 판정(#113) 배선 계약 — **판정 승인 없이는 서빙이 없다**.

hit 조건(순서대로): 유사도 ≥ floor(후보 게이트) → doc집합 동일 → LLM 판정 승인.
자동 서빙 임계와 규칙 기반 기계 가드는 제거됐다(판정기로 일원화 — 사유는 rag/cache.py
모듈 docstring). 유사도는 명시 벡터로 제어하고(코사인이 손으로 계산되는 2-성분 조합),
판정기(_verify_reuse)는 몽키패치로 고정한다 — LLM 판정 품질은 오프라인 검증
(prompt_texts 주석의 40쌍 실측)이 담당하고, 여기서는 배선만 고정한다.
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
    """판정기 도달 여부만 보는 자리표시 — _verify_reuse가 패치되므로 호출되지 않는다."""


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
async def test_판정_승인이면_히트(tenant_id, fake_embed, monkeypatch):
    calls = []
    _patch_verdict(monkeypatch, True, calls)
    async with AsyncSessionLocal() as session:
        await _seed(session, tenant_id)
        hit = await cache.get_semantic(session, tenant_id, '단순변심 반품 기간 알려주세요',
                                       [5], query_embedding=_mix(0.90), llm=_StubLlm())
    assert hit is not None and hit.answer == '14일입니다'
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_판정_거절이면_미스(tenant_id, fake_embed, monkeypatch):
    calls = []
    _patch_verdict(monkeypatch, False, calls)
    async with AsyncSessionLocal() as session:
        await _seed(session, tenant_id)
        hit = await cache.get_semantic(session, tenant_id, '하자 반품 기간 알려주세요',
                                       [5], query_embedding=_mix(0.90), llm=_StubLlm())
    assert hit is None
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_고유사도도_판정을_거친다(tenant_id, fake_embed, monkeypatch):
    """자동 서빙 임계 제거의 핵심 계약 — 유사도 1.0이어도 판정 없이는 서빙 없음.

    근거: 유사도 0.96~0.99에서 답이 반대인 쌍 4건 실측(#113). 판정을 우회하는
    고유사도 지름길을 되살리면 그 4건이 그대로 오답 재생으로 돌아온다.
    """
    calls = []
    _patch_verdict(monkeypatch, True, calls)
    async with AsyncSessionLocal() as session:
        await _seed(session, tenant_id)
        hit = await cache.get_semantic(session, tenant_id, '단순변심 반품 며칠까지 돼요?',
                                       [5], query_embedding=_unit(0), llm=_StubLlm())
    assert hit is not None
    assert len(calls) == 1, '유사도 1.0인데 판정을 건너뛰었다 — 자동 서빙 임계가 되살아남'


@pytest.mark.asyncio
async def test_llm이_없으면_항상_미스(tenant_id, fake_embed, monkeypatch):
    calls = []
    _patch_verdict(monkeypatch, True, calls)
    async with AsyncSessionLocal() as session:
        await _seed(session, tenant_id)
        hit = await cache.get_semantic(session, tenant_id, '단순변심 반품 며칠까지 돼요?',
                                       [5], query_embedding=_unit(0))
    assert hit is None, 'llm=None인데 서빙됐다 — 판정 없는 히트 경로가 생김'
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
async def test_doc집합이_다르면_판정_없이_미스(tenant_id, fake_embed, monkeypatch):
    calls = []
    _patch_verdict(monkeypatch, True, calls)
    async with AsyncSessionLocal() as session:
        await _seed(session, tenant_id)
        hit = await cache.get_semantic(session, tenant_id, '단순변심 반품 며칠까지 돼요?',
                                       [5, 7], query_embedding=_unit(0), llm=_StubLlm())
    assert hit is None
    assert calls == [], 'doc집합 불일치인데 판정 콜이 나갔다 — 순서 위반(비용 누수)'
