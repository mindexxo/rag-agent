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
    # 원문 인자(#153)까지 받는다 — get_semantic이 넷을 넘긴다. calls는 재작성문 쌍만 기록해
    # 기존 단언을 그대로 두고, 원문은 orig_calls에 따로 쌓는다.
    async def fake_verify(llm, cached_query, new_query, cached_original=None, new_original=None):
        calls.append((cached_query, new_query))
        orig_calls.append((cached_original, new_original))
        return verdict
    orig_calls: list = []
    fake_verify.orig_calls = orig_calls
    monkeypatch.setattr(cache, '_verify_reuse', fake_verify)
    return orig_calls


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


# ── 원문 쿼리 배선 (#153) ────────────────────────────────────────────────────
# 판정 품질이 아니라 **원문이 저장되고 판정기까지 도달하는가**만 고정한다.
# 판정 품질은 eval/cache_judge_probe.py(40쌍 + 방어 케이스)가 담당한다.

@pytest.mark.asyncio
async def test_원문이_저장되고_판정기까지_간다(tenant_id, fake_embed, monkeypatch):
    calls = []
    origs = _patch_verdict(monkeypatch, True, calls)
    src = [SourceCitation(document_id=5, filename='정책.pdf', version=1)]
    async with AsyncSessionLocal() as session:
        await cache.save_answer(session, tenant_id, '인사 규정은 어떻게 되나요?', '...', src, [5],
                                query_embedding=_unit(0), original_query='인사 규정')
        await session.commit()
        hit = await cache.get_semantic(session, tenant_id, '인사 규정의 주요 내용은?',
                                       [5], query_embedding=_mix(0.90), llm=_StubLlm(),
                                       original_query='인사 규정')
    assert hit is not None
    assert origs == [('인사 규정', '인사 규정')], \
        '원문이 판정기에 도달하지 않았다 — 저장(save_answer) 또는 전달(get_semantic) 배선 끊김'


@pytest.mark.asyncio
async def test_도입_전_행은_원문이_None으로_간다(tenant_id, fake_embed, monkeypatch):
    """원문 컬럼은 NULL 허용이다. 도입 전 행이 판정기를 깨뜨리지 않아야 한다 (#153)."""
    calls = []
    origs = _patch_verdict(monkeypatch, True, calls)
    async with AsyncSessionLocal() as session:
        await _seed(session, tenant_id)          # original_query 없이 저장 = 도입 전 행
        hit = await cache.get_semantic(session, tenant_id, '단순변심 반품 기간 알려주세요',
                                       [5], query_embedding=_mix(0.90), llm=_StubLlm(),
                                       original_query='반품 기간')
    assert hit is not None
    assert origs == [(None, '반품 기간')], '캐시 쪽 원문이 NULL인 경로가 깨졌다'


def test_원문이_없으면_템플릿에_없음이_박힌다():
    """프롬프트 계약 — 빈 문자열을 넣으면 판정기가 그것도 신호로 읽는다 (#153)."""
    from rag.prompt_texts import CACHE_REUSE_JUDGE_NO_ORIGINAL
    from rag.prompts import build_cache_reuse_judge_messages
    user = build_cache_reuse_judge_messages('재작성A', '재작성B')[1]['content']
    assert user.count(CACHE_REUSE_JUDGE_NO_ORIGINAL) == 2
    assert '__CACHED_ORIGINAL__' not in user and '__NEW_ORIGINAL__' not in user

    filled = build_cache_reuse_judge_messages('재작성A', '재작성B', '원문A', '원문B')[1]['content']
    assert '원문A' in filled and '원문B' in filled
    assert CACHE_REUSE_JUDGE_NO_ORIGINAL not in filled


@pytest.mark.asyncio
async def test_원문이_달라도_같은_행을_갱신한다(tenant_id, fake_embed):
    """원문을 cache_key에 섞으면 안 된다 (#153) — 키 기준은 재작성문 그대로여야 한다.

    같은 재작성문을 원문만 바꿔 두 번 저장하면 행이 둘로 갈리지 않고 하나가 갱신돼야 한다.
    (갈리면 upsert 충돌 계약이 깨진 것이고, 같은 질문의 캐시가 원문 표현마다 쌓인다.)
    """
    from sqlalchemy import select
    from rag.models import AnswerCache as Row
    src = [SourceCitation(document_id=5, filename='정책.pdf', version=1)]
    async with AsyncSessionLocal() as session:
        for orig in ('인사 규정', '인사규정 알려줘'):
            await cache.save_answer(session, tenant_id, '인사 규정은 어떻게 되나요?', '...', src, [5],
                                    query_embedding=_unit(0), original_query=orig)
        await session.commit()
        rows = (await session.execute(
            select(Row).where(Row.tenant_id == tenant_id))).scalars().all()
    assert len(rows) == 1, f'원문이 cache_key에 섞였다 — 행이 {len(rows)}개로 갈림'
    assert rows[0].original_query == '인사규정 알려줘', '갱신 시 원문이 최신 턴 것으로 안 바뀜'
