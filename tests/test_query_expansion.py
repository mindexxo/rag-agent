"""쿼리 확장(#3) 단위 테스트 — conversation.expand_query + retriever._rrf_fuse.

LLM·DB 없이 검증한다:
- expand_query: 정상 파싱 / 폴백(빈 결과·예외·None) / 원본 중복 제거 / 2개 상한
- _rrf_fuse: N-way 융합 순위, 다중 리스트 출현 가점
"""
import pytest

from rag.conversation import expand_query
from rag.retriever import _rrf_fuse


class _FakeLlm:
    def __init__(self, result=None, error=None):
        self._result = result
        self._error = error

    async def acomplete(self, messages):
        if self._error:
            raise self._error
        return self._result


# ===== expand_query =================================================

@pytest.mark.asyncio
async def test_정상_출력이면_변형_2개():
    llm = _FakeLlm(result='반품 절차는 어떻게 되나요?\n반품 신청 방법을 알려주세요.')
    assert await expand_query(llm, '이거 그냥 물러줘요.') == [
        '반품 절차는 어떻게 되나요?',
        '반품 신청 방법을 알려주세요.',
    ]


@pytest.mark.asyncio
async def test_규격_초과_출력은_2개까지만():
    llm = _FakeLlm(result='변형1\n변형2\n변형3\n변형4')
    assert await expand_query(llm, '원본') == ['변형1', '변형2']


@pytest.mark.asyncio
async def test_빈줄과_공백은_걸러진다():
    llm = _FakeLlm(result='\n  변형1  \n\n변형2\n')
    assert await expand_query(llm, '원본') == ['변형1', '변형2']


@pytest.mark.asyncio
async def test_원본과_같은_줄은_제거():
    # 원본을 그대로 되돌려주면 RRF에서 원본 브랜치가 중복 가산되므로 걸러야 한다
    llm = _FakeLlm(result='원본 질문\n변형1')
    assert await expand_query(llm, '원본 질문') == ['변형1']


@pytest.mark.asyncio
async def test_LLM_None이면_빈_리스트_폴백():
    assert await expand_query(_FakeLlm(result=None), '원본') == []


@pytest.mark.asyncio
async def test_LLM_예외면_빈_리스트_폴백():
    # 폴백 = 변형 없이 원본 단독 검색 (기능 자동 off, condense 폴백 패턴과 동일)
    assert await expand_query(_FakeLlm(error=RuntimeError('down')), '원본') == []


# ===== _rrf_fuse (N-way) ============================================

def test_단일_리스트는_순서_보존():
    scores = _rrf_fuse([[10, 20, 30]])
    assert sorted(scores, key=lambda c: -scores[c]) == [10, 20, 30]


def test_여러_리스트_모두_상위면_가점():
    # 20은 두 리스트 모두 출현 → 각자 1위인 10·30보다 총점 높아야 한다
    scores = _rrf_fuse([[10, 20], [30, 20]])
    top = sorted(scores, key=lambda c: -scores[c])
    assert top[0] == 20
    assert scores[10] == scores[30]          # 같은 rank 1끼리는 동점


def test_빈_리스트는_빈_결과():
    assert _rrf_fuse([]) == {}
    assert _rrf_fuse([[], []]) == {}


def test_점수_공식_고정():
    # score = 1/(k+rank), k=60 — 문서화된 공식 그대로인지 고정
    scores = _rrf_fuse([[1], [1]])
    assert scores[1] == pytest.approx(2 / 61)
