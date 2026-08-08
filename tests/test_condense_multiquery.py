"""질의 재작성 의미 확장(#5) 단위 테스트 — conversation.condense_to_queries + retriever._rrf_fuse.

LLM·DB 없이 검증한다:
- condense_to_queries: 3줄 파싱(첫 줄=standalone) / 폴백(None·예외·빈 결과 → [원본]) /
  standalone 중복 변형 제거 / 변형 2개 상한
- _rrf_fuse: N-way 융합 순위, 다중 리스트 출현 가점
"""
import pytest

from rag.conversation import condense_to_queries
from rag.retriever import _rrf_fuse


class _FakeLlm:
    def __init__(self, result=None, error=None):
        self._result = result
        self._error = error

    async def acomplete(self, messages):
        if self._error:
            raise self._error
        return self._result


# ===== condense_to_queries ==========================================

@pytest.mark.asyncio
async def test_정상_3줄이면_standalone과_변형_2개():
    llm = _FakeLlm(result='반품은 한 달 안에 아무 때나 가능한가요?\n'
                          '단순변심 반품 신청 기한이 한 달 이내인지 확인 부탁드립니다.\n'
                          '반품 접수 가능 기간이 한 달인지 알려주세요.')
    out = await condense_to_queries(llm, '반품은 한 달 안에 아무때나 되는거죠?', [])
    assert out[0] == '반품은 한 달 안에 아무 때나 가능한가요?'
    assert len(out) == 3


@pytest.mark.asyncio
async def test_규격_초과_출력은_변형_2개까지만():
    llm = _FakeLlm(result='재작성\n변형1\n변형2\n변형3')
    assert await condense_to_queries(llm, '원본', []) == ['재작성', '변형1', '변형2']


@pytest.mark.asyncio
async def test_standalone과_같은_변형은_제거():
    # 같은 줄이 반복되면 RRF에서 같은 순위 리스트를 중복 가산하게 되므로 걸러야 한다
    llm = _FakeLlm(result='재작성\n재작성\n변형1')
    assert await condense_to_queries(llm, '원본', []) == ['재작성', '변형1']


@pytest.mark.asyncio
async def test_머리말_라벨_줄은_걸러진다():
    # "검색용 독립 질문:" 라벨을 첫 줄로 붙이는 실측 사례(#5 mt003) — 라벨이 standalone이 되면
    # 검색·생성 질문이 라벨 문자열이 되어 거절로 이어진다. 콜론 종결 줄은 제거.
    llm = _FakeLlm(result='검색용 독립 질문:\n재작성\n변형1\n변형2')
    assert await condense_to_queries(llm, '원본', []) == ['재작성', '변형1', '변형2']


@pytest.mark.asyncio
async def test_라벨만_오면_원본_폴백():
    llm = _FakeLlm(result='검색용 독립 질문:\n변형 목록:')
    assert await condense_to_queries(llm, '원본', []) == ['원본']


@pytest.mark.asyncio
async def test_변형끼리_중복도_제거():
    # 변형1==변형2면 같은 순위 리스트가 RRF에 두 번 가산됨 (#3 A/B 희석 부작용 방어)
    llm = _FakeLlm(result='재작성\n변형1\n변형1\n변형2')
    assert await condense_to_queries(llm, '원본', []) == ['재작성', '변형1', '변형2']


@pytest.mark.asyncio
async def test_한_줄만_와도_동작():
    # 변형 없이 재작성만 온 경우 — 검색은 standalone 단독으로 (현행과 동일 동작)
    llm = _FakeLlm(result='재작성만')
    assert await condense_to_queries(llm, '원본', []) == ['재작성만']


@pytest.mark.asyncio
async def test_LLM_None이면_원본_폴백():
    assert await condense_to_queries(_FakeLlm(result=None), '원본', []) == ['원본']


@pytest.mark.asyncio
async def test_빈_출력이면_원본_폴백():
    assert await condense_to_queries(_FakeLlm(result='  \n \n'), '원본', []) == ['원본']


@pytest.mark.asyncio
async def test_LLM_예외면_원본_폴백():
    # 폴백 = 재작성·변형 없이 원본 단독 검색 (기능 자동 off, condense_query 폴백 패턴과 동일)
    assert await condense_to_queries(_FakeLlm(error=RuntimeError('down')), '원본', []) == ['원본']


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
