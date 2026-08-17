"""질의 재작성 의미 확장(#5) 단위 테스트 — conversation.condense_to_queries + retriever._rrf_fuse.

LLM·DB 없이 검증한다. #43에서 출력이 JSON 스키마 강제로 바뀌어:
- 구 3줄 파싱·라벨 줄 방어("검색용 독립 질문:" 실측 2/3)는 필드 이름이 대체 — 테스트도 소멸
- 남은 계약: 스키마가 강제 못 하는 후처리(중복 제거·빈 값 폴백·상한 슬라이스)와
  폴백 3갈래(비JSON·빈 standalone·예외 → [원본])
"""
import json

import pytest

from rag.conversation import condense_to_queries
from rag.retriever import _rrf_fuse


class _FakeLlm:
    def __init__(self, result=None, error=None):
        self._result = result
        self._error = error

    async def acomplete(self, messages, extra_body=None):
        if self._error:
            raise self._error
        return self._result


def _multi(standalone: str, *variants: str) -> str:
    return json.dumps({'standalone': standalone, 'variants': list(variants)}, ensure_ascii=False)


# ===== condense_to_queries ==========================================

@pytest.mark.asyncio
async def test_정상이면_standalone과_변형_2개():
    llm = _FakeLlm(result=_multi('반품은 한 달 안에 아무 때나 가능한가요?',
                                 '단순변심 반품 신청 기한이 한 달 이내인지 확인 부탁드립니다.',
                                 '반품 접수 가능 기간이 한 달인지 알려주세요.'))
    out = await condense_to_queries(llm, '반품은 한 달 안에 아무때나 되는거죠?', [])
    assert out[0] == '반품은 한 달 안에 아무 때나 가능한가요?'
    assert len(out) == 3


@pytest.mark.asyncio
async def test_변형_초과는_2개까지만():
    # maxItems=2를 서버가 무시해도(스키마 미지원 폴백 경로) 방어 슬라이스가 지킨다
    llm = _FakeLlm(result=_multi('재작성', '변형1', '변형2', '변형3'))
    assert await condense_to_queries(llm, '원본', []) == ['재작성', '변형1', '변형2']


@pytest.mark.asyncio
async def test_standalone과_같은_변형은_제거():
    # 같은 줄이 반복되면 RRF에서 같은 순위 리스트를 중복 가산하게 되므로 걸러야 한다
    llm = _FakeLlm(result=_multi('재작성', '재작성', '변형1'))
    assert await condense_to_queries(llm, '원본', []) == ['재작성', '변형1']


@pytest.mark.asyncio
async def test_변형끼리_중복도_제거():
    # 변형1==변형2면 같은 순위 리스트가 RRF에 두 번 가산됨 (#3 A/B 희석 부작용 방어)
    llm = _FakeLlm(result=_multi('재작성', '변형1', '변형1', '변형2'))
    assert await condense_to_queries(llm, '원본', []) == ['재작성', '변형1', '변형2']


@pytest.mark.asyncio
async def test_변형_없이_standalone만_와도_동작():
    llm = _FakeLlm(result=_multi('재작성만'))
    assert await condense_to_queries(llm, '원본', []) == ['재작성만']


@pytest.mark.asyncio
async def test_빈_standalone이면_원본_폴백():
    # minLength가 전 서버에서 강제된다는 보장이 없다 — 후처리가 지키는 계약
    llm = _FakeLlm(result=_multi('  ', '변형1'))
    out = await condense_to_queries(llm, '원본', [])
    assert out[0] == '원본'
    assert '변형1' in out                     # standalone만 폴백, 변형은 살아있다


@pytest.mark.asyncio
async def test_비JSON_응답이면_원본_폴백():
    # 스키마 미지원 서버의 자유 생성이 형식을 안 지킨 경우 — 그럴듯한 복구 없이 폴백 (#43)
    llm = _FakeLlm(result='재작성\n변형1\n변형2')   # 구 3줄 형식 — 이제 유효하지 않다
    assert await condense_to_queries(llm, '원본', []) == ['원본']


@pytest.mark.asyncio
async def test_LLM_None이면_원본_폴백():
    assert await condense_to_queries(_FakeLlm(result=None), '원본', []) == ['원본']


@pytest.mark.asyncio
async def test_LLM_예외면_원본_폴백():
    # 폴백 = 재작성·변형 없이 원본 단독 검색 (기능 자동 off, condense_query 폴백 패턴과 동일)
    assert await condense_to_queries(_FakeLlm(error=RuntimeError('down')), '원본', []) == ['원본']


# ===== condense_query (단일) ========================================

@pytest.mark.asyncio
async def test_단일_condense도_JSON_실값을_반환():
    """condense_query의 JSON 전환 실값 고정 — FakeLlm 잠복 버그(라벨 줄 반향)가
    '이 값을 검증하는 테스트가 없어' 살아남았던 전례의 재발 방지 (리뷰 지적)."""
    from rag.conversation import condense_query
    from rag.models import Message
    llm = _FakeLlm(result=json.dumps({'standalone': '하자 교환 배송비는 누가 부담하나요?'},
                                     ensure_ascii=False))
    history = [Message(role='user', content='하자 교환 조건 알려줘'),
               Message(role='assistant', content='하자 교환은 …', status='done')]
    out = await condense_query(llm, '그럼 배송비는?', history)
    assert out == '하자 교환 배송비는 누가 부담하나요?'


@pytest.mark.asyncio
async def test_단일_condense_히스토리_없으면_LLM_미호출():
    class _Exploding:
        async def acomplete(self, *a, **kw):
            raise AssertionError('히스토리 게이트가 뚫렸다')
    from rag.conversation import condense_query
    assert await condense_query(_Exploding(), '원본', []) == '원본'


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
