"""FAQ 입력 상한 스키마 (#276) — DB 없이 Pydantic만.

길이는 **공백 제거 후** 재고, 빈 유사 질문은 개수에서 뺀다 — 라우터는 스키마가 strip한 값을 그대로 저장하므로
검증 기준과 저장 기준이 같다(schemas/kms.py FAQ_*_MAX).
"""
import pytest
from pydantic import ValidationError

from schemas.kms import (FAQ_ANSWER_MAX, FAQ_QUESTION_MAX, FAQ_VARIANT_MAX, FAQ_VARIANTS_MAX,
                         FaqCreateRequest, FaqUpdateRequest)


def _loc(exc_info):
    return exc_info.value.errors()[0]['loc']


def test_상한은_FE에_회신한_값이다():
    assert (FAQ_QUESTION_MAX, FAQ_VARIANT_MAX, FAQ_VARIANTS_MAX, FAQ_ANSWER_MAX) == (100, 100, 10, 500)


@pytest.mark.parametrize('field, over', [
    ('question', '가' * (FAQ_QUESTION_MAX + 1)),
    ('answer', '가' * (FAQ_ANSWER_MAX + 1)),
])
def test_질문_답변_상한_초과는_422_위치에_필드가_보인다(field, over):
    base = {'question': 'q', 'answer': 'a'}
    with pytest.raises(ValidationError) as e:
        FaqCreateRequest(**{**base, field: over})
    assert _loc(e) == (field,)
    with pytest.raises(ValidationError) as e:
        FaqUpdateRequest(**{field: over})
    assert _loc(e) == (field,)


def test_유사_질문_개수와_항목_길이():
    with pytest.raises(ValidationError) as e:
        FaqCreateRequest(question='q', answer='a', variants=['v'] * (FAQ_VARIANTS_MAX + 1))
    assert _loc(e) == ('variants',)
    with pytest.raises(ValidationError) as e:
        FaqCreateRequest(question='q', answer='a', variants=['ok', '가' * (FAQ_VARIANT_MAX + 1)])
    assert _loc(e) == ('variants', 1)      # 몇 번째 항목인지 보인다
    assert FaqCreateRequest(question='q', answer='a', variants=['v'] * FAQ_VARIANTS_MAX).variants == ['v'] * FAQ_VARIANTS_MAX


def test_길이는_공백_제거_후_잰다():
    padded = ' ' * 5 + '가' * FAQ_QUESTION_MAX + ' ' * 5     # 원문은 상한 초과, strip 후 딱 상한
    assert FaqCreateRequest(question=padded, answer=' a ').question == '가' * FAQ_QUESTION_MAX
    assert FaqUpdateRequest(answer=' 7일 ').answer == '7일'


def test_빈_유사_질문은_버리고_개수에서_뺀다():
    r = FaqCreateRequest(question='q', answer='a', variants=['', '   ', ' 돈 언제 돌려받아요 '] + ['v'] * (FAQ_VARIANTS_MAX - 1))
    assert r.variants == ['돈 언제 돌려받아요'] + ['v'] * (FAQ_VARIANTS_MAX - 1)


def test_PATCH는_안_보낸_필드가_None으로_남는다():
    u = FaqUpdateRequest(is_active=False)
    assert u.question is None and u.variants is None and u.answer is None
