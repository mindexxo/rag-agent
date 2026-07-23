"""FAQ 청크 텍스트 조립 단위 테스트 — faq_indexing.build_faq_chunk_text."""
from rag.faq_indexing import build_faq_chunk_text


def test_기본_형식():
    out = build_faq_chunk_text('환불 기간은?', ['돈 언제 돌려받아요'], '7일 이내 처리됩니다.')
    assert out == 'Q: 환불 기간은?\n(유사 질문: 돈 언제 돌려받아요)\nA: 7일 이내 처리됩니다.'


def test_유사질문_없으면_라인_생략():
    out = build_faq_chunk_text('환불 기간은?', [], '7일')
    assert out == 'Q: 환불 기간은?\nA: 7일'


def test_공백_원소_필터():
    out = build_faq_chunk_text('q', ['  ', '유효한 것', ''], 'a')
    assert '(유사 질문: 유효한 것)' in out


def test_여러_유사질문_쉼표_조인():
    out = build_faq_chunk_text('q', ['a1', 'a2'], 'a')
    assert '(유사 질문: a1, a2)' in out
