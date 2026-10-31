"""#234 측정 도구(eval/contextual_header) — 조건별 텍스트 조립과 운영 검색 바꿔 끼우기가 맞는지. 측정 결과가 이 조립에 달려 있어
(조립이 틀린 채 측정한 사고 #128), 수치보다 먼저 고정한다. DB·OpenSearch·LLM 불필요."""

from types import SimpleNamespace

from config import settings
from eval.contextual_header import CONDITIONS, condition, index_name, is_context_row, rule_lines, variant_text


def _doc(text, chunk_id=1, hp=('매뉴얼', '1. 절'), fn='a_01_매뉴얼.pdf'):
    return {'chunk_id': chunk_id, 'text': text, 'filename': fn, 'heading_path': list(hp)}


def test_조건별_텍스트():
    d = _doc('본문 한 줄.')
    cr, rule = {'1': '이 조각은 2분기 보고의 일부다.'}, {1: '[상황 6] 리셀 A/S'}
    assert variant_text('none', d, cr, rule) == '본문 한 줄.'
    assert variant_text('header', d, cr, rule) == 'a_01_매뉴얼 > 매뉴얼 > 1. 절\n본문 한 줄.'
    assert (
        variant_text('cr', d, cr, rule) == '이 조각은 2분기 보고의 일부다.\na_01_매뉴얼 > 매뉴얼 > 1. 절\n본문 한 줄.'
    )
    assert variant_text('rule', d, cr, rule) == 'a_01_매뉴얼 > 매뉴얼 > 1. 절\n[상황 6] 리셀 A/S\n본문 한 줄.'


def test_맥락이_없으면_현행_머리말과_같다():
    d = _doc('본문.')
    assert variant_text('cr', d, {}, {}) == variant_text('rule', d, {}, {}) == variant_text('header', d, {}, {})


def test_규칙_맥락_줄은_목록_소제목과_가리키는_대상():
    chunks = [
        _doc('[상황 6] 리셀 A/S 요청\n1. 구매처를 확인한다.\n2. 정품 여부를 묻는다.', 10, hp=()),
        _doc('3. 감정은 제공하지 않는다.', 11, hp=()),
        _doc('개점은 1시간까지 늦출 수 있다.', 12, hp=()),
        _doc('이 경우 폐점도 늦춘다.', 13, hp=()),
    ]
    assert rule_lines(chunks) == {11: '[상황 6] 리셀 A/S 요청', 13: '개점은 1시간까지 늦출 수 있다.'}


def test_맥락_셋_판별():
    assert is_context_row('aromanica_cx001') and is_context_row('goodpeople_cb003') and is_context_row('inticube_cx002')
    assert not is_context_row('harim_sf001')


def test_조건_안에서만_임시_인덱스와_변형_리랭커_입력():
    import rag.search.reranker as rr

    before_index, before_fn = settings.opensearch_index, rr._rerank_text
    chunk = SimpleNamespace(
        chunk_id=7,
        folder_name='상담',
        folder_description=None,
        faq_id=None,
        text='본문',
        filename='a.pdf',
        heading_path=[],
    )
    other = SimpleNamespace(
        chunk_id=8,
        folder_name=None,
        folder_description=None,
        faq_id=None,
        text='다른 본문',
        filename='b.pdf',
        heading_path=[],
    )
    with condition('cr', {7: '맥락\nb > h\n본문'}):
        assert settings.opensearch_index == index_name('cr') != before_index
        assert rr._rerank_text(chunk) == '상담 > 맥락\nb > h\n본문'  # 폴더는 운영처럼 리랭커에만, 맨 앞에
        assert rr._rerank_text(other) == before_fn(other)  # 바꿀 대상이 아닌 조각은 운영 그대로
    with condition('none', {7: '본문'}):
        assert rr._rerank_text(chunk) == '상담\n본문'  # 머리말이 없으면 폴더만 한 줄
    assert settings.opensearch_index == before_index and rr._rerank_text is before_fn  # 끝나면 원상 복구


def test_변형_텍스트가_없는_측정_조각은_조용히_운영_머리말로_돌아가지_않는다():
    # 스냅숏과 어긋나면 cr·rule의 리랭커 입력이 몰래 header가 된다(#234 리뷰 지적 1) — 터뜨린다
    import pytest

    import rag.search.reranker as rr

    doc_id = 5
    stray = SimpleNamespace(
        chunk_id=(doc_id << 20) | 3,
        document_id=doc_id,
        faq_id=None,
        folder_name=None,
        folder_description=None,
        text='본문',
        filename='a.pdf',
        heading_path=[],
    )
    with condition('cr', {(doc_id << 20) | 1: '맥락\n본문'}), pytest.raises(RuntimeError, match='스냅숏'):
        rr._rerank_text(stray)


def test_조건은_넷이고_임시_인덱스는_운영_이름과_다르다():
    assert CONDITIONS == ('none', 'header', 'cr', 'rule')
    assert len({index_name(c) for c in CONDITIONS} | {settings.opensearch_index}) == 5
