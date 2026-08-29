"""C-1: eval 채점기 단위 테스트 — 수정된 채점 동작 고정.

채점기가 틀리면 측정 전체가 무의미 — corpus v2 기준선 측정의 전제.
임베딩 폴백은 가짜(_sync 패치)로 통제한다.
"""
import pytest

import eval.generation as gen
from eval.generation import _citation_match, citation_accuracy, expected_points_coverage
from rag.citation_labels import TAIL_END, TAIL_START, citation_tail
from rag.retriever import RetrievedChunk


def _tail(*nums: int) -> str:
    """출처 꼬리 픽스처 — 운영이 내는 형태 그대로.

    형식을 손으로 조립하지 않고 citation_tail을 거친다 (#65) — 픽스처가 옛 형식으로
    낡으면 채점기 테스트가 "현실적인 입력"을 안 보게 된다.
    """
    return citation_tail(nums)


def _chunks(*filenames: str) -> list[RetrievedChunk]:
    """문서당 청크 1개 — 등장 순서가 곧 인용 번호(sources_from_chunks 불변식).
    'FAQ'는 운영과 같게 faq_id 청크로 만든다(후보 목록 마지막에 접힘)."""
    out = []
    for i, fn in enumerate(filenames, start=1):
        faq = fn == 'FAQ'
        out.append(RetrievedChunk(chunk_id=i, document_id=None if faq else i, text='본문',
                                  heading_path=[], page=None, filename=fn, version=1,
                                  faq_id=1 if faq else None))
    return out


class TestCitationMatch:
    def test_정확_일치와_FAQ(self):
        assert _citation_match('FAQ', 'FAQ')
        assert _citation_match('배송지연대응', '배송지연대응')

    def test_프리픽스형_파일명(self):
        # gold 기대명과 DB 파일명의 prefix 차이 흡수 (kms_03_ 등)
        assert _citation_match('배송지연대응', 'kms_03_배송지연대응')

    def test_짧은_토큰_우연_매칭_거부(self):
        assert not _citation_match('배송', 'kms_03_배송지연대응')     # 2자 포함 — 오탐 방지
        assert not _citation_match('첨부: 영수증', '배송지연대응')


class TestCitationAccuracy:
    """v3 (#56): 인용은 답변 끝 출처 꼬리(번호 목록)에서만 — 운영과 같은 배관으로 해석한다."""

    def test_전_기대문서_인용시_만점(self):
        score = citation_accuracy('답변 ' + _tail(1, 2), ['환불정책.pdf', '배송정책.docx'],
                                  _chunks('환불정책.pdf', '배송정책.docx'))
        assert score == 1.0

    def test_multi_doc_부분_인용은_비율(self):
        # v2 개정의 핵심 — any-match였으면 1.0 과대평가
        score = citation_accuracy('답변 ' + _tail(1), ['환불정책.pdf', '배송정책.docx'],
                                  _chunks('환불정책.pdf', '배송정책.docx'))
        assert score == 0.5

    def test_FAQ_인용_미탐_해소(self):
        assert citation_accuracy('답변입니다 ' + _tail(1), ['FAQ'], _chunks('FAQ')) == 1.0

    def test_오인용만_있으면_0(self):
        # 1번(엉뚱문서)만 인용 — 기대(환불정책)는 2번인데 안 짚었다
        assert citation_accuracy('답변 ' + _tail(1), ['환불정책.pdf'],
                                 _chunks('엉뚱문서.pdf', '환불정책.pdf')) == 0.0

    def test_꼬리_없으면_0(self):
        # 형식 미준수(꼬리 누락)는 인용 0으로 집계 — 그게 사실이다 (#56)
        assert citation_accuracy('인용 없는 답변', ['환불정책.pdf'], _chunks('환불정책.pdf')) == 0.0

    def test_본문_숫자는_스캔하지_않는다(self):
        # v3의 핵심 — 본문의 숫자·[1]류가 인용으로 오인되지 않고, 꼬리 밖은 무시된다
        assert citation_accuracy('[1] 본문 표기는 무효, 30일 같은 수치도 무효 ' + _tail(),
                                 ['환불정책.pdf'], _chunks('환불정책.pdf')) == 0.0

    def test_corpus_v2_확장자_스트립(self):
        assert citation_accuracy(_tail(1), ['공지.txt'], _chunks('공지.txt')) == 1.0
        assert citation_accuracy(_tail(1), ['멤버십혜택표.xlsx'], _chunks('멤버십혜택표.xlsx')) == 1.0

    def test_대체_출처_그룹은_하나만_인용해도_인정(self):
        # 같은 정보가 md 정책 문서와 xlsx 표에 중복 — 어느 쪽을 인용해도 정당 (multi_doc 오탐 보정)
        exp = [['멤버십포인트.pdf', '멤버십혜택표.xlsx'], '제휴카드결제안내.docx']
        chunks = _chunks('멤버십포인트.pdf', '멤버십혜택표.xlsx', '제휴카드결제안내.docx')
        assert citation_accuracy(_tail(2, 3), exp, chunks) == 1.0
        assert citation_accuracy(_tail(1, 3), exp, chunks) == 1.0
        assert citation_accuracy(_tail(3), exp, chunks) == 0.5   # 그룹 미커버는 여전히 감점


@pytest.fixture
def fake_embed(monkeypatch):
    """포인트 텍스트가 답변 조각에 그대로 들어 있으면 유사도 1, 아니면 0.

    임베딩 서버 없이 **판정 구조**(어떤 조각과 대조하는가)만 검증하려는 픽스처다.
    의미 유사도를 흉내내지는 않는다 — 그건 실측(골드 450문항)이 맡는 몫이다.
    """
    from rag.embeddings import Embedding
    # 벡터 자리에 원문을 실어 보내고, 코사인 자리에서 포함 여부로 답한다.
    monkeypatch.setattr(gen, 'embed_texts_sync', lambda texts: [Embedding(dense=[t]) for t in texts])
    monkeypatch.setattr(gen, '_cosine', lambda p, s: 1.0 if p[0] in s[0] else 0.0)


class TestSplitSegments:
    def test_문장과_절을_모두_후보로(self):
        segs = gen._split_segments('A는 1이고, B는 2입니다.')
        assert 'A는 1이고, B는 2입니다' in segs      # 문장 통째
        assert 'B는 2입니다' in segs                 # 절

    def test_숫자_사이_마침표는_문장_경계가_아니다(self):
        # '78.4%'가 '78'/'4%'로 쪼개지면 그 사실을 담은 포인트는 영영 못 맞는다
        assert gen._split_sentences('집행률은 78.4%입니다.') == ['집행률은 78.4%입니다']

    def test_숫자_사이_쉼표는_절_경계가_아니다(self):
        assert '3,000원을 지급합니다' in ' '.join(gen._split_segments('보상으로 3,000원을 지급합니다.'))


class TestExpectedPointsCoverage:
    def test_절에_담긴_포인트를_잡는다(self, fake_embed):
        # 문장 단위로만 보면 못 잡던 것 — 비교 단위를 절까지 내린 이유
        assert expected_points_coverage('A는 1이고, B는 2입니다.', ['B는 2입니다']) == 1.0

    def test_없는_포인트는_감점(self, fake_embed):
        assert expected_points_coverage('A는 1입니다.', ['B는 2입니다']) == 0.0

    def test_중복_포인트_각자_집계(self, fake_embed):
        assert expected_points_coverage('서술 포인트입니다.', ['서술 포인트', '서술 포인트']) == 1.0

    def test_포인트_없으면_None(self):
        assert expected_points_coverage('답변', []) is None

    def test_빈_답변은_0(self, fake_embed):
        assert expected_points_coverage('', ['어떤 포인트']) == 0.0
