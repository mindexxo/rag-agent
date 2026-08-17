"""C-1: eval 채점기 단위 테스트 — 수정된 채점 동작 고정.

채점기가 틀리면 측정 전체가 무의미 — corpus v2 기준선 측정의 전제.
임베딩 폴백은 가짜(_sync 패치)로 통제한다.
"""
import pytest

import eval.generation as gen
from eval.generation import _citation_match, _contains_point, citation_accuracy, expected_points_coverage
from rag.citation_labels import TAIL_END, TAIL_START


def _tail(*labels: str) -> str:
    """출처 꼬리(#56) 픽스처 — 답변 끝에 붙는 형태 그대로."""
    return TAIL_START + ''.join(f'[{l}]' for l in labels) + TAIL_END


class TestCitationMatch:
    def test_정확_일치와_FAQ(self):
        assert _citation_match('FAQ', 'FAQ')
        assert _citation_match('배송지연대응', '배송지연대응')

    def test_프리픽스형_파일명(self):
        # 모델이 [배송지연대응 v1]처럼 prefix 없이 인용하는 관행
        assert _citation_match('배송지연대응', 'kms_03_배송지연대응')

    def test_짧은_토큰_우연_매칭_거부(self):
        assert not _citation_match('배송', 'kms_03_배송지연대응')     # 2자 포함 — 오탐 방지
        assert not _citation_match('첨부: 영수증', '배송지연대응')


class TestCitationAccuracy:
    """v3 (#56): 인용은 답변 끝 출처 꼬리에서만 스캔한다."""

    def test_전_기대문서_인용시_만점(self):
        score = citation_accuracy('답변 ' + _tail('환불정책.pdf v1', '배송정책 v2'),
                                  ['환불정책.pdf', '배송정책.docx'])
        assert score == 1.0

    def test_multi_doc_부분_인용은_비율(self):
        # v2 개정의 핵심 — any-match였으면 1.0 과대평가
        score = citation_accuracy('답변 ' + _tail('환불정책.pdf v1'), ['환불정책.pdf', '배송정책.docx'])
        assert score == 0.5

    def test_FAQ_인용_미탐_해소(self):
        assert citation_accuracy('답변입니다 ' + _tail('FAQ'), ['FAQ']) == 1.0

    def test_오인용만_있으면_0(self):
        assert citation_accuracy('답변 ' + _tail('엉뚱문서.pdf v1'), ['환불정책.pdf']) == 0.0

    def test_꼬리_없으면_0(self):
        # 형식 미준수(꼬리 누락)는 인용 0으로 집계 — 그게 사실이다 (#56)
        assert citation_accuracy('인용 없는 답변', ['환불정책.pdf']) == 0.0

    def test_본문_대괄호는_스캔하지_않는다(self):
        # v3의 핵심 — 본문 [참고]류가 인용으로 오인되지 않고, 꼬리 밖 라벨도 무시된다
        assert citation_accuracy('[환불정책.pdf v1] 본문 라벨은 무효 ' + _tail(), ['환불정책.pdf']) == 0.0

    def test_corpus_v2_확장자_스트립(self):
        assert citation_accuracy(_tail('공지 v1'), ['공지.txt']) == 1.0
        assert citation_accuracy(_tail('멤버십혜택표 v1'), ['멤버십혜택표.xlsx']) == 1.0

    def test_대체_출처_그룹은_하나만_인용해도_인정(self):
        # 같은 정보가 md 정책 문서와 xlsx 표에 중복 — 어느 쪽을 인용해도 정당 (multi_doc 오탐 보정)
        exp = [['멤버십포인트.pdf', '멤버십혜택표.xlsx'], '제휴카드결제안내.docx']
        assert citation_accuracy(_tail('멤버십혜택표 v1', '제휴카드결제안내 v1'), exp) == 1.0
        assert citation_accuracy(_tail('멤버십포인트 v1', '제휴카드결제안내 v1'), exp) == 1.0
        assert citation_accuracy(_tail('제휴카드결제안내 v1'), exp) == 0.5   # 그룹 미커버는 여전히 감점


@pytest.fixture
def no_embed_fallback(monkeypatch):
    """임베딩 폴백을 무력화 — 부분문자열 단계만 검증할 때."""
    from rag.embeddings import Embedding
    monkeypatch.setattr(gen, 'embed_texts_sync',
                        lambda texts: [Embedding(dense=[0.0] * 8) for _ in texts])


class TestExpectedPointsCoverage:
    def test_숫자_경계_오탐_방지(self, no_embed_fallback):
        # '30분'이 '130분'에 매칭되면 안 됨 (수정 전 오탐)
        assert expected_points_coverage('처리에 130분 걸립니다', ['30분']) == 0.0

    def test_정상_수치_매칭(self, no_embed_fallback):
        assert expected_points_coverage('30분 이내 처리됩니다', ['30분']) == 1.0
        assert expected_points_coverage('기간은 12개월입니다', ['12개월']) == 1.0

    def test_공백_변형_허용(self, no_embed_fallback):
        assert expected_points_coverage('3 0분... 아니 30 분 이내', ['30분']) == 1.0

    def test_콤마_표기_차이_흡수(self, no_embed_fallback):
        # xlsx 숫자셀 gold('38000') vs 모델의 자연 표기('38,000원') — 눈검증에서 잡은 오탐
        assert expected_points_coverage('정가는 38,000원입니다', ['38000']) == 1.0
        assert expected_points_coverage('연간 1,200,000원 이상', ['1200000']) == 1.0

    def test_중복_포인트_각자_집계(self, monkeypatch):
        # 임베딩 단계에서 points.index()가 첫 항목만 갱신하던 버그 고정
        from rag.embeddings import Embedding
        monkeypatch.setattr(gen, 'embed_texts_sync',
                            lambda texts: [Embedding(dense=[1.0, 0.0]) for _ in texts])  # 전부 동일 → 유사도 1
        score = expected_points_coverage('서술형 답변 문장입니다.', ['서술 포인트', '서술 포인트'])
        assert score == 1.0                                  # 수정 전엔 0.5

    def test_포인트_없으면_None(self):
        assert expected_points_coverage('답변', []) is None
