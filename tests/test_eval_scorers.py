"""C-1: eval 채점기 단위 테스트 — 수정된 채점 동작 고정.

채점기가 틀리면 측정 전체가 무의미 — corpus v2 기준선 측정의 전제.
임베딩 폴백은 가짜(_sync 패치)로 통제한다.
"""
import pytest

import eval.generation as gen
from eval.generation import _citation_match, citation_accuracy, expected_points_coverage
from rag.citation_labels import TAIL_END, TAIL_START, citation_tail
from rag.retrieval_types import RetrievedChunk


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
    """임베딩 2단만 검증하는 픽스처 — 1단(부분문자열)은 차단한다.

    가짜 코사인은 "포인트 텍스트가 답변 조각에 들어 있으면 1"이라, 1단을 살려두면
    같은 조건에서 1단이 먼저 잡아 2단이 영영 실행되지 않는다. 의미 유사도를
    흉내내지는 않는다 — 그건 실측(생성물 900행 눈검증)이 맡는 몫이고, 여기서는
    **판정 구조**(어떤 조각과 대조하는가)만 본다.
    """
    from rag.embeddings import Embedding
    monkeypatch.setattr(gen, '_contains_point', lambda *a: False)
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

    # --- 목록 블록 결합 (#196) — 한 사실이 "머리말 + 불릿"으로 쪼개진 경우 -------------
    # 생성 프롬프트 규칙 8이 목록 서식을 지시하므로 이 형태는 정상 출력이다. 결합 후보가
    # 없으면 주어가 든 줄과 값이 든 줄이 따로 놀아 어느 쪽도 포인트와 안 닮는다(#196 실측).

    def test_머리말과_불릿을_묶은_후보가_생긴다(self):
        segs = gen._split_segments('수습 및 시용기간은 다음과 같습니다\n'
                                   '- 일반 신규 채용: 3개월 이상\n'
                                   '- 특별 채용: 두지 않을 수 있음')
        joined = [x for x in segs if '수습 및 시용기간은 다음과 같습니다' in x and '3개월 이상' in x]
        assert joined, f'머리말+불릿 결합 후보가 없다: {segs}'

    def test_머리말_없는_런만도_후보로_남긴다(self):
        # 머리말이 다른 사실일 수 있어 런 단독 후보도 필요하다
        segs = gen._split_segments('- HR팀장 합의\n- 재무팀장 합의\n- 경영전략부문장 결재')
        assert any('HR팀장 합의 - 재무팀장 합의 - 경영전략부문장 결재' in x for x in segs), segs

    def test_런_전체를_묶는다_고정윈도우가_아니다(self):
        # 결재선 4줄을 2줄씩 묶으면 체인 절반만 모여 여전히 미달이었다(#196 실측 0.734)
        segs = gen._split_segments('- 작성자\n- HR팀장\n- 재무팀장\n- 경영전략부문장')
        assert any(all(k in x for k in ('작성자', 'HR팀장', '재무팀장', '경영전략부문장'))
                   for x in segs), segs

    def test_불릿_한_줄짜리는_묶지_않는다(self):
        # 쪼개진 게 아니므로 결합 후보를 만들 이유가 없다 — 후보만 늘면 잡음이다
        segs = gen._split_segments('안내드립니다\n- 단일 항목입니다')
        assert '안내드립니다 - 단일 항목입니다' not in segs, segs

    def test_번호_목록도_같이_묶는다(self):
        segs = gen._split_segments('절차는 다음과 같습니다\n1) 신청\n2) 승인')
        assert any('1) 신청 2) 승인' in x for x in segs), segs

    def test_마침표_번호목록도_묶는다(self):
        # _SENT_SPLIT이 숫자 뒤 마침표를 문장 경계로 잘라 '1. 신청'이 ['1','신청']이 된다 —
        # 문장 분리 결과로 탐지하면 이 형태가 영영 안 걸린다(리뷰 지적). 원문 줄로 봐야 한다.
        segs = gen._split_segments('절차는 다음과 같습니다\n1. 신청\n2. 승인')
        assert any('1. 신청 2. 승인' in x for x in segs), segs

    def test_빈_줄이_끼어도_머리말을_찾는다(self):
        # 실제 생성물은 머리말과 목록 사이에 빈 줄이 있다(마크다운 관행)
        segs = gen._split_segments('수습기간은 다음과 같습니다\n\n- 일반: 3개월\n- 특별: 없음')
        assert any('수습기간은 다음과 같습니다' in x and '3개월' in x for x in segs), segs

    def test_결합_후보는_원문에_없는_내용을_만들지_않는다(self):
        """위양성 방지의 핵심 불변식 — 결합은 **있는 줄을 잇는 것**이지 합성이 아니다.

        임계(0.75)를 낮추지 않고 후보만 늘리는 설계라, 안전성은 "후보가 답변에 실제로
        존재하는 문자열인가"에 달려 있다. 이게 깨지면 없는 사실이 통과할 수 있다.
        """
        text = ('지원 금액은 다음과 같습니다\n'
                '- 유치원: 24회차까지\n'
                '- 대학교: 8학기까지')
        src = text.replace('\n', ' ')
        for cand in gen._bullet_run_candidates(text):
            for tok in cand.split():
                assert tok in src, f'원문에 없는 토큰 {tok!r}이 후보 {cand!r}에 들어갔다'

    def test_런이_끊기면_따로_묶는다(self):
        # 목록 사이에 설명 문단이 끼면 서로 다른 사실이다 — 가로질러 묶으면 안 된다
        segs = gen._split_segments('첫째 묶음\n- A1\n- A2\n설명 문단입니다\n- B1\n- B2')
        assert not any('A2' in x and 'B1' in x for x in segs), segs

    def test_기존_문장절_후보는_그대로_남는다(self):
        # 결합은 **추가**다. 기존 후보를 대체하면 긴 포인트가 손해를 본다
        segs = gen._split_segments('안내드립니다\n- A는 1입니다\n- B는 2입니다')
        assert '안내드립니다' in segs and '- A는 1입니다' in segs and '- B는 2입니다' in segs, segs


@pytest.fixture
def no_embed_fallback(monkeypatch):
    """임베딩 2단을 무력화 — 부분문자열 1단만 검증할 때."""
    from rag.embeddings import Embedding
    monkeypatch.setattr(gen, 'embed_texts_sync',
                        lambda texts: [Embedding(dense=[0.0] * 8) for _ in texts])


class TestExpectedPointsCoverage:
    # ── 1단: 부분문자열 (서술문 포인트의 원문 인용을 임베딩 감쇠 없이 인정)
    def test_숫자_경계_오탐_방지(self, no_embed_fallback):
        # '30분'이 '130분'에 매칭되면 안 됨 (수정 전 오탐)
        assert expected_points_coverage('처리에 130분 걸립니다', ['30분']) == 0.0

    def test_콤마_표기_차이_흡수(self, no_embed_fallback):
        # xlsx 숫자셀 gold('38000') vs 모델의 자연 표기('38,000원')
        assert expected_points_coverage('정가는 38,000원입니다', ['38000']) == 1.0

    def test_공백_변형_허용(self, no_embed_fallback):
        assert expected_points_coverage('3 0분... 아니 30 분 이내', ['30분']) == 1.0

    # ── 2단: 임베딩 (패러프레이즈)
    def test_절에_담긴_포인트를_잡는다(self, fake_embed):
        # 문장 단위로만 보면 못 잡던 것 — 비교 단위를 절까지 내린 이유
        assert expected_points_coverage('A는 X고, B는 Y입니다.', ['B는 Y입니다']) == 1.0

    def test_없는_포인트는_감점(self, fake_embed):
        assert expected_points_coverage('A는 X입니다.', ['B는 Y입니다']) == 0.0

    def test_중복_포인트_각자_집계(self, fake_embed):
        assert expected_points_coverage('서술문 포인트입니다.', ['서술문 포인트', '서술문 포인트']) == 1.0

    def test_포인트_없으면_None(self):
        assert expected_points_coverage('답변', []) is None

    def test_빈_답변은_0(self, fake_embed):
        assert expected_points_coverage('', ['어떤 포인트']) == 0.0


class TestMustNotContainViolations:
    """오정보 채점 (#95) — 금지값이 답변에 섞였는지. 매칭 정의점은 _contains_point 재사용."""

    def test_필드_없으면_None(self):
        # None = 해당 없음 — 위반율 분모에서 제외된다 (빈 리스트=위반 없음과 다른 값)
        assert gen.must_not_contain_violations('아무 답변', []) is None

    def test_위반_없으면_빈리스트(self):
        assert gen.must_not_contain_violations('30일 이내 반품됩니다', ['14일지나면반품불가']) == []

    def test_금지값_등장하면_그_값을_반환(self):
        # 반환이 bool이 아닌 목록인 이유: 어떤 문구가 걸렸는지 결과 파일에 감사용으로 남긴다
        v = gen.must_not_contain_violations('네, 보증기간은 9개월입니다.', ['보증기간은 9개월'])
        assert v == ['보증기간은 9개월']

    def test_숫자_경계_오탐_방지(self):
        # '14일'이 '114일'에 걸리면 안 됨 — _contains_point의 숫자 경계 규칙 상속 검증
        assert gen.must_not_contain_violations('처리에 114일 걸립니다', ['14일']) == []

    def test_콤마_표기_차이_흡수(self):
        # xlsx 숫자셀('32000')과 모델 표기('32,000원') — _contains_point의 콤마 규칙 상속
        assert gen.must_not_contain_violations('정가는 32,000원입니다', ['32000']) == ['32000']

    def test_공백_차이_흡수(self):
        assert gen.must_not_contain_violations('등급 구간은 50만 원부터입니다', ['50만원']) == ['50만원']

    def test_여러_금지값_중_걸린_것만(self):
        v = gen.must_not_contain_violations(
            '위약금은 50%이며 예약금은 30%입니다', ['위약금은 50%', '예약금은 10%', '무관한 값'])
        assert v == ['위약금은 50%']

    def test_정정_문장은_단언문_규약으로_통과(self):
        # gold 규약 검증: 금지값이 정답에 정당하게 등장할 수 있으면 단언문으로 쓴다.
        # 정정 답변("9개월은 신 기준, 고객님 건은 6개월")은 맨값 '9개월'을 담지만
        # 단언문 '보증기간은 9개월'과는 매칭되지 않아야 한다.
        answer = '2026-07-01 이후 구매는 9개월이지만, 고객님 구매 건의 보증기간은 6개월입니다.'
        assert gen.must_not_contain_violations(answer, ['보증기간은 9개월']) == []
