"""위첨자 각주 표시 — 글자 크기로 각주를 가려 본문에서 떼는 규칙과 PDF 사이드카. docling 없이 돈다."""

from rag.chunking.docling_elements import docling_elements, markers_from_words, strip_footnote_markers


class TestFootnoteMarkers:
    """위첨자 각주 참조 'N)'. 열거 '1) …'와 글자 모양이 같아 **글자 크기 비율**로만 가른다(각주는 본문의 약 0.7배)."""

    def test_같은_줄_본문보다_작은_N괄호_토큰만_각주(self):
        words = [
            ('승인자', 12.0, 144.0),
            (':', 12.0, 144.0),
            ('PM', 12.0, 144.0),
            ('2)', 8.0, 142.0),
            ('>팀장>사업부장', 12.0, 144.0),
        ]  # 위첨자는 top이 2pt 위
        assert markers_from_words(words) == {'2)'}

    def test_열거는_본문과_같은_크기라_각주가_아니다(self):
        words = [
            ('다.', 11.0, 300.0),
            ('1)', 11.0, 300.0),
            ('직원이', 11.0, 300.0),
            ('2)', 11.0, 300.0),
            ('회사의', 11.0, 300.0),
        ]
        assert markers_from_words(words) == set()

    def test_정의_줄은_줄_전체가_작아_걸리지_않는다(self):
        words = [('2)', 8.9, 500.0), ('PM', 8.9, 500.0), ('없는', 8.9, 500.0), ('경우', 8.9, 500.0)]
        assert markers_from_words(words) == set()

    def test_줄_묶기는_top_허용치_밖이면_다른_줄(self):
        # 작은 글자 '3)'이 혼자 다른 줄(10pt 아래)에 있으면 비교 대상이 없어 각주가 아니다
        words = [('계획대상선택', 14.0, 70.0), ('3)', 9.4, 85.0)]
        assert markers_from_words(words) == set()

    def test_본문에서_단어_뒤_표시만_뗀다(self):
        assert (
            strip_footnote_markers('( 승인자 : PM 2) > 팀장 > 사업부장 )', {'2)'})
            == '( 승인자 : PM > 팀장 > 사업부장 )'
        )
        assert (
            strip_footnote_markers('확정 실적은 PM 2) , 팀장 승인 전까지', {'2)'})
            == '확정 실적은 PM , 팀장 승인 전까지'
        )
        assert (
            strip_footnote_markers('계획대상선택 3) → [ 이전계획불러오기 ]', {'3)'})
            == '계획대상선택 → [ 이전계획불러오기 ]'
        )

    def test_정의_줄과_다른_번호_열거는_그대로(self):
        assert strip_footnote_markers('2) PM 없는 경우 생략', {'2)'}) == '2) PM 없는 경우 생략'
        enum = '다. 1) 직원이 임원을 수행하여 출장하는 경우 2) 회사의 손님을 수행'
        assert strip_footnote_markers(enum, set()) == enum  # 그 쪽에 각주 표시가 없으면 무변경
        assert strip_footnote_markers(enum, {'3)'}) == enum  # 다른 번호면 무변경

    def test_더_큰_숫자의_일부는_각주가_아니다(self):
        # lookbehind의 \\w가 숫자를 포함해 "서식 12)"의 "2)"를 지우면 숫자가 변조된다
        assert strip_footnote_markers('서식 12) 참조', {'2)'}) == '서식 12) 참조'
        assert strip_footnote_markers('총 1000 2) 명', {'2)'}) == '총 1000 명'  # 공백 뒤는 각주

    def test_두_자리_각주와_뒤에_붙은_화살표(self):
        words = [('상세', 12.0, 50.0), ('10)', 8.0, 48.0), ('→', 12.0, 50.0)]
        assert markers_from_words(words) == {'10)'}
        assert strip_footnote_markers('상세 10)→ 다음', {'10)'}) == '상세→ 다음'

    def test_줄바꿈_너머의_정의_줄_머리는_지우지_않는다(self):
        # 요소가 합쳐진 뒤 적용돼도 "상신\n2) PM 없는 경우 생략"의 줄 머리 '2)'는 남아야 한다
        text = '6. 상신\n2) PM 없는 경우 생략'
        assert strip_footnote_markers(text, {'2)'}) == text

    def test_요소_흐름에_쪽별로_적용된다(self, monkeypatch):
        class _El:
            def __init__(self, label, text, page):
                self.label, self.text, self.marker = label, text, ''
                self.prov = [type('P', (), {'page_no': page})()]

        class _Doc:
            def iterate_items(self):
                return [(_El('text', '승인자 : PM 2) > 팀장', 4), None), (_El('text', '열거 2) 유지', 5), None)]

        els = list(docling_elements(_Doc(), {4: {'2)'}}))
        assert els[0][1] == '승인자 : PM > 팀장'
        assert els[1][1] == '열거 2) 유지'  # 5쪽엔 각주 표시가 없다


class TestFootnoteSidecarPdf:
    """pdf_footnote_markers I/O 래퍼 — reportlab으로 그린 실제 PDF에서 위첨자만 잡히는지. docling 불필요."""

    def _pdf(self, tmp_path):
        from reportlab.pdfgen import canvas

        p = tmp_path / 'footnote.pdf'
        c = canvas.Canvas(str(p), pagesize=(595, 842))
        c.setFont('Helvetica', 12)
        c.drawString(72, 700, 'approver : PM')
        c.setFont('Helvetica', 8)
        c.drawString(152, 703, '2)')  # 위첨자 — 작고 3pt 위
        c.setFont('Helvetica', 12)
        c.drawString(166, 700, '> team > division')
        c.setFont('Helvetica', 12)
        c.drawString(72, 650, 'case 1) employee 2) guest')  # 열거 — 본문 크기
        c.setFont('Helvetica', 9)
        c.drawString(72, 100, '2) omitted when no PM')  # 정의 줄 — 전체가 작음
        c.showPage()
        c.save()
        return p

    def test_실제_PDF에서_위첨자만_각주로_잡힌다(self, tmp_path):
        from rag.chunking.docling_elements import pdf_footnote_markers

        assert pdf_footnote_markers(self._pdf(tmp_path)) == {1: {'2)'}}

    def test_못_여는_파일은_빈_dict와_경고(self, tmp_path, caplog):
        import logging

        from rag.chunking.docling_elements import pdf_footnote_markers

        with caplog.at_level(logging.WARNING, logger='rag.chunking'):
            assert pdf_footnote_markers(tmp_path / 'missing.pdf') == {}
        assert any('각주 사이드카 실패' in r.message for r in caplog.records)
