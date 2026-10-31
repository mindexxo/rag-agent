"""상용 OCR(OCR_PROVIDER=llm, #299) — 사진 표는 잘라서, 스캔 쪽은 쪽 통째로 LLM이 읽는다. 정의점은 rag/chunking/ocr_llm.py.

두 층:
  - 순수 함수(판정·묶음·자를 영역·답 해석·라우팅): docling 모델 없이 몇 초.
  - 실변환: docling + 가짜 VLM 서버(tests/conftest.py fake_vlm)로 chunk_file → 변환 자식 프로세스까지 실제 경로. 가짜 서버는 모든 요청에
    같은 글(vlm.caption)을 돌려주므로 테스트마다 사진 표 답·스캔 쪽 답을 그 글로 정한다.
"""

from pathlib import Path

import pytest

import rag.chunking as ch
from config import settings
from rag.chunking import LlmReadFailedError, chunk_file
from rag.chunking.ocr_llm import PdfPage, is_scan_page, page_runs, photo_box, require_pages, split_table_reply

# ── 순수 함수 ─────────────────────────────────────────────────────────────────────


class TestScanJudgement:
    def _page(self, chars, image_area):
        side = (image_area * 100 * 100) ** 0.5  # 100×100pt 쪽에 정사각형 사진
        return PdfPage(100.0, 100.0, chars, ((0.0, 0.0, side, side),) if image_area else ())

    def test_글자가_거의_없고_사진이_쪽을_덮으면_스캔이다(self):
        assert is_scan_page(self._page(0, 1.0))
        assert is_scan_page(
            self._page(ch.ocr_llm.SCAN_MAX_CHARS - 1, ch.ocr_llm.SCAN_MIN_IMAGE_AREA)
        )  # 쪽 번호·도장 몇 글자는 허용

    def test_글자가_있거나_사진이_작거나_없으면_스캔이_아니다(self):
        assert not is_scan_page(self._page(ch.ocr_llm.SCAN_MAX_CHARS, 1.0))  # 실측 최소 21자(표 이미지 축)가 이쪽
        assert not is_scan_page(self._page(0, 0.4))
        assert not is_scan_page(self._page(0, 0))  # 글자 없는 빈 쪽·표지

    def test_연속된_같은_종류끼리_묶는다(self):
        assert page_runs([False, False, True, False]) == [(False, 1, 2), (True, 3, 3), (False, 4, 4)]
        assert page_runs([True, True]) == [(True, 1, 2)]
        assert page_runs([]) == []


class TestPhotoBox:
    TABLE = (100.0, 200.0, 400.0, 400.0)  # 300×200

    def test_표를_담은_사진_전체를_자른다(self):
        # 결정 4 — 사진 안 표 제목이 docling 표 영역 밖에 있다(실문서 07: 사진 247pt 중 위 39pt)
        photo = (90.0, 160.0, 410.0, 420.0)
        assert photo_box(self.TABLE, (photo,)) == photo

    def test_여럿이면_가장_작은_사진(self):
        small, big = (90.0, 160.0, 410.0, 420.0), (50.0, 100.0, 450.0, 500.0)
        assert photo_box(self.TABLE, (big, small)) == small

    def test_사진이_너무_크면_표_영역만(self):
        # 쪽 배경·레터헤드 사진 — 통째로 보내면 사진 밖 디지털 글이 섞여 두 번 색인된다
        page_bg = (0.0, 0.0, 595.0, 842.0)
        assert photo_box(self.TABLE, (page_bg,)) == self.TABLE

    def test_표를_안_담은_사진은_무시한다(self):
        assert photo_box(self.TABLE, ((0.0, 0.0, 50.0, 50.0),)) == self.TABLE
        assert photo_box(self.TABLE, ()) == self.TABLE


class TestTableReply:
    def test_표_밖_줄은_글로_표는_행으로(self):
        reply = (
            '```markdown\n**<표 1> 지원 기준**\n\n| 구분 | 한도 |\n| :--- | ---: |\n| 어학 | 300,000원 |\n| 자격증 |\n```\n'
            '※ 연 1회 한도'
        )
        outside, rows = split_table_reply(reply)
        assert outside == '<표 1> 지원 기준 ※ 연 1회 한도'  # 아래 주석도 표 앞 글에 합친다
        assert rows == [['구분', '한도'], ['어학', '300,000원'], ['자격증', '']]  # 구분선 버림·짧은 행은 빈 칸으로

    def test_제목_줄의_마크다운_머리표시를_뗀다(self):
        assert split_table_reply('## 표 제목\n| a |')[0] == '표 제목'

    def test_표가_없으면_행이_비고_글만_남는다(self):
        assert split_table_reply('표가 아닌 글') == ('표가 아닌 글', [])


class TestRouting:
    def test_기본은_triton이다(self):
        from config import Settings

        assert Settings.model_fields['ocr_provider'].default == 'triton'

    def test_triton_모드는_상용_경로를_안_탄다(self, monkeypatch, tmp_path):
        # 사내 모드 불변(결정 1) — PDF도 예전 그대로 document_converter.docling_converter 하나로 변환한다
        monkeypatch.setattr(settings, 'ocr_provider', 'triton')
        monkeypatch.setattr(ch.ocr_llm, 'convert_pdf', lambda *a, **k: pytest.fail('상용 경로를 탔다'))
        monkeypatch.setattr(ch.document_converter, 'convert_with', lambda conv, p, **k: ('기존', conv))
        monkeypatch.setattr(ch.document_converter, 'docling_converter', lambda: 'standard')
        assert ch.convert_process.convert_local(tmp_path / 'a.pdf') == ('기존', 'standard')

    def test_llm_모드도_PDF만_상용_경로다(self, monkeypatch, tmp_path):
        monkeypatch.setattr(settings, 'ocr_provider', 'llm')
        monkeypatch.setattr(ch.ocr_llm, 'convert_pdf', lambda p, **k: '상용')
        monkeypatch.setattr(ch.document_converter, 'convert_with', lambda conv, p, **k: '기존')
        monkeypatch.setattr(ch.document_converter, 'docling_converter', lambda: 'standard')
        assert ch.convert_process.convert_local(tmp_path / 'a.PDF') == '상용'
        assert ch.convert_process.convert_local(tmp_path / 'a.docx') == '기존'

    def test_llm_모드는_Triton_OCR을_끈다(self, monkeypatch):
        pytest.importorskip('docling')
        from docling.datamodel.base_models import InputFormat

        monkeypatch.setattr(settings, 'docling_do_ocr', True)
        # 부모 프로세스의 변환기·그림 가드 목록은 테스트 뒤 되돌린다 — 다른 테스트가 만든 것을 버리거나 가드를 쌓지 않게(리뷰 반영)
        monkeypatch.setattr(ch.document_converter, '_converter', ch.document_converter._converter)
        monkeypatch.setattr(ch.document_converter, '_picture_guards', list(ch.document_converter._picture_guards))
        for provider, expected in (('triton', True), ('llm', False)):
            monkeypatch.setattr(settings, 'ocr_provider', provider)
            ch.document_converter._converter = None
            opts = ch.document_converter.docling_converter().format_to_options[InputFormat.PDF].pipeline_options
            assert opts.do_ocr is expected, provider


class TestEmptyReply:
    def test_빈_스캔_쪽은_일시_실패다(self):
        import types

        from rag.outbox import is_transient

        item = types.SimpleNamespace(prov=[types.SimpleNamespace(page_no=3)])
        doc = types.SimpleNamespace(iterate_items=lambda: iter([(item, 0)]))
        require_pages(doc, 3, 3)  # 3쪽에 글이 있으면 통과
        with pytest.raises(LlmReadFailedError, match=r'\[4\]') as e:
            require_pages(doc, 3, 4)
        assert is_transient(e.value)  # outbox가 문서를 다시 처리한다(결정 5)


# ── 실변환 (docling + 가짜 VLM) ────────────────────────────────────────────────────


def _text_png(path: Path, lines: list[str], size=(1240, 1754)) -> Path:
    from PIL import Image, ImageDraw, ImageFont

    im = Image.new('RGB', size, 'white')
    d = ImageDraw.Draw(im)
    font = ImageFont.load_default(size=40)
    for i, line in enumerate(lines):
        d.text((80, 80 + i * 70), line, fill='black', font=font)
    im.save(path)
    return path


def _table_photo_png(path: Path) -> Path:
    """제목 줄 + 격자 표 사진(글자층 없음). 내용은 가짜 VLM이 답하므로 무관 — 표로 분류되게 격자만 또렷하게."""
    from PIL import Image, ImageDraw, ImageFont

    rows = [['Part', 'Warranty', 'Fee'], ['Motor', '3y', '48000'], ['Blade', '1y', '12000'], ['Pack', '2y', '35000']]
    cw, rh, top = 400, 110, 120
    im = Image.new('RGB', (cw * 3, top + rh * len(rows) + 20), 'white')
    d = ImageDraw.Draw(im)
    font = ImageFont.load_default(size=44)
    d.text((10, 30), 'Table 1. Warranty fees', fill='black', font=font)
    for r, row in enumerate(rows):
        for c, cell in enumerate(row):
            x0, y0 = c * cw, top + r * rh
            d.rectangle([x0, y0, x0 + cw, y0 + rh], outline='black', width=4)
            d.text((x0 + 30, y0 + 28), cell, fill='black', font=font)
    im.save(path)
    return path


def _pdf(path: Path, pages: list) -> Path:
    """쪽 목록으로 PDF — 쪽마다 ('text', [줄...]) · ('photo', png, 줄...) · ('scan', png). canvas로 그려 위치를 정확히 둔다."""
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    W, H = A4
    c = canvas.Canvas(str(path), pagesize=A4)
    for kind, *rest in pages:
        if kind == 'scan':
            c.drawImage(str(rest[0]), 0, 0, width=W, height=H)  # 쪽 전체 사진, 글자층 없음
        else:
            lines = rest[-1]
            c.setFont('Helvetica-Bold', 20)
            c.drawString(60, H - 80, lines[0])
            c.setFont('Helvetica', 11)
            for i, line in enumerate(lines[1:]):
                c.drawString(60, H - 120 - i * 18, line)
            if kind == 'photo':
                c.drawImage(str(rest[0]), 60, H - 520, width=480, height=480 * 580 / 1200)
        c.showPage()
    c.save()
    return path


def _mixed_pdf(tmp_path) -> Path:
    """1쪽 디지털(본문+사진 표) · 2쪽 스캔(쪽 전체 사진) · 3쪽 디지털 — 섞인 문서(결정 3)."""
    photo = _table_photo_png(tmp_path / 'photo.png')
    scan = _text_png(tmp_path / 'scan.png', ['Scanned page', 'scanned body text'])
    return _pdf(
        tmp_path / 'mixed.pdf',
        [
            ('photo', photo, ['Digital Title', 'First page digital body text.']),
            ('scan', scan),
            ('text', ['Third Page', 'Third page digital body text.']),
        ],
    )


@pytest.fixture
def llm_ocr(fake_vlm, monkeypatch):
    pytest.importorskip('docling')
    monkeypatch.setattr(settings, 'ocr_provider', 'llm')
    return fake_vlm


class TestRealConversion:
    def test_쪽마다_판정한다(self, tmp_path):
        pages = ch.ocr_llm.scan_profile(_mixed_pdf(tmp_path), None)
        assert [is_scan_page(p) for p in pages] == [False, True, False]
        assert ch.ocr_llm.scan_profile(_mixed_pdf(tmp_path), 2) is None  # 쪽 상한 초과 — 거절은 docling에 맡긴다

    def test_섞인_문서는_쪽별로_읽고_쪽_번호를_지킨다(self, tmp_path, llm_ocr):
        # 가짜 VLM은 사진 표·스캔 쪽 모두에 같은 답을 준다 — 표 밖 줄 + 표. 스캔 쪽에선 같은 글이 마크다운 문서가 된다.
        # 쪽 번호는 변환 결과(요소의 쪽)로 본다 — 청크의 page는 청크가 시작한 쪽이라 제목 없는 쪽은 앞 쪽 청크에 이어 붙는다.
        llm_ocr.caption = 'Table 1. Warranty fees\n\n| Part | Fee |\n| --- | --- |\n| Motor | 48000 |'
        p = _mixed_pdf(tmp_path)
        doc = ch.docling_convert(p, max_num_pages=settings.docling_max_num_pages)  # 변환 자식 프로세스 — 운영과 같은 길
        by_page: dict[int, list[str]] = {}
        for label, text, page, table in ch.docling_elements.docling_elements(doc):
            by_page.setdefault(page, []).append(text or ' | '.join(' | '.join(r) for r in table or []))
        assert sorted(by_page) == [1, 2, 3]
        page1, page2 = '\n'.join(by_page[1]), '\n'.join(by_page[2])
        assert 'First page digital body text.' in page1  # 디지털 글은 글자층 그대로
        assert 'Motor | 48000' in page1 and 'Table 1. Warranty fees' in page1  # 사진 표 칸 + 제목 줄(결정 4)
        assert page1.index('Table 1. Warranty fees') < page1.index('Motor | 48000')  # 제목은 표 앞
        assert '48000' in page2  # 스캔 쪽은 LLM 마크다운
        assert 'Third page digital body text.' in '\n'.join(by_page[3])
        assert llm_ocr.calls >= 2  # 사진 표 1 + 스캔 쪽 1
        body = '\n'.join(c.text for c in chunk_file(p))  # 청킹까지 — 표가 표로 남는다
        assert '| Motor | 48000 |' in body and 'Third page digital body text.' in body

    def test_스캔_쪽의_문서_제목은_헤딩이_된다(self, tmp_path, llm_ocr):
        # '#' 줄은 docling title 라벨 — section_header로 안 바꾸면 PDF 청킹이 본문 줄로 떨어뜨린다(#296)
        p = _pdf(tmp_path / 'scan.pdf', [('scan', _text_png(tmp_path / 's.png', ['x']))])
        llm_ocr.caption = '# Scan Title\n\n## 1. Section\n\nbody line'
        chunks = chunk_file(p)
        assert chunks[0].heading_path[0] == 'Scan Title'
        assert any(c.heading_path[-1] == '1. Section' for c in chunks)

    def test_LLM이_실패하면_문서가_일시_실패한다(self, tmp_path, llm_ocr):
        # docling은 5xx를 몇 번 재시도한 뒤 빈 글자를 준다 — 빈 표·빈 쪽으로 ready가 되면 안 된다(결정 5)
        from rag.outbox import is_transient

        llm_ocr.status = 500
        with pytest.raises(LlmReadFailedError) as e:
            chunk_file(_mixed_pdf(tmp_path))
        assert is_transient(e.value)

    def test_출력이_상한에서_잘리면_결정적_실패다(self, tmp_path, llm_ocr):
        # 잘린 표·쪽은 행이 빠진 채 조용히 색인된다 — 다시 보내도 같은 길이라 재시도 대신 바로 failed(사유에 상한 이름, 리뷰 반영)
        from rag.outbox import is_transient

        llm_ocr.caption = '| Part | Fee |\n| --- | --- |\n| Motor | 48'
        llm_ocr.finish_reason = 'length'
        photo = _pdf(tmp_path / 'photo.pdf', [('photo', _table_photo_png(tmp_path / 'p.png'), ['Title', 'body'])])
        with pytest.raises(RuntimeError, match='OCR_LLM_MAX_TOKENS') as e:
            chunk_file(photo)
        assert not is_transient(e.value)
        scan = _pdf(tmp_path / 'scan.pdf', [('scan', _text_png(tmp_path / 's.png', ['x']))])
        with pytest.raises(
            RuntimeError, match='docling 변환 불완전'
        ) as e:  # 스캔 쪽은 docling이 PARTIAL_SUCCESS로 낸다
            chunk_file(scan)
        assert not is_transient(e.value)
