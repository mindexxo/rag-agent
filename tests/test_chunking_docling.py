"""docling 실변환 — 표·머리말·쪽 번호·파이프라인 설정 전달·변환 가드·원격 OCR. docling이 설치된 환경에서만 돈다."""

import pytest

from config import settings
from rag.chunking import chunk_file, count_picture_placeholders

docling = pytest.importorskip('docling', reason='docling 미설치 — 실변환 테스트는 설치 환경에서만')


class TestRealConversion:
    """실제 변환 — 결함 1·2·5가 모델로 처리되는지. reportlab 합성 픽스처."""

    def test_격자표_빈_셀이_자리를_지킨다(self, tmp_path):
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle

        p = tmp_path / 'grid.pdf'
        table = Table([['Part', 'Free', 'Company', 'Personal'], ['Board', '2y', 'O', ''], ['Water', 'none', '', 'O']])
        table.setStyle(TableStyle([('GRID', (0, 0), (-1, -1), 0.5, colors.black)]))
        SimpleDocTemplate(str(p), pagesize=A4).build([table])
        body = '\n'.join(c.text for c in chunk_file(p))
        assert '| Board | 2y | O |  |' in body
        assert '| Water | none |  | O |' in body  # 빈 셀을 지우면 두 줄이 같아진다 (결함 5)

    def test_인쇄_헤더_푸터가_본문에_없다(self, tmp_path):
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas

        W, H = A4
        p = tmp_path / 'printed.pdf'
        c = canvas.Canvas(str(p), pagesize=A4)
        c.setFont('Helvetica', 8)
        c.drawString(20, H - 20, '26. 9. 8. 오후 10:25 [총무 | 가이드] 문서 제목')
        c.drawString(20, 18, 'https://intranet.example.com/print?readform 1/1')
        c.setFont('Helvetica-Bold', 13)
        c.drawString(50, H - 80, 'Software License Guide')
        c.setFont('Helvetica', 10)
        for i in range(6):
            c.drawString(50, H - 120 - i * 18, f'Body sentence {i} about the approval process and returns.')
        c.save()
        chunks = chunk_file(p)
        joined = '\n'.join(c.text for c in chunks) + ' ' + ' '.join(' '.join(c.heading_path) for c in chunks)
        assert 'approval process' in joined  # 본문 보존 (결함 1)
        assert 'https://' not in joined and '오후 10:25' not in joined  # 부산물 제외

    def test_청크마다_실제_쪽(self, tmp_path):
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas

        W, H = A4
        p = tmp_path / 'multipage.pdf'
        c = canvas.Canvas(str(p), pagesize=A4)
        c.setFont('Helvetica-Bold', 16)
        c.drawString(50, H - 60, 'Single Heading For Whole Document')
        for page_no in range(3):
            c.setFont('Helvetica', 10)
            for i in range(30):
                c.drawString(50, H - 100 - i * 20, f'Page {page_no + 1} line {i} with enough words to fill the budget.')
            c.showPage()
        c.save()
        chunks = chunk_file(p)
        assert len({ch.page for ch in chunks}) >= 2
        for ch in chunks:
            first = ch.text.strip().split('\n')[0]
            if first.startswith('Page '):
                assert ch.page == int(first.split()[1])

    def test_변환기는_프로세스당_하나(self):
        from rag.chunking.document_converter import docling_converter

        assert docling_converter() is docling_converter()

    def test_메모리_설정이_파이프라인에_실제로_전달된다(self):
        # 조용히 무시되는 설정(값은 들어가는데 파이프라인이 그 필드를 안 읽는 경우)을 잡는다.
        # PdfPipelineOptions는 pydantic 모델이라 필드명이 바뀌면 대입 자체가 실패한다 —
        # docling 업그레이드 때 이 테스트가 먼저 깨진다.
        from docling.datamodel.base_models import InputFormat

        from rag.chunking.document_converter import docling_converter

        converter = docling_converter()
        opts = converter.format_to_options[InputFormat.PDF].pipeline_options
        assert opts.layout_batch_size == settings.docling_layout_batch_size
        assert opts.queue_max_size == settings.docling_queue_max_size
        assert opts.accelerator_options.num_threads == settings.docling_num_threads
        assert opts.do_ocr is settings.docling_do_ocr

    def test_캡션_설정이_파이프라인에_실제로_전달된다(self):
        # 위와 같은 이유. 특히 scale이 틀리면 VLM이 글자를 못 읽고 원문에 없는 내용을 지어내는데
        # 에러가 없어 드러나지 않는다 — 그래서 값까지 단언한다.
        from docling.datamodel.base_models import InputFormat

        from rag.chunking.document_converter import PICTURE_CAPTION_PROMPT, caption_headers, docling_converter

        converter = docling_converter()
        opts = converter.format_to_options[InputFormat.PDF].pipeline_options
        assert opts.do_picture_description is True
        assert opts.enable_remote_services is True  # 빠지면 OperationNotAllowed로 변환 자체가 죽는다
        api = opts.picture_description_options
        assert api.scale == settings.vlm_caption_scale
        assert api.timeout == settings.vlm_caption_timeout_seconds
        assert api.headers == caption_headers() == {}  # 키가 비면 인증 헤더 없음 — 기본 .env는 사내
        # temperature 0 = 결정적 캡션. 빠지면 샘플링으로 돌아가 재인제스트마다 캡션이 달라진다
        assert api.params == {
            'model': settings.vlm_caption_model,
            'temperature': settings.vlm_caption_temperature,
            'max_completion_tokens': settings.vlm_caption_max_tokens,
        }  # max_completion_tokens — gpt-5가 max_tokens를 거부한다
        # 출력 상한 — 빠지면 반복 고리 폭주가 색인되고 45초 상한을 넘겨 문서 재시도로 번진다
        assert 655 * 2 <= api.params['max_completion_tokens'] <= 4000
        # 면적 하한 1% — 내용 있는 작은 캡처는 캡션하고 아이콘은 제외한다
        assert api.picture_area_threshold == settings.vlm_caption_area_threshold == 0.01
        # XLSX만 하한 0 — 엑셀은 시트 칸 단위라 면적 비율이 그림 크기를 반영하지 못한다. DOCX는 1% 그대로
        office = {
            fmt: converter.format_to_options[fmt].pipeline_options.picture_description_options
            for fmt in (InputFormat.DOCX, InputFormat.XLSX)
        }
        assert office[InputFormat.DOCX].picture_area_threshold == 0.01
        assert office[InputFormat.XLSX].picture_area_threshold == 0.0
        assert office[InputFormat.XLSX].prompt == PICTURE_CAPTION_PROMPT  # 하한 말고는 같은 캡션 설정
        assert office[InputFormat.XLSX].params == api.params
        # 형식마다 캡션 감시(PictureGuard)가 따로 붙는다 — XLSX가 DOCX 파이프라인을 공유하지 않으므로
        from rag.chunking.document_converter import PictureGuard

        guarded = [
            p
            for p in converter.initialized_pipelines.values()
            if any(isinstance(m, PictureGuard) for m in p.enrichment_pipe)
        ]
        assert len(guarded) >= 3
        assert api.params['temperature'] == 0.0 and api.timeout == 45.0  # .env.dev가 config를 덮으므로 값 자체도 단언
        # 동시 1 — 동시로 보내면 vLLM 배치 탓에 온도 0이어도 캡션이 실행마다 갈린다. 올리기 전에 config 주석부터
        assert api.concurrency == settings.vlm_caption_concurrency == 1
        assert api.prompt == PICTURE_CAPTION_PROMPT
        assert str(api.url) == settings.vlm_caption_url  # AnyUrl — str 캐스팅 후 비교

    def test_VLM이_실패하면_문서가_일시_실패한다(self, tmp_path, fake_vlm):
        """docling은 VLM 오류를 삼켜 빈 캡션을 준다 — 그림 내용이 빠진 채 "완료"가 되지 않게 CaptionFailedError(일시)로 올린다.
        docling 업그레이드로 실패를 삼키는 방식이 바뀌면(예: 예외로 바뀜) 이 테스트가 먼저 알려준다."""
        from rag.chunking import CaptionFailedError

        fake_vlm.status = 500
        with pytest.raises(CaptionFailedError):
            chunk_file(_picture_pdf(tmp_path))

    def test_VLM이_되면_캡션이_자리표시를_대체한다(self, tmp_path, fake_vlm):
        from tests.conftest import FAKE_CAPTION

        chunks = chunk_file(_picture_pdf(tmp_path))
        assert FAKE_CAPTION in '\n'.join(c.text for c in chunks)
        assert count_picture_placeholders(chunks) == 0


def _picture_pdf(tmp_path, pictures: int = 1):
    """그림이 든 PDF — 1×1 PNG를 300×200pt로(쪽 면적의 약 12%, 캡션 면적 하한 위). 내용은 무관하다."""
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import Image as RLImage
    from reportlab.platypus import PageBreak, SimpleDocTemplate

    png = tmp_path / 'fig.png'
    png.write_bytes(
        bytes.fromhex(
            '89504e470d0a1a0a0000000d494844520000000100000001080200000090'
            '7753de0000000c4944415408d763f8cfc00000030101003c2f8e3b0000000049454e44ae426082'
        )
    )
    p = tmp_path / 'pic.pdf'
    story = []
    for _ in range(pictures):
        story += [RLImage(str(png), width=300, height=200), PageBreak()]
    SimpleDocTemplate(str(p), pagesize=A4).build(story)
    return p


# ── 원격 OCR ─────────────────────────────────────────────────────────────────
# 표 이미지(TableItem)의 셀은 Triton에 올린 RapidOCR이 읽는다. 여기서 고정하는 것 셋:
#   ① 옵션이 실제로 파이프라인에 들어가는가(특히 scale 3.0 — KServe 옵션 기본값 2.0과 다르다)
#   ② 원격이 죽으면 문서가 **실패**하는가(조용한 빈 격자 색인 금지 — docling은 삼키므로 우리가 올린다)
#   ③ 서버가 있으면 숫자·영문 표 셀이 실제로 읽히는가(도달 가능할 때만 — 한글은 폰트 자산이 필요해 안 쓴다)


def _reset_converter(ch):
    """설정 변경이 먹으려면 컨버터 싱글턴을 비워야 한다 — 전역 캐시라 재생성이 안 된다."""
    ch.document_converter._converter = None


def _triton_ready(base_url: str) -> bool:
    import requests

    try:
        return requests.get(f'{base_url}/v2/health/ready', timeout=2).status_code == 200
    except Exception:  # noqa: BLE001 — 도달 불가는 스킵 사유일 뿐
        return False


def _table_image_pdf(tmp_path):
    """숫자·영문 표를 **비트맵**으로 그려 PDF에 넣는다 — 텍스트 레이어가 없어 OCR만이 읽을 수 있다."""
    from PIL import Image, ImageDraw, ImageFont
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import Image as RLImage
    from reportlab.platypus import SimpleDocTemplate

    rows = [['Part', 'Warranty', 'Fee'], ['Motor', '3y', '48000'], ['Blade', '1y', '12000'], ['Pack', '2y', '35000']]
    cw, rh = 400, 110
    W, H = cw * 3, rh * len(rows) + 20
    im = Image.new('RGB', (W, H), 'white')
    d = ImageDraw.Draw(im)
    font = ImageFont.load_default(size=44)
    for r, row in enumerate(rows):
        for c, cell in enumerate(row):
            x0, y0 = c * cw, 10 + r * rh
            d.rectangle([x0, y0, x0 + cw, y0 + rh], outline='black', width=4)
            d.text((x0 + 30, y0 + 28), cell, fill='black', font=font)
    png = tmp_path / 'table.png'
    im.save(png)
    p = tmp_path / 'table_image.pdf'
    SimpleDocTemplate(str(p), pagesize=A4).build([RLImage(str(png), width=480, height=H * 480 / W)])
    return p, rows


class TestConversionGuard:
    """raises_on_error=True가 못 잡는 불완전 변환을 convert_process.convert_local이 올리는지 — 모델 없이 가짜 컨버터로."""

    # 가드는 변환 프로세스 안의 convert_process.convert_local에 있다 — 가짜 컨버터는 이 프로세스에서만 보이므로 그것을 직접 부른다.
    def _with_result(self, monkeypatch, status, errors):
        import types

        import rag.chunking as ch

        self.doc = types.SimpleNamespace(iterate_items=lambda: iter(()))
        res = types.SimpleNamespace(status=status, errors=errors, document=self.doc)
        conv = types.SimpleNamespace(convert=lambda *a, **k: res)
        monkeypatch.setattr(ch.document_converter, 'docling_converter', lambda: conv)
        return ch

    def test_부분_성공은_실패로_올린다(self, monkeypatch, tmp_path):
        # 문서 타임아웃이 나면 docling은 처리된 쪽까지만 담고 PARTIAL_SUCCESS로 **정상 반환**한다 —
        # 그대로 두면 뒤쪽 쪽이 빠진 문서가 ready가 된다.
        from docling.datamodel.base_models import ConversionStatus

        ch = self._with_result(monkeypatch, ConversionStatus.PARTIAL_SUCCESS, [])
        with pytest.raises(RuntimeError, match='partial_success'):
            ch.convert_process.convert_local(tmp_path / 'x.pdf')

    def test_성공이어도_errors가_있으면_실패로_올린다(self, monkeypatch, tmp_path):
        # 원격 OCR 실패는 status를 안 바꾸고 errors에만 남긴다(kserve_v2_ocr_model.py) — 표가 빈 격자로 색인된다.
        from docling.datamodel.base_models import ConversionStatus, DoclingComponentType, ErrorItem, FailureCategory

        err = ErrorItem(
            component_type=DoclingComponentType.MODEL,
            module_name='KserveV2OcrModel',
            error_message='connection refused',
            category=FailureCategory.INFERENCE_FAILURE,
        )
        ch = self._with_result(monkeypatch, ConversionStatus.SUCCESS, [err])
        with pytest.raises(RuntimeError, match='KserveV2OcrModel'):
            ch.convert_process.convert_local(tmp_path / 'x.pdf')

    def test_완전_성공은_통과한다(self, monkeypatch, tmp_path):
        from docling.datamodel.base_models import ConversionStatus

        ch = self._with_result(monkeypatch, ConversionStatus.SUCCESS, [])
        assert ch.convert_process.convert_local(tmp_path / 'x.pdf') is self.doc


class TestRemoteOcr:
    def test_OCR_설정이_파이프라인에_실제로_전달된다(self, monkeypatch):
        # 캡션 테스트와 같은 이유 — 조용히 무시되는 설정을 잡는다. scale이 특히 그렇다:
        # KserveV2OcrOptions 기본값은 2.0이라 빠뜨리면 에러 없이 3.0과 다른 결과가 나온다.
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import OcrMode

        import rag.chunking as ch

        monkeypatch.setattr(settings, 'docling_do_ocr', True)
        _reset_converter(ch)
        try:
            converter = ch.document_converter.docling_converter()
            opts = converter.format_to_options[InputFormat.PDF].pipeline_options
            assert opts.do_ocr is True
            o = opts.ocr_options
            assert o.kind == 'kserve_v2_ocr'
            assert o.scale == ch.document_converter.OCR_SCALE == 3.0
            assert o.lang == [ch.document_converter.OCR_LANG] == ['korean']
            assert o.transport == 'http'
            assert o.model_name == settings.ocr_kserve_model_name
            assert o.url == settings.ocr_kserve_url
            assert o.timeout == settings.ocr_kserve_timeout_seconds
            assert o.mode == OcrMode.DEFAULT  # 비트맵 영역만 — 본문 텍스트 레이어 무손상의 근거
        finally:
            _reset_converter(ch)

    def test_원격_OCR이_죽으면_문서가_실패한다(self, tmp_path, monkeypatch):
        """VLM과 반대다 — 캡션은 없어도 진행하지만 OCR 실패는 문서를 failed로 확정한다.

        docling은 KServe 실패를 삼키고 쪽을 그대로 내보내므로(errors에만 기록) 그대로 두면 표가
        **빈 격자로 조용히 색인**된다. '읽는다'는 약속이 깨진 것이라 실패로 올린다. 오래 죽어 있으면 docling_do_ocr로 끄는 게 운영 절차다(config.py).
        """
        import rag.chunking as ch

        p, _ = _table_image_pdf(tmp_path)
        monkeypatch.setattr(settings, 'docling_do_ocr', True)
        monkeypatch.setattr(settings, 'ocr_kserve_url', 'http://127.0.0.1:1')
        monkeypatch.setattr(settings, 'ocr_kserve_timeout_seconds', 2.0)
        _reset_converter(ch)
        try:
            with pytest.raises(RuntimeError, match='docling 변환 불완전'):
                chunk_file(p)
        finally:
            _reset_converter(ch)

    def test_표_이미지_셀을_원격_OCR이_읽는다(self, tmp_path, monkeypatch):
        import rag.chunking as ch

        if not _triton_ready(settings.ocr_kserve_url):
            pytest.skip(f'Triton OCR 미도달: {settings.ocr_kserve_url} (OCR_KSERVE_URL로 지정)')
        p, rows = _table_image_pdf(tmp_path)
        monkeypatch.setattr(settings, 'docling_do_ocr', True)
        _reset_converter(ch)
        try:
            body = '\n'.join(c.text for c in chunk_file(p))
        finally:
            _reset_converter(ch)
        for value in ('48000', '12000', '35000'):  # 금액 셀 — 표 이미지에서 제일 중요한 값
            assert value in body, body
