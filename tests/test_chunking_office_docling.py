"""DOCX·XLSX 공통 — 채팅 첨부가 docling을 타지 않는지와 워드·엑셀 변환기 캡션 설정이 PDF와 같은지."""

import openpyxl
import pytest

pytest.importorskip('docling')


@pytest.fixture(autouse=True)
def _vlm(fake_vlm, monkeypatch):
    """캡션은 로컬 대역이 준다. 설정이 바뀌면 변환 프로세스가 스냅숏 불일치로 재기동하므로 싱글턴을 비울 필요가 없다."""
    return fake_vlm


def _docx(tmp_path, build, name='t.docx'):
    from docx import Document

    doc = Document()
    build(doc)
    p = tmp_path / name
    doc.save(p)
    return p


def _xlsx(tmp_path, build, name='t.xlsx'):
    wb = openpyxl.Workbook()
    build(wb)
    p = tmp_path / name
    wb.save(p)
    return p


# ── 채팅 첨부는 docling을 타지 않는다 ─────────────────────────────────────────────


class TestAttachmentStaysPlain:
    def test_첨부_추출은_docling_변환을_부르지_않는다(self, tmp_path, monkeypatch):
        # 첨부는 사용자 대기 경로 — docling 변환 프로세스를 색인과 나누면 PDF 색인 뒤에 줄을 선다
        from rag.chunking import convert_process, extract_text

        def boom(*a, **k):
            raise AssertionError('첨부가 docling을 탔다')

        monkeypatch.setattr(convert_process, 'docling_convert', boom)

        def build_x(wb):
            wb.active.append(['col'])
            wb.active.append(['첨부값'])

        def build_d(doc):
            doc.add_paragraph('첨부 본문')

        assert '첨부값' in extract_text(_xlsx(tmp_path, build_x))
        assert '첨부 본문' in extract_text(_docx(tmp_path, build_d))
        md = tmp_path / 'a.md'  # MD 색인은 docling — 첨부는 원문 그대로다
        md.write_text('# 제목\n\n| 칸 | 값 |\n| --- | --- |\n| A\\|B | 1<br>2 |\n', encoding='utf-8')
        assert extract_text(md) == md.read_text(encoding='utf-8').strip()


# ── 변환기 설정 ────────────────────────────────────────────────────────────────────


class TestConverterOptions:
    def test_워드_엑셀_캡션_설정이_PDF와_같다(self):
        # 조용히 무시되는 설정을 잡는 테스트 — 한 형식만 캡션이 빠지거나 다른 프롬프트로 돌면 에러 없이 품질만 갈린다
        from docling.datamodel.base_models import InputFormat

        from rag.chunking.document_converter import docling_converter

        converter = docling_converter()
        pdf = converter.format_to_options[InputFormat.PDF].pipeline_options
        for fmt in (InputFormat.DOCX, InputFormat.XLSX):
            opts = converter.format_to_options[fmt].pipeline_options
            assert opts.do_picture_description is True
            assert opts.enable_remote_services is True  # 빠지면 OperationNotAllowed로 변환 자체가 죽는다
            # XLSX는 면적 하한만 0(시트 칸 단위라 비율이 그림 크기를 반영 못 함), 그 밖의 캡션 설정은 PDF와 같다
            expected = pdf.picture_description_options
            if fmt == InputFormat.XLSX:
                expected = expected.model_copy(update={'picture_area_threshold': 0.0})
            assert opts.picture_description_options == expected
