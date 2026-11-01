"""docling 변환 자식 프로세스 — 마감 kill·재기동·직렬화 무손실·abort·설정 전달. docling이 설치된 환경에서만 돈다."""

from pathlib import Path

import pytest

from config import settings

docling = pytest.importorskip('docling', reason='docling 미설치 — 실변환 테스트는 설치 환경에서만')


# ── 변환 프로세스 ─────────────────────────────────────────────────────
# docling에는 취소 API가 없다 — 스레드로 돌면 마감을 넘겨도 멈출 수 없어 고아 스레드가 세마포어를 쥐고 남는다.
# 그래서 변환을 자식 프로세스로 돌리고 마감에 kill한다. 여기서 고정하는 것:
#   ① 마감을 넘기면 TimeoutError이고, 자식이 교체돼 다음 변환은 정상이다
#   ② 자식이 죽어 있어도 다음 요청이 새로 띄운다
#   ③ 자식을 거쳐 온 문서가 이 프로세스에서 변환한 것과 같다(직렬화 무손실)


class TestConvertProcess:
    def test_마감을_넘기면_TimeoutError_자식_교체_다음_변환은_정상(self, tmp_path, monkeypatch, fake_vlm):
        import rag.chunking as ch

        p = _picture_pdf(tmp_path)
        deadline = settings.docling_document_timeout_seconds
        fake_vlm.delay = 30  # 캡션 단계에서 멈춘다 — 이 단계도 마감이 끊어야 한다
        monkeypatch.setattr(settings, 'docling_document_timeout_seconds', 3.0)
        with pytest.raises(TimeoutError, match='3초'):
            ch.docling_convert(p)
        assert ch.convert_process.process._proc is None  # kill하고 비웠다 — 다음 요청이 새로 띄운다
        fake_vlm.delay = 0
        monkeypatch.setattr(settings, 'docling_document_timeout_seconds', deadline)
        assert ch.docling_convert(p).pictures  # 새 자식에서 정상 변환

    def test_자식이_죽어_있으면_다음_요청이_새로_띄운다(self, tmp_path, fake_vlm):
        import rag.chunking as ch

        p = _picture_pdf(tmp_path)
        ch.docling_convert(p)
        old = ch.convert_process.process._proc
        old.kill()
        old.wait(10)  # OOM kill 같은 외부 종료
        assert ch.docling_convert(p).pictures
        assert ch.convert_process.process._proc is not old

    def test_자식을_거친_문서가_이_프로세스_변환과_같다(self, tmp_path, fake_vlm):
        # 자식은 DoclingDocument를 피클로 돌려준다. 손실이 있으면 청크가 조용히 달라진다 — 형식별로 대조한다
        from io import BytesIO

        import openpyxl
        from docx import Document
        from PIL import Image

        import rag.chunking as ch

        buf = BytesIO()
        Image.new('RGB', (120, 80), (200, 30, 30)).save(buf, format='PNG')
        d = Document()
        d.add_heading('제목', level=1)
        d.add_paragraph('본문 문단.')
        t = d.add_table(rows=1, cols=2)
        t.cell(0, 0).text = '칸'
        d.add_picture(BytesIO(buf.getvalue()))
        docx = tmp_path / 'a.docx'
        d.save(docx)
        wb = openpyxl.Workbook()
        wb.active.append(['등급', '적립'])
        wb.active.append(['VIP', '2.0'])
        xlsx = tmp_path / 'a.xlsx'
        wb.save(xlsx)
        for path, kw in (
            (_picture_pdf(tmp_path), {'max_num_pages': settings.docling_max_num_pages}),
            (docx, {}),
            (xlsx, {}),
        ):
            assert (
                ch.docling_convert(path, **kw).export_to_dict()
                == ch.convert_process.convert_local(path, **kw).export_to_dict()
            ), path

    def test_abort하면_기다리던_변환이_바로_끝나고_다음_변환은_정상(self, tmp_path, fake_vlm):
        # 색인 잡이 취소될 때(rag/ingestion.py) 부른다 — 기다리던 스레드가 마감(900초)까지 잠금을 쥐지 않게
        import threading
        import time

        import rag.chunking as ch

        p = _picture_pdf(tmp_path)
        ch.docling_convert(p)  # 자식 기동(모델 로딩)은 측정에서 뺀다
        fake_vlm.delay = 60
        errors = []
        t = threading.Thread(target=lambda: errors.append(_raises(ch.docling_convert, p)))
        t.start()
        time.sleep(3)  # 캡션 단계에 들어가 VLM을 기다리는 중
        started = time.monotonic()
        ch.abort_conversion()
        t.join(30)
        assert not t.is_alive() and time.monotonic() - started < 15
        assert isinstance(errors[0], RuntimeError) and '비정상 종료' in str(errors[0])
        fake_vlm.delay = 0
        assert ch.docling_convert(p).pictures

    def test_변환과_무관한_설정이_바뀌면_자식을_다시_띄우지_않는다(self, tmp_path, monkeypatch):
        # 자식 재기동은 콜드 기동(수 초)이라 변환 설정이 바뀔 때만 해야 한다
        import rag.chunking as ch

        p = tmp_path / 'a.md'
        p.write_text('# 제목\n\n본문.\n', encoding='utf-8')
        ch.docling_convert(p)
        first = ch.convert_process.process._proc
        monkeypatch.setattr(settings, 'rerank_timeout', settings.rerank_timeout + 1)
        ch.docling_convert(p)
        assert ch.convert_process.process._proc is first
        monkeypatch.setattr(
            settings, 'docling_table_mode', 'fast' if settings.docling_table_mode != 'fast' else 'accurate'
        )
        ch.docling_convert(p)
        assert ch.convert_process.process._proc is not first  # 변환 설정은 여전히 자식에 닿아야 한다

    def test_이_모듈이_읽는_설정은_모두_자식에_넘어간다(self):
        # 자식은 CHILD_SETTING_PREFIXES 설정만 받는다 — 다른 접두사 설정을 자식 코드가 읽으면 부모가 바꿔도 옛 값으로 돈다
        import re

        import rag.chunking as ch

        src = ''.join(f.read_text(encoding='utf-8') for f in Path(ch.__file__).parent.glob('*.py'))  # 패키지 전체
        names = set(re.findall(r'settings\.([a-z_]+)', src)) - {'model_dump'}
        assert names and all(n.startswith(ch.convert_process.CHILD_SETTING_PREFIXES) for n in names), sorted(names)


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


def _raises(fn, *args):
    try:
        fn(*args)
    except Exception as e:  # noqa: BLE001
        return e
    return None
