"""측정용 VLM 엔진·요약 계약(#296) — 네트워크·키 없이.

고정하는 것:
1. 헤딩 연결: VLM이 '#'로 쓴 title 라벨은 section_header로 바뀌고(PDF 청킹이 헤딩으로 보는 라벨), 글·목록은 그대로다 —
   글 모양으로 헤딩을 추측하던 보정은 오탐(목록 → 헤딩)으로 기각했으므로 되살아나면 안 된다
2. install/uninstall은 rag.chunking의 변환기를 바꿨다가 정확히 원복한다 — 스캔 대조의 디지털 기준 변환이 VLM을 타면 비교가 무너진다
3. PDF만 VLM으로, 다른 형식은 원래 변환기
4. 빈 응답이 5회 연속이면 측정을 멈춘다 — 한도 초과가 '미검출 0점'으로 둔갑하지 않게(10/24 실측 사고)
5. 엔진 이름표에 키가 섞이지 않는다
6. 요약: 숫자 오독(기대가 숫자인 칸의 오독)·추가(환각 후보)를 센다
"""

import json
from types import SimpleNamespace

import pytest

import rag.chunking as C
from eval import vlm_engine


def _doc():
    from docling_core.types.doc import DocItemLabel, DoclingDocument

    d = DoclingDocument(name='t')
    d.add_title(text='문서 제목')
    d.add_text(
        label=DocItemLabel.TEXT, text='1. 주문번호, 상품명, 수령일'
    )  # 번호 꼴이지만 목록 항목 — 헤딩으로 올리면 안 된다
    d.add_heading(text='2.1 사진 증빙', level=1)
    return d


def _labels(d):
    return [(it.label.value, it.text) for it, _ in d.iterate_items()]


def test_title만_section_header로_바뀌고_글은_그대로():
    d = vlm_engine._fix_headings(_doc())
    assert _labels(d) == [
        ('section_header', '문서 제목'),
        ('text', '1. 주문번호, 상품명, 수령일'),
        ('section_header', '2.1 사진 증빙'),
    ]


def test_install_uninstall은_변환기를_원복한다():
    before = (C.convert_process.convert_local, C.docling_convert)
    vlm_engine.install()
    try:
        assert C.convert_process.convert_local is vlm_engine._convert_vlm
    finally:
        vlm_engine.uninstall()
    assert (C.convert_process.convert_local, C.docling_convert) == before and vlm_engine._converter is None


def test_PDF가_아니면_원래_변환기(monkeypatch):
    called = []
    monkeypatch.setattr(vlm_engine, '_orig_convert_local', lambda p, **k: called.append(p) or 'orig')
    assert vlm_engine._convert_vlm('a.docx') == 'orig' and called == ['a.docx']


def test_빈_응답_5회면_측정_중단(monkeypatch):
    monkeypatch.setattr(vlm_engine.time, 'sleep', lambda s: None)
    monkeypatch.setattr(vlm_engine.os, '_exit', lambda code: (_ for _ in ()).throw(SystemExit(code)))
    calls = []
    wrapped = vlm_engine._paced(lambda *a, **k: calls.append(1) or SimpleNamespace(text=''))
    with pytest.raises(SystemExit) as e:
        wrapped()
    assert e.value.code == 3 and len(calls) == 5


def test_정상_응답은_원래_응답을_남기고_돌려준다(monkeypatch):
    monkeypatch.setattr(vlm_engine, 'RAW', [])
    r = vlm_engine._paced(lambda *a, **k: SimpleNamespace(text='## 1. 목적'))()
    assert r.text == '## 1. 목적' and vlm_engine.RAW == ['## 1. 목적']


def test_빈_응답_뒤_재시도로_성공(monkeypatch):
    monkeypatch.setattr(vlm_engine.time, 'sleep', lambda s: None)
    monkeypatch.setattr(vlm_engine, 'RAW', [])
    seq = iter(['', '## 1. 목적'])
    r = vlm_engine._paced(lambda *a, **k: SimpleNamespace(text=next(seq)))()
    assert r.text == '## 1. 목적' and vlm_engine.RAW == ['## 1. 목적']


def test_온도는_옵션_필드로_넘어간다(monkeypatch):
    """docling ApiVlmModel은 params 위에 vlm_options.temperature를 덮어쓴다 — params에만 넣으면 기본 0.0이 나가 gpt-5가 400(10/25 실측)."""
    from config import settings

    monkeypatch.setattr(settings, 'vlm_caption_temperature', 1.0)
    conv = vlm_engine.build_converter()
    from docling.datamodel.base_models import InputFormat

    vo = conv.format_to_options[InputFormat.PDF].pipeline_options.vlm_options
    assert vo.temperature == 1.0 and 'temperature' not in vo.params
    assert (
        vo.params['max_completion_tokens'] == vlm_engine.MAX_TOKENS and 'max_tokens' not in vo.params
    )  # gpt-5는 max_tokens 거부(#290)


def test_헤딩_보정_스위치(monkeypatch):
    doc = _doc()
    monkeypatch.setattr(
        vlm_engine,
        '_converter',
        SimpleNamespace(
            convert=lambda *a, **k: SimpleNamespace(
                status=__import__(
                    'docling.datamodel.base_models', fromlist=['ConversionStatus']
                ).ConversionStatus.SUCCESS,
                errors=[],
                document=doc,
            )
        ),
    )
    monkeypatch.setenv('VLM_FIX_HEADINGS', '0')
    assert _labels(vlm_engine._convert_vlm('x.pdf'))[0][0] == 'title'  # 끄면 docling 그대로
    monkeypatch.setenv('VLM_FIX_HEADINGS', '1')
    assert _labels(vlm_engine._convert_vlm('x.pdf'))[0][0] == 'section_header'


def test_이름표에_키가_없다(monkeypatch):
    from config import settings

    monkeypatch.setattr(settings, 'vlm_caption_api_key', 'sk-secret')
    monkeypatch.setattr(settings, 'vlm_caption_url', 'https://api.openai.com/v1/chat/completions')
    monkeypatch.setattr(settings, 'vlm_caption_model', 'gpt-5-mini')
    assert vlm_engine.label() == 'vlm_gpt-5-mini@api.openai.com' and 'secret' not in vlm_engine.label()
    monkeypatch.setattr(settings, 'vlm_caption_url', 'https://user:sk-secret@api.openai.com/v1/chat/completions')
    assert 'secret' not in vlm_engine.label()


def test_요약은_숫자오독과_추가를_센다(tmp_path):
    from eval.ocr_vlm_summary import table_matrix

    run = tmp_path / 'eng'
    run.mkdir()
    row = dict(
        id='S07',
        axis='07 숫자',
        path='img',
        layer='l1',
        seconds=1.0,
        clean=False,
        cells=[2, 4],
        numeric=[1, 2],
        presence=[3, 4],
        mismatches=[
            dict(kind='오독', expected='12,000원', actual='12,00원'),
            dict(kind='오독', expected='본인 결혼', actual='본인결혼'),
            dict(kind='추가', expected='', actual='지어낸 칸'),
        ],
    )
    (run / 'results.jsonl').write_text(json.dumps(row, ensure_ascii=False) + '\n')
    out = table_matrix([run])
    assert '숫자오독 1' in out and '추가 1' in out and '2/4' in out
