"""OCR 서버(serving/triton-ocr) 엔진 파라미터 고정(#168·#224) — 서버 코드는 Triton 안에서만 import되므로 함수 소스를 읽어 실행한다.

eval.scan_quality.ocr_variant가 같은 방식으로 서버 파라미터를 읽어 로컬 재현에 쓴다 — 이 테스트가 그 읽기 경로도 함께 지킨다.
"""
import pytest

pytest.importorskip('rapidocr')


def test_글자_검출은_이미지를_키우지_않는다():
    # #224: 기본(min·736)은 잘라낸 제목 띠를 6배 넘게 키워 스캔 PDF 제목을 깨뜨렸다 — 되돌리면 이 테스트가 막는다
    from eval.scan_quality.ocr_variant import server_params
    p = server_params()
    assert p['Det.limit_type'] == 'max'
    assert 'Det.limit_side_len' not in p          # max에선 쓰이지 않는다(960/1500/2000 자동) — 넣으면 의도가 흐려진다


def test_각도_분류기_끔과_GPU_수치_고정은_그대로다():
    # #168 결정들 — #224 변경이 건드리지 않았는지
    from eval.scan_quality.ocr_variant import ROOT, server_params
    src = (ROOT / 'serving/triton-ocr/model_repository/ocr/1/model.py').read_text()
    assert 'use_cls=False' in src
    assert server_params()['Global.text_score'] == 0.5
