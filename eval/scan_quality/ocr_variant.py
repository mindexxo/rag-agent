"""OCR 검출 설정 실험 — 운영 OCR 서버(serving/triton-ocr)와 **같은 엔진·같은 파라미터**를 이 프로세스에서 돌리고 검출 크기
설정만 바꾼다. 운영 코드·서버는 건드리지 않는다.

배경: RapidOCR의 글자 검출은 Det.limit_type=min·limit_side_len=736 — 넘겨받은 이미지의 짧은 변을 736px로
**키운다**. docling은 쪽을 통째로가 아니라 레이아웃 영역을 잘라 OCR하므로 납작한 제목 띠(834×118)가 6배 넘게 커져 검출이
흔들린다("기부금영수증 발급 규정" → "금영 / 밤 / T / 0"). 쪽 전체(짧은 변 1,836)는 키우지 않아 정상. 큰 이미지 축소는
Global.max_side_len(2000)이 검출 앞에서 이미 하므로 설정과 무관하다.

변형은 서버 파라미터(_rapidocr_params) 위에 덮어쓰는 값만 다르다. 서버 코드는 'max'라 rapidocr 기본(min·736)은
local_min736으로 명시해 덮어쓴다. 로컬 CPU가 서버를 재현하는지는 같은 설정의 local ↔ remote로 먼저 확인한다
(GPU·CPU 차이로 글머리표 줄 순서·줄 묶음은 달라질 수 있다).

install(name): docling_convert를 이 프로세스의 convert_process.convert_local로, KserveV2OcrOptions를 같은 파라미터의 RapidOcrOptions로 바꾼다.
'remote'는 OCR 옵션은 그대로 두고 변환만 이 프로세스로 옮긴다(실험 간 같은 경로를 쓰려고).
"""

from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
VARIANTS = {
    'remote': None,  # 운영 서버 그대로(배포된 설정)
    'local_max': {},  # 서버 코드 그대로(키우지 않음)
    'local_min736': {'Det.limit_type': 'min', 'Det.limit_side_len': 736},  # rapidocr 기본 — 재현·기준선
    'local_min400': {'Det.limit_type': 'min', 'Det.limit_side_len': 400},  # 덜 키운다(중간안)
}


def server_params() -> dict:
    """serving/triton-ocr/.../model.py의 _rapidocr_params(CPU) — 서버 코드를 그대로 읽어 쓴다(복제하면 어긋난다)."""
    src = (ROOT / 'serving/triton-ocr/model_repository/ocr/1/model.py').read_text()
    start = src.index('def _rapidocr_params')
    end = src.index('\ndef ', start + 1)
    ns: dict = {}
    exec(src[start:end], ns)
    return ns['_rapidocr_params']('korean', use_cuda=False, text_score=0.5)


def install(name: str) -> None:
    import docling.datamodel.pipeline_options as PO

    import rag.chunking as C

    if not hasattr(PO, '_scan224_orig'):
        PO._scan224_orig = PO.KserveV2OcrOptions
    PO.KserveV2OcrOptions = PO._scan224_orig
    override = VARIANTS[name]
    if override is None:
        PO.KserveV2OcrOptions = PO._scan224_orig
    else:
        params = {**server_params(), **override}

        # OCR 엔진 등록부를 **바꾸기 전에** 만든다 — 등록부가 엔진마다 PO의 옵션 클래스를 읽어 짝짓기 때문에, 바꾼 뒤에
        # 만들면 Kserve 엔진이 이 대역과 짝지어진다('function' has no attribute 'kind' 같은 오류).
        from docling.models.factories import get_ocr_factory

        get_ocr_factory(False), get_ocr_factory(True)

        def _local(**kw: Any):
            return PO.RapidOcrOptions(
                lang=kw.get('lang', ['korean']),
                scale=kw.get('scale', 3.0),
                text_score=0.5,
                use_cls=False,
                rapidocr_params=params,
            )

        PO.KserveV2OcrOptions = _local
    C.convert_process.docling_convert = lambda p, **kw: C.convert_process.convert_local(p, **kw)
    C.document_converter._converter = None  # 다음 변환에서 새 옵션으로 변환기를 다시 만든다
