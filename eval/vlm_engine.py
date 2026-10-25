"""측정용 VLM 변환 엔진(#296) — docling VlmPipeline + ApiVlmOptions로 **쪽 전체**를 VLM에 넣어 마크다운으로 받고, 그걸 DoclingDocument로
만든다. 상용 전환 a안("캡션과 같은 VLM이 OCR까지")의 운영 모양 그대로라, 하네스가 이 엔진으로 돌린 결과 = a안을 채택했을 때의 색인이다.

운영 코드는 건드리지 않는다 — rag/chunking의 `_convert_local`만 이 프로세스 안에서 바꿔 끼운다(eval/scan_quality/ocr_variant와 같은 패턴).
VLM 주소·모델·키·온도는 캡션 설정(VLM_CAPTION_*)을 그대로 쓴다 — 벤더 전환은 .env 세 줄(#290 "주소가 곧 벤더").

고정값(비교 조건 — 보고에 적을 것): 프롬프트 PROMPT(기본 V2), scale 2.0(72dpi 기준 배수 — docling 기본값. 캡션의 4.0보다 낮은 건 쪽
전체라 픽셀이 많아서다. **scale을 바꿔 비교한 실험은 하지 않았다**), 출력 상한 4,000(표 한 쪽이 캡션 상한 1,500을 넘을 수 있어서),
응답 형식 markdown. 사내 Qwen2.5-VL 7B 원형 실측(10/24): 표 이미지 1쪽 3.4초, 표 2개 TableItem 추출.

측정 결과(10/25, 이슈 #296 코멘트가 정본): gpt-5-mini 표 22축 셀 94.2%·숫자 오독 6(Triton 80.9%·26), 스캔 5건 헤딩 0.93·띄어쓰기
0.96(Triton 0.80·0.68). 사내 7B는 숫자 오독 51로 Triton보다 나쁘다. Gemini 2.5 Flash는 무료 한도(하루 20건)로 표 12축만.

사용: install() 뒤 rag.chunking.chunk_file / _convert_local 호출. uninstall()로 원복(스캔 하네스의 디지털 기준 변환이 섞이지 않게).
"""
import os
import time
from urllib.parse import urlsplit

import rag.chunking as C
from config import settings

PROMPT_V1 = ("이 쪽을 마크다운으로 옮겨 적어라. 표는 마크다운 표(| 칸 | 칸 |)로 — 병합된 칸은 그 값을 해당 칸마다 반복해 적고, 빈 칸은 비워 둔다. "
             "제목은 #로, 본문은 문단으로. 보이는 글자를 그대로 쓰고 없는 내용을 지어내지 마라. 설명·주석 없이 마크다운만 출력.")
# V2(#296): 절 제목 표시를 VLM 판단에 맡기지 않고 규칙으로 못박는다. V1에서 gpt-5-mini가 1쪽 절 제목('2. 증빙 요건')을 평문으로 써서
# docling 마크다운 해석기가 앞 번호 목록의 6번째 항목으로 합치고 번호를 다시 매겼다('2.'→'6.' — 글자 내용이 바뀐다, 실측 10/25).
# '## '를 붙이면 목록으로 합쳐지지 않는다. 문서 제목 '#'는 1쪽 맨 위 하나뿐 — 쪽 단위 호출이라 쪽마다 '#'가 생기던 것도 막는다.
PROMPT_V2 = ("이 쪽을 마크다운으로 옮겨 적어라. 규칙:\n"
             "1. 문서 제목(문서 맨 앞의 큰 제목)만 '# '로 쓴다. 쪽 중간이나 둘째 쪽 이후의 제목은 문서 제목이 아니다.\n"
             "2. 절·항 제목은 번호가 있든 없든 반드시 '## '로 시작한다. 예: '## 1. 접수 원칙', '## 2.1 사진 증빙'. 번호는 원문 그대로 쓴다.\n"
             "3. 목록은 원문 표시 그대로 — 글머리표는 '- ', 번호 목록은 원문 번호 그대로.\n"
             "4. 표는 마크다운 표(| 칸 | 칸 |)로. 병합된 칸은 그 값을 해당 칸마다 반복해 적고, 빈 칸은 비워 둔다.\n"
             "5. 보이는 글자를 그대로 쓰고 없는 내용을 지어내지 마라. 설명·주석 없이 마크다운만 출력.")
PROMPT = PROMPT_V2 if os.getenv('VLM_PROMPT', 'v2') == 'v2' else PROMPT_V1
SCALE = 2.0
MAX_TOKENS = 4000
TIMEOUT = 180.0

_orig_convert_local = C._convert_local
_orig_docling_convert = C.docling_convert
_converter = None


def label() -> str:
    """결과 디렉터리·행에 적는 엔진 이름 — 모델명 기준(주소는 키 유출 없이 호스트만)."""
    host = urlsplit(settings.vlm_caption_url).hostname or 'unknown'   # user:key@host 형태여도 키가 안 섞이게
    return f"vlm_{settings.vlm_caption_model}@{host}".replace('/', '_').replace(':', '_')


def build_converter():
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import VlmPipelineOptions
    from docling.datamodel.pipeline_options_vlm_model import ApiVlmOptions, ResponseFormat
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.pipeline.vlm_pipeline import VlmPipeline
    opts = ApiVlmOptions(
        url=settings.vlm_caption_url, headers=C._caption_headers(), prompt=PROMPT, response_format=ResponseFormat.MARKDOWN,
        scale=SCALE, timeout=TIMEOUT, concurrency=1,
        # 온도는 **옵션 필드**로 줘야 한다 — docling ApiVlmModel이 params 위에 vlm_options.temperature를 덮어쓴다(실측: params에만
        # 1.0을 주니 gpt-5가 0.0 거부 400). 인자 이름은 캡션과 같은 새 표기(#290 결정 3 — gpt-5가 max_tokens 거부)
        temperature=settings.vlm_caption_temperature,
        params={'model': settings.vlm_caption_model, 'max_completion_tokens': MAX_TOKENS})
    po = VlmPipelineOptions(enable_remote_services=True, vlm_options=opts)
    return DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_cls=VlmPipeline, pipeline_options=po)})


# 벤더 한도 대응(측정 전용). docling은 429·5xx를 0.1초 배수로 5회만 재시도하고 실패하면 **빈 결과를 돌려준다**(예외 없음 — 그 쪽은 "미검출"로
# 채점돼 한도 초과가 품질 저하로 둔갑한다). 그래서 요청 사이 최소 간격(VLM_MIN_INTERVAL초, Gemini 무료 티어 10 RPM → 7)과 빈 응답
# 재시도(30초 쉬고 최대 4회)를 요청 함수 앞뒤에 끼운다. 운영 캡션 경로엔 없는 장치다 — 운영은 유료 티어 한도로 설계해야 한다.
_MIN_INTERVAL = float(os.getenv('VLM_MIN_INTERVAL', '0'))
_last_call = [0.0]
_retries = [0]
RAW: list[str] = []


def _paced(fn):
    def wrapper(*a, **kw):
        for attempt in range(5):
            wait = _MIN_INTERVAL - (time.monotonic() - _last_call[0])
            if wait > 0:
                time.sleep(wait)
            _last_call[0] = time.monotonic()
            r = fn(*a, **kw)
            if getattr(r, 'text', None):
                RAW.append(r.text)            # VLM 원래 응답(docling 해석 전) — 번호 바뀜이 누구 탓인지 가르는 데 쓴다
                return r
            _retries[0] += 1
            time.sleep(30)
        # 다섯 번 다 비었다 = 한도 소진(무료 티어 일일 20건 등). 한계: 정당하게 빈 쪽(백지)도 같은 판정이 된다 — 현재 측정 문서엔 백지 쪽이 없다.
        # 빈 결과로 두면 그 쪽이 '미검출 0점'으로 채점돼 한도 초과가 품질
        # 저하로 둔갑한다(10/24 실측). 측정을 통째로 멈춘다 — 남은 문서는 --only로 이어 돌린다.
        print(f'[vlm_engine] 빈 응답 5회 연속 — 한도 소진으로 보고 측정 중단', flush=True)
        os._exit(3)
    return wrapper


def _patch_requests():
    import docling.models.vlm_pipeline_models.api_vlm_model as M
    if not getattr(M, '_vlm296_patched', False):
        M.api_image_request = _paced(M.api_image_request)
        M.api_image_request_streaming = _paced(M.api_image_request_streaming)
        M._vlm296_patched = True


# 헤딩 조립 보정(#296) — VLM이 '#'로 쓴 줄은 docling에서 title 라벨이 되는데 PDF 청킹(_sections_from_docling_elements)은 section_header만
# 헤딩으로 본다 → 문서 제목이 본문 줄로 떨어졌다(채점 '제목 ✗'의 원인, 실측 10/25). title → section_header로 바꾸면 청킹이 첫 헤딩을
# 문서 제목, 나머지를 절로 층을 다시 만든다.
# 기각한 보정: '번호 헤딩 꼴 글 줄을 헤딩으로 올리기' — gpt-5-mini 33줄 중 확인 항목 목록 4줄 이상을 헤딩으로 올렸고(오탐), 근본 원인
# (VLM이 절 제목을 평문으로 써서 docling이 앞 번호 목록에 합치고 번호까지 다시 매김 '2.'→'6.')을 못 막는다. 원인은 PROMPT_V2로 막았다
# (아로마니카 스캔본 헤딩 13/13 원본과 일치, 번호 보존).


def _fix_headings(doc):
    from docling_core.types.doc import DocItemLabel
    for item, _ in doc.iterate_items():
        label = item.label.value if hasattr(item.label, 'value') else str(item.label)
        if label == 'title':
            item.label = DocItemLabel.SECTION_HEADER
    return doc


def _convert_vlm(file_path, **kwargs):
    """PDF(표 이미지·스캔)만 VLM으로. 다른 형식은 원래 변환기 — 측정 대상이 아니다."""
    global _converter
    if str(file_path).lower().endswith('.pdf'):
        if _converter is None:
            _patch_requests()
            _converter = build_converter()
        from docling.datamodel.base_models import ConversionStatus
        result = _converter.convert(str(file_path), raises_on_error=True, **kwargs)
        if result.status != ConversionStatus.SUCCESS or result.errors:
            raise RuntimeError(f'VLM 변환 불완전 — status={result.status.value}, errors={len(result.errors)}')
        return _fix_headings(result.document) if os.getenv('VLM_FIX_HEADINGS', '1') == '1' else result.document
    return _orig_convert_local(file_path, **kwargs)


def install() -> None:
    C._convert_local = _convert_vlm
    C.docling_convert = lambda p, **kw: C._convert_local(p, **kw)     # 자식 프로세스 대신 이 프로세스에서(설정 교체가 먹게)


def uninstall() -> None:
    global _converter
    if _retries[0]:
        print(f'[vlm_engine] 빈 응답 재시도 {_retries[0]}회', flush=True)
    _retries[0] = 0
    RAW.clear()                      # 모드 사이에 누적되지 않게(스캔 대조는 한 프로세스에서 여러 모드를 돈다)
    C._convert_local = _orig_convert_local
    C.docling_convert = _orig_docling_convert
    _converter = None
