"""docling 변환기 조립과 변환 — 프로세스당 1회 변환기(PDF·DOCX·XLSX)·캡션 가드(그림 장수·캡션 실패)·불완전 결과 가드.

변환 프로세스(_ConvertProcess) 안에서 불린다. docling은 톱레벨에 들이지 않는다(함수 안 지연 import).
"""

import threading
from pathlib import Path
from typing import TYPE_CHECKING

from config import settings
from rag.chunking.errors import CaptionFailedError, PictureLimitError

if TYPE_CHECKING:
    from docling.document_converter import DocumentConverter
    from docling_core.types.doc import DoclingDocument

# Triton OCR 상수 — 서버(serving/triton-ocr/model_repository/ocr/1/model.py)와 짝이라 설정으로 빼지 않는다.
OCR_LANG = 'korean'  # docling 기본은 ['chinese'] — 한글 표를 틀린 값으로 채운다
OCR_SCALE = 3.0  # 크롭 렌더 배수(72dpi 기준). KserveV2OcrOptions 기본은 2.0이라 인프로세스 RapidOCR(3.0)과 맞추려면 명시해야 한다

# 그림 캡션 프롬프트 — 모든 그림에 하나. 유형별 지시는 '…라면' 조건부로만 둔다(형태를 전제한 문구는 다른 유형에 해롭다).
# 박스 아래 설명을 박스에 귀속시키라는 지시는 넣지 않는다 — 귀속을 한 칸씩 밀어 확신 있는 오답을 낸다.
PICTURE_CAPTION_PROMPT = (
    '이미지의 내용을 한국어로 옮겨라.\n'
    '- 이미지 안의 글자를 하나도 빠뜨리지 말고 원문 그대로 옮긴다. 고쳐 쓰거나 요약하지 않는다.\n'
    '- 요소들 사이에 관계가 있으면 그 관계를 함께 적는다.\n'
    '- 관계가 분명하지 않으면 보이는 위치대로 나열만 한다. 관계가 없다는 말은 쓰지 않는다.\n'
    '- 흐름도·절차도라면 화살표 하나를 한 줄로 "<출발 박스 이름> → <도착 박스 이름>" 형식으로 적고, 화살표 위 글자(예/아니오 등)가 '
    '있으면 줄 끝에 괄호로 붙인다.\n'
    '- 조직도·계층도라면 위에서 아래로, 연결선 하나를 한 줄로 "<상위 이름> → <하위 이름>" 형식으로 적는다.\n'
    '- 차트라면 제목·축 이름·단위·범례를 먼저 적고, 값은 "가로축 항목 (계열): 값" 형식으로 항목마다 한 줄씩 적는다.\n'
    '- 이미지에 없는 내용은 쓰지 않는다.\n'
    '- 값이 비어 있는 칸·막대·점은 지어내지 말고 "값 없음"으로 적는다.\n'
    '- 머리말·맺음말·질문·제안 없이, 첫 줄부터 이미지 안의 글자로 시작한다.'
)
# 머리말·맺음말 금지는 금지할 문구를 인용하지 않고 적는다 — 인용하면 오히려 그 문구를 유도한다.


DOCLING_LOCK = threading.Lock()
_docling_converter = None


class PictureGuard:
    """docling 캡션 모델을 감싸 그림 장수 상한과 캡션 실패를 건다 — docling에는 둘 다 옵션이 없다.

    파이프라인 enrichment_pipe의 캡션 모델을 이 대리 객체로 바꿔 끼우고(_guard_picture_models), 나머지 속성은 원본에 넘긴다.
    - 장수: 문서의 첫 배치에서 캡션 대상(면적 하한을 넘는 그림)을 한 번 센다 — 넘으면 캡션을 하나도 붙이기 전에 끝난다.
    - 실패: 빈 캡션이면 멈춘다. 동시 N이면 그 배치에서 이미 보낸 요청은 끝까지 기다린다.
    문서 경계는 convert_with가 arm()으로 알린다 — id(doc)로 가르면 해제된 문서의 id가 재사용될 수 있다.
    """

    def __init__(self, inner):
        self._inner = inner
        self._counted = False

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def arm(self) -> None:
        self._counted = False

    def __call__(self, doc, element_batch):
        if not self._counted:
            self._counted = True
            n = _captionable_count(doc, self._inner.options.picture_area_threshold)
            if n > settings.docling_max_pictures:
                raise PictureLimitError(
                    f'그림 {n}장 — 문서당 {settings.docling_max_pictures}장을 넘는 문서는 받지 않는다'
                )
        for item in self._inner(doc, element_batch):
            desc = item.meta.description if item.meta is not None else None
            if desc is not None and not (desc.text or '').strip():
                raise CaptionFailedError('그림 캡션 실패(VLM 응답 없음·오류) — 나중에 다시 시도한다')
            yield item


def _captionable_count(doc, area_threshold: float) -> int:
    """docling이 캡션을 요청할 그림 수 — PictureDescriptionBaseModel.__call__의 면적 판정을 그대로 따른다.

    순회도 docling과 같은 doc.iterate_items()다(doc.pictures는 머리글 등 furniture까지 담아 더 크다).
    """
    from docling_core.types.doc import PictureItem  # 지연 import — docling은 톱레벨에 안 들인다

    n = 0
    for item, _ in doc.iterate_items():
        if not isinstance(item, PictureItem):
            continue
        if item.prov:
            prov = item.prov[0]
            page = doc.pages.get(prov.page_no)
            if page is not None:
                page_area = page.size.width * page.size.height
                if page_area > 0 and prov.bbox.area() / page_area < area_threshold:
                    continue
        n += 1
    return n


_XLSX_PICTURE_AREA_THRESHOLD = 0.0  # 엑셀 그림 캡션 면적 하한 — 근거는 docling_runtime의 xlsx_opts 주석

_picture_guards: list[PictureGuard] = []


def _guard_picture_models(converter) -> None:
    """세 형식의 파이프라인을 미리 띄우고 캡션 모델을 PictureGuard로 바꿔 끼운다.

    initialized_pipelines는 DocumentConverter의 파이프라인 캐시다. XLSX는 캡션 면적 하한이 달라 DOCX와 파이프라인을
    공유하지 않으므로 형식마다 따로 감싼다. docling 업그레이드로 이 구조가 바뀌면 캡션 실패·장수 테스트가 먼저 깨진다.
    """
    from docling.datamodel.base_models import InputFormat
    from docling.models.picture_description_base_model import PictureDescriptionBaseModel

    for fmt in (InputFormat.PDF, InputFormat.DOCX, InputFormat.XLSX):
        converter.initialize_pipeline(fmt)
    for pipeline in converter.initialized_pipelines.values():
        for i, model in enumerate(pipeline.enrichment_pipe):
            if isinstance(model, PictureDescriptionBaseModel):
                guard = PictureGuard(model)
                pipeline.enrichment_pipe[i] = guard
                _picture_guards.append(guard)


def caption_headers() -> dict[str, str]:
    """캡션·OCR 요청에 붙일 HTTP 헤더 — 키가 있을 때만 Bearer(사내 VLM은 키가 없다)."""
    key = settings.vlm_caption_api_key
    return {'Authorization': f'Bearer {key}'} if key else {}


def docling_runtime() -> 'DocumentConverter':
    """프로세스당 1회: PDF·DOCX·XLSX 변환기(모델 로딩). 변환 프로세스(_ConvertProcess) 안에서 불린다.

    메모리: 모델이 상주하고 glibc가 해제분을 바로 반납하지 않아, 여러 문서를 돈 정상상태 RSS가 단일 문서 피크보다 높다 —
    컨테이너 메모리 상한(docker-compose.yml)은 단일 문서 피크로 정하면 안 된다.
    """
    global _docling_converter
    with DOCLING_LOCK:
        if _docling_converter is None:
            from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import (
                ConvertPipelineOptions,
                KserveV2OcrOptions,
                PdfPipelineOptions,
                PictureDescriptionApiOptions,
                TableFormerMode,
            )
            from docling.document_converter import (
                DocumentConverter,
                ExcelFormatOption,
                PdfFormatOption,
                WordFormatOption,
            )

            opts = PdfPipelineOptions(artifacts_path=settings.docling_artifacts_path)
            # 메모리 노브 — config.py 주석 참조. perf.page_batch_size는 쓰지 않는다: 기본 파이프라인이 그 값을 읽지 않는다.
            opts.layout_batch_size = settings.docling_layout_batch_size
            opts.queue_max_size = settings.docling_queue_max_size
            # 상용 OCR(OCR_PROVIDER=llm)이면 Triton을 안 부른다 — 사진 표·스캔 쪽은 '상용 OCR' 절이 LLM으로 읽는다.
            opts.do_ocr = settings.docling_do_ocr and settings.ocr_provider == 'triton'
            if opts.do_ocr:
                # 엔진은 원격 Triton. 기본 OcrMode는 비트맵 영역만 OCR하고 텍스트 레이어는 건드리지 않는다.
                # 요청 실패는 docling이 삼키고 result.errors에만 남기므로 convert_with가 errors를 보고 실패로 올린다.
                opts.ocr_options = KserveV2OcrOptions(
                    url=settings.ocr_kserve_url,
                    model_name=settings.ocr_kserve_model_name,
                    transport='http',
                    lang=[OCR_LANG],
                    scale=OCR_SCALE,
                    timeout=settings.ocr_kserve_timeout_seconds,
                )
            opts.do_table_structure = True
            opts.table_structure_options.mode = (
                TableFormerMode.FAST if settings.docling_table_mode == 'fast' else TableFormerMode.ACCURATE
            )
            opts.table_structure_options.do_cell_matching = settings.docling_do_cell_matching
            # opts.document_timeout은 걸지 않는다 — 문서 마감은 _ConvertProcess가 프로세스째 끊는다. docling의 것은
            # PDF 쪽 루프에서만 검사하고 PARTIAL_SUCCESS(→ 결정적 실패)로 끝나 "마감 초과 = 일시 실패"와 어긋난다.
            opts.accelerator_options = AcceleratorOptions(
                num_threads=settings.docling_num_threads, device=AcceleratorDevice(settings.docling_device)
            )
            # enable_remote_services를 안 켜면 do_picture_description이 설정 단계에서 OperationNotAllowed로 죽는다.
            caption_opts = PictureDescriptionApiOptions(
                url=settings.vlm_caption_url,
                params={
                    'model': settings.vlm_caption_model,
                    'temperature': settings.vlm_caption_temperature,  # params는 요청 body에 그대로 들어간다
                    # gpt-5 계열은 'max_tokens'를 400으로 거부한다 — 사내 vLLM·Gemini는 이 이름도 상한으로 받는다
                    'max_completion_tokens': settings.vlm_caption_max_tokens,
                },
                prompt=PICTURE_CAPTION_PROMPT,
                picture_area_threshold=settings.vlm_caption_area_threshold,
                scale=settings.vlm_caption_scale,
                headers=caption_headers(),
                timeout=settings.vlm_caption_timeout_seconds,
                concurrency=settings.vlm_caption_concurrency,
            )
            opts.enable_remote_services = True
            opts.do_picture_description = True
            opts.picture_description_options = caption_opts
            # DOCX·XLSX는 본문이 원래 텍스트라 레이아웃·OCR 없는 SimplePipeline. 캡션 설정만 PDF와 공유한다.
            office_opts = ConvertPipelineOptions(
                enable_remote_services=True, do_picture_description=True, picture_description_options=caption_opts
            )
            # XLSX만 면적 하한 0 — 엑셀은 쪽이 없어 docling이 시트를 칸 단위로 재고 그림을 크기와 무관하게 1×1칸으로 잡아,
            # 비율이 그림 크기가 아니라 시트 데이터 양을 잰다(큰 차트도 하한에 걸려 빠진다). 장수 상한도 이 하한으로 센다.
            # 네이티브 차트 객체는 그림으로 세지지만 이미지가 없어 캡션되지 않고 자리표시만 남는다(값은 시트 칸에 있다).
            xlsx_opts = ConvertPipelineOptions(
                enable_remote_services=True,
                do_picture_description=True,
                picture_description_options=caption_opts.model_copy(
                    update={'picture_area_threshold': _XLSX_PICTURE_AREA_THRESHOLD}
                ),
            )
            _docling_converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=opts),
                    InputFormat.DOCX: WordFormatOption(pipeline_options=office_opts),
                    InputFormat.XLSX: ExcelFormatOption(pipeline_options=xlsx_opts),
                }
            )
            _guard_picture_models(_docling_converter)
    return _docling_converter


def convert_with(converter: 'DocumentConverter', file_path: str | Path, **kwargs) -> 'DoclingDocument':
    """변환기 하나로 변환 + 불완전 결과 가드의 정의점.

    raises_on_error=True만으로는 부족하다 — docling은 PARTIAL_SUCCESS(원격 OCR 실패 등)에 예외를 안 던지므로 status와
    errors를 직접 보고 올린다. PictureGuard의 예외는 docling이 RuntimeError로 감싸므로 원래 예외로 풀어 다시 던진다
    (감싼 채면 outbox.is_transient가 결정적으로 본다). 상용 OCR은 쪽 묶음마다 부르므로 그림 장수 상한도 묶음마다 센다.
    """
    for guard in _picture_guards:
        guard.arm()
    try:
        result = converter.convert(str(file_path), raises_on_error=True, **kwargs)
    except Exception as e:
        cause = e.__cause__
        while cause is not None:
            if isinstance(cause, (PictureLimitError, CaptionFailedError)):
                raise cause from None
            cause = cause.__cause__
        raise
    from docling.datamodel.base_models import ConversionStatus  # 지연 import — docling은 톱레벨에 안 들인다

    if result.status != ConversionStatus.SUCCESS or result.errors:
        detail = '; '.join(
            f'{e.category.value}/{e.module_name}: {(e.error_message or "")[:100]}' for e in result.errors[:3]
        )
        raise RuntimeError(f'docling 변환 불완전 — status={result.status.value}, errors={len(result.errors)}: {detail}')
    return result.document
