"""실문서 그림 전수 추출 (#247 결정 1·5) — 문서를 운영과 같은 docling 경로로 다시 변환해 그림마다 원본 크롭·캡션·위치를 뽑고,
그 캡션이 **지금 색인(OpenSearch)에 그대로 있는지** 대조한다.

다시 변환하면 VLM이 캡션을 새로 만든다. temperature 0이라 대개 같지만 보장은 아니다(10/17 같은 코드 재색인 2회가 청크 444·443).
그래서 판정 대상(새 캡션)과 사용자가 받는 캡션(색인)이 같은지를 `in_index`로 남기고, 다르면 색인 캡션도 따로 판정한다(결정 5).

원본 크롭: DOCX·XLSX는 docling `get_image`가 파일에 든 원본을 준다. PDF는 None이다 — 운영 파이프라인이 쪽 이미지를 보관하지
않는다(generate_page_images 꺼짐, 실측). 그래서 PDF는 pypdfium2로 쪽을 그려 위치(bbox)대로 자른다. 배수는 VLM에 보내는
해상도와 같은 settings.vlm_caption_scale — 판정자가 VLM과 같은 해상도의 그림을 본다.

캡션이 없는 그림도 목록에 넣는다(이슈 최신화 10/18): 면적 하한(settings.vlm_caption_area_threshold — 결정 17 전 docling 기본 0.05) 미만은
캡션 대상이 아니라 글자가 있어도 색인에 0이다. 안 보이는 누락이라 판정 대상에 넣는다(rubric: 캡션 없음 = 글자 전부 누락).
"""
import hashlib
import io
import re
from dataclasses import asdict, dataclass
from pathlib import Path

_WS = re.compile(r'\s+')
PLACEHOLDER = '<!-- image -->'      # rag.chunking._PICTURE_PLACEHOLDER — 캡션 대상에서 빠진 그림이 청크에 남기는 표시


def norm(text: str) -> str:
    """대조용 정규화 — 공백·표 구분자를 지운다(OCR 띄어쓰기 소실·표 렌더 차이를 무시, #253)."""
    return _WS.sub('', (text or '').replace('|', ''))


IN_INDEX_RATIO = 0.95


def index_line_ratio(caption: str, index_text: str) -> float:
    """캡션 줄(정규화 4자 이상) 중 색인 글(정규화)에 있는 몫. 캡션 전체를 한 문자열로 찾으면 안 된다 — 긴 캡션(화면 캡처)은
    분할기가 두 청크에 나눠 담고 경계에 겹침(50토큰)이 생겨, 청크를 이은 글에 캡션이 연속으로 없다(타임레포트 3장 실측 10/18 —
    내용은 전부 색인에 있었다)."""
    lines = [n for n in (norm(x) for x in (caption or '').split('\n')) if len(n) >= 4]
    return sum(1 for n in lines if n in index_text) / len(lines) if lines else 1.0


@dataclass
class PictureRecord:
    picture_id: str            # 문서 id + 순번 — 리뷰 HTML·사람 판정 JSON의 키
    document_id: int
    filename: str
    fmt: str                   # pdf | docx | xlsx
    index: int                 # 문서 안 그림 순서(본문 층 iterate_items 순서)
    page: int | None
    area_ratio: float | None   # 쪽 면적 대비(PDF만)
    in_table: bool             # 표 칸 안 그림(DOCX·XLSX — #220)
    captioned: bool            # VLM 캡션이 붙었나
    skip_reason: str           # 캡션이 없는 이유: '면적 하한' | 'XLSX 하한(#249)' | ''
    caption: str
    in_index: bool | None      # 새 캡션 줄의 95% 이상이 색인 청크에 있나(index_line_ratio, 캡션 없으면 None)
    crop: str                  # 크롭 PNG 파일명(out/crops/ 아래)
    crop_size: tuple[int, int] | None


def area_threshold() -> float:
    """docling이 캡션을 요청하는 면적 하한 — 운영 설정값(settings.vlm_caption_area_threshold, #247 결정 17 전에는 docling 기본 0.05)."""
    from config import settings
    return settings.vlm_caption_area_threshold


def _pdf_crop(pdf_path: Path, page_no: int, bbox, page_h: float, scale: float):
    import pypdfium2
    pdf = pypdfium2.PdfDocument(str(pdf_path))
    try:
        image = pdf[page_no - 1].render(scale=scale).to_pil()
    finally:
        pdf.close()
    tl = bbox.to_top_left_origin(page_height=page_h)
    return image.crop((int(tl.l * scale), int(tl.t * scale), int(tl.r * scale), int(tl.b * scale)))


def _in_table(item, doc) -> bool:
    from docling_core.types.doc import TableItem
    parent = item.parent.resolve(doc) if item.parent else None
    while parent is not None:
        if isinstance(parent, TableItem):
            return True
        parent = parent.parent.resolve(doc) if getattr(parent, 'parent', None) else None
    return False


def extract_document(document_id: int, filename: str, path: Path, chunk_texts: list[str], crops_dir: Path,
                     convert=None) -> list[PictureRecord]:
    """문서 하나 → 그림 기록들. convert는 테스트 주입용(기본: 운영 docling_convert — 변환 자식 프로세스, #223)."""
    from config import settings
    from docling_core.types.doc import PictureItem
    from rag import chunking as ch

    fmt = path.suffix.lower().lstrip('.')
    convert = convert or ch.docling_convert
    doc = convert(path, **({'max_num_pages': 100} if fmt == 'pdf' else {}))
    index_text = norm('\n'.join(chunk_texts))
    threshold = area_threshold()
    out: list[PictureRecord] = []
    for item, _ in doc.iterate_items():
        if not isinstance(item, PictureItem):
            continue
        i = len(out)
        prov = item.prov[0] if item.prov else None
        page = prov.page_no if prov else None
        ratio = None
        image = None
        if prov is not None and page in doc.pages:
            size = doc.pages[page].size
            ratio = prov.bbox.area() / (size.width * size.height) if size.width * size.height else None
            if fmt == 'pdf':
                image = _pdf_crop(path, page, prov.bbox, size.height, settings.vlm_caption_scale)
        if image is None:
            image = item.get_image(doc)
        caption = ch._docling_picture_caption(item)
        skip = ''
        if not caption:
            skip = 'XLSX 하한(#249)' if fmt == 'xlsx' else ('면적 하한' if ratio is not None and ratio < threshold else '기타')
        name = f'{document_id}_{i:03d}.png'
        if image is not None:
            buf = io.BytesIO()
            image.convert('RGB').save(buf, format='PNG')
            (crops_dir / name).write_bytes(buf.getvalue())
        out.append(PictureRecord(
            picture_id=f'{document_id}-{i:03d}', document_id=document_id, filename=filename, fmt=fmt, index=i,
            page=page, area_ratio=round(ratio, 5) if ratio is not None else None, in_table=_in_table(item, doc),
            captioned=bool(caption), skip_reason=skip, caption=caption,
            in_index=(index_line_ratio(caption, index_text) >= IN_INDEX_RATIO) if caption else None,
            crop=name if image is not None else '', crop_size=image.size if image is not None else None))
    return out


def record_dict(r: PictureRecord) -> dict:
    return asdict(r)


def crop_digest(path: Path) -> str:
    """같은 그림이 여러 문서에 반복될 때(로고 등) 묶어 보기 위한 지문."""
    return hashlib.sha1(path.read_bytes()).hexdigest()[:12]
