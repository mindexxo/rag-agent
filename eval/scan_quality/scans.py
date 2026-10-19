"""합성 스캔 PDF 생성(#224) — 디지털 PDF를 쪽 이미지로 렌더하고 스캔처럼 열화시켜 이미지 전용 PDF로 묶는다.

실제 스캔본이 없어 만든 대역이다(결론은 '합성 조건 기준'). 열화는 결정적이다(난수 시드 고정) — 같은 명령이면 같은 PDF.
조건 5개는 실제 스캔에서 흔한 것을 하나씩 겨눈다: 깨끗한 저해상(#220 재현) · 기울기 · 복사기 잡음·번짐 · 저해상도 · 조명 얼룩+JPEG.

사용: python -m eval.scan_quality.scans   → eval/results/scan224/<원본 stem>.scan.pdf
"""
import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'eval' / 'results' / 'scan224'


@dataclass(frozen=True)
class ScanSpec:
    tenant: str
    filename: str
    dpi: int
    rotate: float = 0.0     # 도(반시계)
    noise: float = 0.0      # 가우시안 잡음 표준편차(0~255) — 0이 아니면 점 잡음·번짐도 더한다
    jpeg: int = 0           # JPEG 품질(0 = 압축 안 함)
    uneven: float = 0.0     # 조명 얼룩 세기(0 = 없음)
    note: str = ''

    @property
    def src(self) -> Path:
        return ROOT / 'sample_docs' / 'corpus_v2' / self.tenant / self.filename

    @property
    def out(self) -> Path:
        return OUT / f'{Path(self.filename).stem}.scan.pdf'


SPECS = [
    ScanSpec('goodpeople', 'goodpeople_03_기부금영수증.pdf', 150, note='깨끗한 150dpi — #220 샘플 재현'),
    ScanSpec('homeplus', 'homeplus_05_멤버십포인트.pdf', 200, rotate=1.5, note='기울기'),
    ScanSpec('harim', 'harim_13_선물세트등급별가격할인표.pdf', 300, noise=18, note='복사기 잡음·번짐'),
    ScanSpec('adererror', 'adererror_03_사이즈교환가이드.pdf', 100, note='저해상도'),
    ScanSpec('aromanica', 'aromanica_03_피부트러블대응.pdf', 200, jpeg=35, uneven=1.0, note='조명 얼룩 + JPEG'),
]


def degrade(img: Image.Image, spec: ScanSpec, rng: np.random.Generator) -> Image.Image:
    im = img.convert('L')
    if spec.rotate:
        im = im.rotate(spec.rotate, resample=Image.BICUBIC, expand=False, fillcolor=255)
    a = np.asarray(im).astype(np.float32)
    if spec.uneven:                                   # 왼쪽 위가 최대 60 어둡고 오른쪽 아래로 갈수록 밝다
        h, w = a.shape
        g = (np.linspace(0, 1, w)[None, :] + np.linspace(0, 1, h)[:, None]) / 2
        a = a - 60 * spec.uneven * (1 - g)
    if spec.noise:
        a = a + rng.normal(0, spec.noise, a.shape)
        a = np.where(rng.random(a.shape) < spec.noise / 4000, 0, a)
    im = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    if spec.noise:
        im = im.filter(ImageFilter.GaussianBlur(0.6))
    if spec.jpeg:
        buf = io.BytesIO()
        im.save(buf, 'JPEG', quality=spec.jpeg)
        im = Image.open(io.BytesIO(buf.getvalue()))
    return im.convert('RGB')


def build(spec: ScanSpec) -> Path:
    import pypdfium2
    rng = np.random.default_rng(7)
    pages = [degrade(p.render(scale=spec.dpi / 72).to_pil(), spec, rng) for p in pypdfium2.PdfDocument(str(spec.src))]
    spec.out.parent.mkdir(parents=True, exist_ok=True)
    pages[0].save(spec.out, 'PDF', resolution=spec.dpi, save_all=True, append_images=pages[1:])
    return spec.out


if __name__ == '__main__':
    for s in SPECS:
        print(build(s), s.note)
