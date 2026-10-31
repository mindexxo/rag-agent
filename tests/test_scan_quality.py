"""#224 스캔 품질 도구(eval/scan_quality) — 대조 지표와 합성 열화가 의도대로 세는지. 변환·OCR 없이 도는 순수 부분만."""

from types import SimpleNamespace

import numpy as np
from PIL import Image

from eval.scan_quality import compare as cmp
from eval.scan_quality.scans import SPECS, ScanSpec, degrade


def _c(text, *heads):
    return SimpleNamespace(text=text, heading_path=list(heads))


def test_띄어쓰기_유지율은_원본_공백_중_남은_비율이다():
    assert cmp._space_kept('가나 다라 마바', '가나 다라 마바') == 1.0
    assert cmp._space_kept('가나 다라 마바', '가나다라 마바') == 0.5
    assert cmp._space_kept('가나 다라', '가나다라') == 0.0
    assert cmp._space_kept('가나다라', '가 나다라') == 1.0  # 원본에 공백이 없으면 셀 것이 없다


def test_제목이_깨지면_title_ok가_거짓이고_깨진_헤딩이_보고된다():
    digital = [_c('본 규정은 기부금영수증의 발급 기준을 정한다.', '기부금영수증 발급 규정', '1. 목적')]
    scan = [_c('본 규정은 기부금영수증의 발급기준을정한다.', 'D 금영 0 밤 D 정', '1.목적')]
    m = cmp.measure(digital, scan)
    assert not m['title_ok'] and m['bad_headings'] == ['D 금영 0 밤 D 정']
    assert m['heading_recall'] == 0.5  # '1. 목적'은 공백 무시로 일치
    assert m['sent_exact'] == 0.0 and m['sent_nospace'] == 1.0  # 띄어쓰기만 틀린 문장은 nospace에서 산다


def test_가운뎃점_한_글자_차이는_제목_깨짐이_아니다():
    m = cmp.measure(
        [_c('본문 문장은 여덟 글자 이상이다.', '선물세트 등급별 가격ㆍ할인표 (HR-CS-013)')],
        [_c('본문 문장은 여덟 글자 이상이다.', '선물세트 등급별 가격·할인표 (HR-CS-013)')],
    )
    assert m['title_ok'] and m['title_sim'] >= 0.95


def test_표_행은_문장_대조에서_뺀다():
    assert cmp._sentences([_c('| 등급 | 금액 |\n| --- | --- |\n안내 문장은 충분히 길어야 센다.')]) == [
        '안내 문장은 충분히 길어야 센다.'
    ]


def test_열화는_결정적이고_조건마다_다른_이미지를_만든다():
    img = Image.new('RGB', (200, 100), 'white')
    for x in range(20, 180, 10):
        img.paste((0, 0, 0), (x, 40, x + 4, 60))
    rng = lambda: np.random.default_rng(7)  # noqa: E731
    noisy = ScanSpec('t', 'x.pdf', 150, noise=18, jpeg=35, uneven=1.0, rotate=1.5)
    a, b = degrade(img, noisy, rng()), degrade(img, noisy, rng())
    assert np.array_equal(np.asarray(a), np.asarray(b))  # 같은 시드면 같은 결과(재현 가능한 정본)
    clean = degrade(img, ScanSpec('t', 'x.pdf', 150), rng())
    assert not np.array_equal(np.asarray(a), np.asarray(clean))


def test_합성_스캔_대상은_테넌트가_서로_다르다():
    # 검색 영향 측정(retrieval swap)이 문서를 테넌트별로 교체한다 — 겹치면 서로의 순위에 섞인다
    assert len({s.tenant for s in SPECS}) == len(SPECS) == 5
    assert all(s.src.exists() for s in SPECS)
