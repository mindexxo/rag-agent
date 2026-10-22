"""#244 배치 유형별 읽기 순서 측정 도구(eval/pdf_layout) — 명세·채점 규칙과, 합성 PDF 13종이 의도한 배치로 그려지는지.

채점기는 측정 중 네 번 틀렸다 — 같은 표식 중복(거짓 섞임) / "연속 구간" 판정이 표·헤딩으로 빠진 문단을 빈틈으로 오인 / 표식 순서만 봐서
줄 단위 단 섞임을 1/4로 과소 집계 / "연속 문자열" 판정이 단 끝 분리(split)를 섞임(mixed)과 혼동. 그 넷을 여기서 고정한다.
docling 실변환은 eval/results/pdf_layout/ 산출물과 리포트가 맡는다(모델 로딩이 필요해 테스트에선 안 돌린다).
"""
from pathlib import Path

import pytest

from eval.pdf_layout.build import build_all
from eval.pdf_layout.run import chunk_mixing, order_score, paragraph_fate, tokens_in
from eval.pdf_layout.specs import SPECS, expected_order


def test_명세의_표식은_빈틈_없는_연번():
    # 중복·건너뜀이 있으면 순서 대조가 거짓 섞임을 낸다(첫 측정의 L06 FAQ)
    for s in SPECS:
        nums = [int(t[1:]) for t in expected_order(s)]
        assert nums == list(range(1, len(nums) + 1)), s['id']


def test_배치_유형은_열셋이고_2단_간격_변형이_다섯이다():
    ids = [s['id'] for s in SPECS]
    assert len(ids) == 13 and sum(i.startswith('L01_two_col') for i in ids) == 5


def test_순서_점수():
    e = ['P01', 'P02', 'P03', 'P04']
    assert order_score(e, ['P01', 'P02', 'P03', 'P04']) == {'inversions': 0, 'displaced': 0, 'lost': [], 'dup': 0, 'n': 4}
    assert order_score(e, ['P03', 'P04', 'P01', 'P02'])['displaced'] == 4             # 블록이 통째로 밀림 — 인접 역전은 1뿐이라 따로 센다
    assert order_score(e, ['P01', 'P03', 'P02', 'P04'])['inversions'] == 1          # 이웃 쌍 하나 뒤집힘
    assert order_score(e, ['P01', 'P02', 'P04'])['lost'] == ['P03']
    assert order_score(e, ['P01', 'P02', 'P02', 'P03', 'P04'])['dup'] == 1
    assert order_score(e, ['P01', 'X9', 'P02'])['inversions'] == 0                  # 정답에 없는 표식은 무시


def test_조각_섞임은_내림만_본다():
    e = [f'P{i:02d}' for i in range(1, 12)]
    assert chunk_mixing(e, [['P02', 'P03', 'P06']])['mixed'] == []                  # 빈틈(표·헤딩으로 빠진 문단)은 섞임이 아니다
    assert chunk_mixing(e, [['P07', 'P10', 'P08', 'P09']])['mixed'][0]['chunk'] == 0   # 오른쪽 단이 끼어듦(좁은 2단 실측 모양)
    assert chunk_mixing(e, [['P08', 'P11', 'P04']])['mixed']                          # 표 위 단이 뒤로 밀림(2단+표 실측 모양)
    r = chunk_mixing(e, [[], ['P01'], ['Z']])
    assert r['chunks_checked'] == 1 and r['mixed'] == []                               # 표식 없는 조각은 제외


def test_문단_보존_세_갈래():
    # intact: 한 요소 안 / split: 요소 경계에서만 끊김(이어 붙이면 원문 — 넓은 2단 단 끝 모양) / mixed: 다른 줄이 끼거나 줄이 사라짐(좁은 2단)
    bodies = {'P1': '가나다라마바사', 'P2': '아자차카타파하', 'P3': '거너더러머버서'}
    texts = ['[P1] 가나다라', '마바사 [P2] 아자차', '카타파하', '[P3] 거너더 XX 러머버서']
    assert paragraph_fate(bodies, texts) == {'P1': 'split', 'P2': 'split', 'P3': 'mixed'}
    assert paragraph_fate({'P1': '가나다'}, ['[P1] 가나다 끝'])['P1'] == 'intact'
    assert paragraph_fate({'P1': '가나다'}, ['[P1] 가 나\n다'])['P1'] == 'intact'       # 공백·줄바꿈은 무시


def test_표식_추출():
    assert tokens_in('[P06]\xa02.\xa0절 [P07]\xa0항목') == ['P06', 'P07'] and tokens_in('') == []


@pytest.fixture(scope='module')
def pdfs(tmp_path_factory):
    return {p.stem: p for p in build_all(tmp_path_factory.mktemp('layout'))}


def test_합성_PDF의_쪽_수와_방향이_명세대로(pdfs):
    import pdfplumber
    want = {'L01_two_col_narrow': [(595, 842)] * 2, 'L02_three_col': [(842, 595)], 'L04_mixed_orient': [(595, 842), (842, 595), (595, 842)],
            'L05_two_col_table': [(595, 842)], 'L07_header_footer': [(595, 842)]}
    for sid, sizes in want.items():
        with pdfplumber.open(pdfs[sid]) as pdf:
            assert [(round(p.width), round(p.height)) for p in pdf.pages] == sizes, sid


def test_다단_PDF는_단마다_글이_나뉘어_있다(pdfs):
    # 좁은 2단 1쪽: 왼쪽 단 문단(P02~)과 오른쪽 단 문단(P07~)이 각각 쪽 절반 안에 있어야 "2단"이다 — 겹치면 배치가 틀린 것(L05 첫 생성)
    import pdfplumber
    with pdfplumber.open(pdfs['L01_two_col_narrow']) as pdf:
        page = pdf.pages[0]
        words = page.extract_words()
        x_of = {w['text'][:5]: w['x0'] for w in words if w['text'].startswith('[P')}
        assert x_of['[P02]'] < page.width / 2 < x_of['[P07]']


def test_머리글_PDF의_머리글은_본문_프레임_밖(pdfs):
    import pdfplumber
    with pdfplumber.open(pdfs['L07_header_footer']) as pdf:
        page = pdf.pages[0]
        head = next(w for w in page.extract_words() if w['text'].startswith('사내'))
        body = next(w for w in page.extract_words() if w['text'].startswith('[P01]'))
        assert head['top'] < body['top'] and head['top'] < 40
