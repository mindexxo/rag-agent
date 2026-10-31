"""#252 표 렌더 보강 문서 — 합성 3건(sample_docs/corpus_v2/*_15_*)을 실제로 변환해 골드 *_tb 문항의 정답 스니펫이 청크에 있는지.

실문서 102건 diff에서 근거가 없거나 1건뿐이던 결정(4 표 안의 표·8 조 헤딩 바로 아래 표·9 가로 병합·C 칸 안 줄바꿈·E 쪽 넘김 표·
F 엑셀 날짜)을 실제 파일 경로로 고정한다. #252 전 코드로는 DOCX·XLSX 6문항의 스니펫이 청크에 아예 없었다(빈 칸·깨진 행·
`00:00:00`, 2026-10-17 실측). 문서 생성의 정본은 sample_docs/corpus_v2/_src/build_table_docs.py.
"""
import json
from pathlib import Path

import pytest

from rag.chunking import chunk_file

pytest.importorskip('docling')

ROOT = Path(__file__).resolve().parent.parent
GOLD = [json.loads(line) for line in open(ROOT / 'eval' / 'gold_set_v2.jsonl') if '_tb0' in line]


def _doc(filename: str) -> Path:
    return ROOT / 'sample_docs' / 'corpus_v2' / filename.split('_')[0] / filename


@pytest.fixture(scope='module')
def chunks():
    out = {}
    for g in GOLD:
        name = g['expected_docs'][0]
        if name not in out:
            desc = name.rsplit('.', 1)[0].split('_', 2)[-1] if name.endswith('.xlsx') else ''   # 업로드와 같은 표 설명
            out[name] = chunk_file(_doc(name), description=desc)
    return out


def test_골드는_문서_셋_결정_여섯을_덮는다():
    assert len(GOLD) == 9 and {g['id'].split('_')[0] for g in GOLD} == {'goodpeople', 'summers', 'harim'}


@pytest.mark.parametrize('g', GOLD, ids=[g['id'] for g in GOLD])
def test_정답_스니펫이_청크에_있다(g, chunks):
    snippet = g['expected_chunks'][0]['snippet']
    assert any(snippet in c.text for c in chunks[g['expected_docs'][0]]), snippet


def test_PDF_조_헤딩_바로_아래_표는_헤딩과_같은_청크에_있고_식별자로_기록된다(chunks):
    # 결정 8: '제 N 장' 아래 짧은 조끼리 묶일 때만 생긴다 — 실문서 PDF 0건이라 합성으로 고정
    c = next(c for c in chunks['goodpeople_15_자원봉사활동규정.pdf'] if '| 교육 봉사 |' in c.text)
    lines = c.text.split('\n')
    i = lines.index('제 3 조(활동 수당)')
    assert lines[i + 1].startswith('| 활동 구분 |') and '제 3 조(활동 수당)' in (c.meta or {}).get('headings', [])


def test_쪽을_넘긴_70행_표는_한_표로_이어져_묶음마다_헤더와_도입_줄이_있다(chunks):
    # 결정 E·1·7c: 쪽마다 헤더가 반복된 표가 하나로 이어지고, 묶음마다 헤더·표 위 한 줄을 달며, 뒤 묶음은 제 쪽 번호
    batches = [c for c in chunks['goodpeople_15_자원봉사활동규정.pdf'] if '나눔센터-' in c.text]
    assert len(batches) >= 2
    # 도입 줄 = 표 바로 앞 글의 마지막 줄. docling이 앞 문단과 한 줄로 합칠 수 있어(실제 문서 모양, #247 보강 10/18) 끝만 본다
    leads = {c.text.split('\n')[0] for c in batches}
    assert len(leads) == 1 and leads.pop().endswith('봉사처별 월 배정 인원은 다음과 같다.')
    assert all(c.text.split('\n')[1] == '| 번호 | 봉사처 | 지역 | 월 배정 인원 |' for c in batches)
    assert sum(c.text.count('| 나눔센터-') for c in batches) == 70 and batches[-1].page > batches[0].page


def test_그림_면적_하한_1퍼센트_경계(fake_vlm):
    """#247 결정 17 — 0.5% 상자(유니폼)는 캡션 대상이 아니라 자리표시, 1.5% 상자(안전화)는 캡션이 붙는다. 실문서에서 내용 있는 작은
    캡처(1.2~4.6%)가 5% 하한에 걸려 빠졌다. 그림 생성의 정본은 sample_docs/corpus_v2/_src/build_picture_docs.py(P06·P07)."""
    from rag.chunking.docling_elements import PICTURE_PLACEHOLDER
    from tests.conftest import FAKE_CAPTION
    chunks = chunk_file(ROOT / 'sample_docs' / 'corpus_v2' / 'homeplus' / 'homeplus_15_매장운영지표안내.pdf')
    sec = next(c.text for c in chunks if '안전화는 매장 바닥 작업' in c.text)
    before, after = sec.split('안전화는 매장 바닥 작업', 1)
    assert PICTURE_PLACEHOLDER in before and FAKE_CAPTION not in before.split('업무용 물품을 지급한다')[-1]   # 0.5%: 자리표시
    assert FAKE_CAPTION in after                                                                         # 1.5%: 캡션
