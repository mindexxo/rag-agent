"""#245 청크 경계 — 진단 규칙(eval/chunk_boundary)과 합성 최악 사례 5건(sample_docs/corpus_v2/*_18_*).

측정 결론(eval/report_chunk_boundary.md): 경계는 글자를 잃지 않고 **맥락**을 잃는다. 여기서 고정하는 것:
  1. 진단 규칙이 경계·맥락 손실을 맞게 가른다(손으로 만든 조각 — 문서 변환 없이 수 초)
  2. 합성 문서의 골드 *_cb0NN 정답 스니펫이 조각 하나 안에 있다(질문이 "경계를 걸친 스니펫"이 아니라 "맥락만 앞 조각에 있는
     사실"을 묻게 설계했다 — 스니펫이 경계에 걸리면 이 축이 재는 것이 바뀐다)
  3. 글자 소실 0(TXT·MD — 분할기 오버랩이 잘린 문장을 다시 싣는다)
  4. **알려진 결함 재현** — 운영 청킹이 지금 이 맥락을 뒤 조각에서 떨어뜨린다는 사실. 고치는 PR(후속 이슈)은 해당 단언을 뒤집는다.
문서 생성의 정본은 sample_docs/corpus_v2/_src/build_boundary_docs.py.
"""
import json
from pathlib import Path

import pytest

from eval.chunk_boundary import census, classify_boundary, context_losses
from eval.chunk_invariants import chunks_lost

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / 'sample_docs' / 'corpus_v2'
GOLD = [json.loads(line) for line in open(ROOT / 'eval' / 'gold_set_v2.jsonl') if '_cb0' in line]


def _c(text: str, path=('문서', '절')) -> dict:
    return {'heading_path': list(path), 'text': text}


# ── 1. 진단 규칙 ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize('a, b, kind', [
    ('첫 문장이다.\n둘째 문장이다.', '셋째 문장이다.', 'sentence'),
    ('목록 머리는 다음과 같다\n1. 첫 항목', '2. 둘째 항목', 'next_item'),
    ('출입은 밤 10시로 하며, 마감 뒤 외출은', '하며, 마감 뒤 외출은 문자로 알린다', 'mid_overlap'),
    ('⑧ 화기를 소홀히 하거나 점화,', '흡연을 자행한 자', 'mid_cut'),
    ('숙소 출입은 밤 10시로 하며, 외출은', '은 문자로 알린다', 'mid_cut'),          # 우연한 짧은 일치(1자)는 오버랩이 아니다
    ('보관 온도는 섭씨로 재고 무게는', '3.5kg 이하로 담는다', 'mid_cut'),           # 소수로 시작하는 본문은 항목이 아니다
    ('| A-1 | 오일 | 90.', '1 | 14 | 표본 검사 |\n| A-2 | 오일 | 93.0 | 12 | 전수 검사 |', 'table_mid_row'),
    ('| A-1 | 오일 | 90.1 |', '| A-2 | 오일 | 93.0 |', 'table_row'),
    ('| A-1 | 오일 | 90.1 |', '다음 절 본문이다', 'table_text'),
])
def test_경계_분류(a, b, kind):
    assert classify_boundary(a, b) == kind


def test_목록_머리가_앞_조각에만_있으면_list_head():
    cs = [_c('다음 각 호의 어느 하나에 해당하면 즉시 귀국 조치한다.\n1. 무단 이탈\n2. 음주'), _c('3. 금품 수수\n4. 폭언')]
    assert [(x['kind'], x['at']) for x in context_losses(cs)] == [('list_head', 1)]


def test_목록_소제목이_앞_조각에만_있으면_list_label_뒤_조각에_다시_있으면_아님():
    a = _c('[상황 6] 리셀 상품 A/S 요청\n1. 구매처를 확인한다.\n2. 정품 여부를 묻는다.')
    assert [x['kind'] for x in context_losses([a, _c('3. 감정은 제공하지 않는다.')])] == ['list_label']
    assert context_losses([a, _c('[상황 6] 리셀 상품 A/S 요청\n3. 감정은 제공하지 않는다.')]) == []


def test_목록이라도_다른_섹션이면_세지_않는다():
    cs = [_c('다음 각 호와 같다.\n1. 가'), _c('2. 나', path=('문서', '다른 절'))]
    assert context_losses(cs) == []


def test_가리키는_말로_시작하면_reference():
    cs = [_c('개점은 1시간까지 늦출 수 있다.'), _c('이 경우 폐점도 늦춘다.')]
    assert [x['kind'] for x in context_losses(cs)] == ['reference']


_HDR = '| 코드 | 단가 |'


def test_표_묶음의_앞_설명_주석_머리행_손실():
    cs = [_c('단가는 2027년분부터 적용한다.\n단위는 kg당 원이다.'),
          _c(f'단위는 kg당 원이다.\n{_HDR}\n| A-1 | 100 |'),
          _c(f'단위는 kg당 원이다.\n{_HDR}\n| A-2 | 200 |\n※ 부가세 별도'),
          _c('다음 절')]
    got = sorted((x['kind'], x['at'], x['line']) for x in context_losses(cs))
    assert got == [('table_lead', 1, '단가는 2027년분부터 적용한다.'), ('table_lead', 2, '단가는 2027년분부터 적용한다.'),
                   ('table_note', 1, '※ 부가세 별도')]


def test_MD_표_뒤_조각은_머리행을_잃는다():
    cs = [_c(f'{_HDR}\n| A-1 | 1'), _c('00 |\n| A-2 | 200 |')]
    assert [x['kind'] for x in context_losses(cs)] == ['table_header']


def test_이웃한_다른_표는_한_표로_묶지_않는다():
    # 실문서 오탐(부칙 뒤 별지 표가 앞 표의 묶음으로 잡힘, 10/18) — 머리 행이 다르고 글로 시작하면 다른 표다
    cs = [_c(f'{_HDR}\n| A-1 | 100 |'), _c('#별지 1 보고서\n| 항목 | 내용 |\n| 일시 | 오늘 |')]
    assert context_losses(cs) == []


def test_머리_행이_다른_이웃_표가_바로_이어져도_한_표로_묶지_않는다():
    # PDF·DOCX 렌더도 MD처럼 머리 행 아래 구분 행('| --- |')을 단다 — 제 머리 행을 가진 조각은 새 표
    cs = [_c(f'{_HDR}\n| --- | --- |\n| A-1 | 100 |'), _c('| 항목 | 내용 |\n| --- | --- |\n| 일시 | 오늘 |')]
    assert context_losses(cs) == []


def test_표_뒤_다른_섹션의_주석은_표_주석이_아니다():
    cs = [_c(f'설명\n{_HDR}\n| A-1 | 100 |'), _c(f'설명\n{_HDR}\n| A-2 | 200 |'), _c('※ 다른 절의 주석', path=('문서', '다른 절'))]
    assert context_losses(cs) == []


def test_census는_XLSX를_세지_않고_같은_손실을_한_건으로_센다():
    cs = [_c('단가는 2027년분부터 적용한다.\n단위는 원이다.'),
          _c(f'단위는 원이다.\n{_HDR}\n| A-1 | 100 |'), _c(f'단위는 원이다.\n{_HDR}\n| A-2 | 200 |')]
    res = census({'t/표.pdf': cs, 't/시트.xlsx': cs})
    assert res['context_losses'] == {'table_lead': 1}        # 묶음 둘에서 빠진 같은 줄 = 1건, XLSX는 0건
    assert res['boundaries'] == {'sentence': 1, 'table_text': 1}   # XLSX 경계도 세지 않는다(PDF 쪽 둘만)


# ── 2~4. 합성 문서 ───────────────────────────────────────────────────────────

def _path(name: str) -> Path:
    return CORPUS / name.split('_')[0] / name


class _Chunked(dict):
    """문서별 지연 청킹 — docling은 PDF·DOCX에만 필요하다. TXT·MD 테스트가 docling 없는 환경에서 같이 스킵되지 않게."""
    def __missing__(self, name):
        if not name.endswith(('.txt', '.md')):
            pytest.importorskip('docling')
        from rag.chunking import chunk_file
        self[name] = chunk_file(_path(name))
        return self[name]


@pytest.fixture(scope='module')
def chunked():
    return _Chunked()


def _dicts(chunks) -> list[dict]:
    return [{'heading_path': c.heading_path, 'text': c.text, 'table_text': (c.meta or {}).get('table_text')} for c in chunks]


def test_골드는_합성_다섯_문서를_덮는다():
    assert len(GOLD) == 13
    assert {g['expected_docs'][0].split('_')[0] for g in GOLD} == {'goodpeople', 'summers', 'harim', 'homeplus', 'aromanica'}


@pytest.mark.parametrize('g', GOLD, ids=[g['id'] for g in GOLD])
def test_정답_스니펫은_스니펫마다_조각_하나_안에_있다(g, chunked):
    for e in g['expected_chunks']:
        assert any(e['snippet'] in c.text for c in chunked[e['filename']]), e['snippet']


@pytest.mark.parametrize('name', ['goodpeople_18_해외봉사단복무지침.txt', 'aromanica_18_향료원료입고기준.md'])
def test_글자_소실_0(name, chunked):
    lines = [l.lstrip('#').strip() for l in _path(name).read_text(encoding='utf-8').split('\n')]   # MD 헤딩 표지는 제목 경로에서 빠진다
    assert chunks_lost(lines, chunked[name]) == []


def _kinds(chunked, name) -> set[str]:
    return {x['kind'] for x in context_losses(_dicts(chunked[name]))}


@pytest.mark.parametrize('name, kind, case', [
    ('goodpeople_18_해외봉사단복무지침.txt', 'list_head', 'T6 목록 머리와 뒤 항목 분리'),
    ('summers_18_매장운영매뉴얼.docx', 'reference', "T5 '이 경우' 문장만 따로"),
    ('aromanica_18_향료원료입고기준.md', 'table_header', 'B6 MD 표 뒤 조각에 머리 행 없음'),
])
def test_알려진_결함_재현(name, kind, case, chunked):
    """운영 청킹의 현재 동작 — 후속 이슈가 고치면 이 단언을 뒤집는다(eval/report_chunk_boundary.md '후속')."""
    assert kind in _kinds(chunked, name), case


@pytest.mark.parametrize('name, kind, case', [
    ('harim_18_협력농가납품단가표.pdf', 'table_lead', 'B1 세 줄 설명 중 마지막만 묶음에'),
    ('homeplus_18_PB품질검사기준표.docx', 'table_lead', 'B2 150자 넘는 설명은 묶음에 안 붙음'),
    ('homeplus_18_PB품질검사기준표.docx', 'table_note', 'B3 표 아래 단위 주석이 묶음 밖'),
])
def test_표_설명과_주석은_모델이_받는_표_전체에_있다(name, kind, case, chunked):
    # #268 — 큰 표 묶음은 근거 준비에서 표 전체(table_text)로 바뀐다(#257). 그 표 전체에 앞 설명 여러 줄·아래 주석을 싣는다
    assert kind not in _kinds(chunked, name), case


def test_TXT_긴_문단은_문장_중간에서_잘리지만_오버랩이_다시_싣는다(chunked):
    cs = chunked['goodpeople_18_해외봉사단복무지침.txt']
    kinds = [classify_boundary(a.text, b.text) for a, b in zip(cs, cs[1:]) if a.heading_path == b.heading_path]
    assert 'mid_overlap' in kinds and 'mid_cut' not in kinds
