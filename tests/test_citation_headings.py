"""#233 조문 인용 ≠ 조 제목 — 원문 소실 0과 조 이름표.

제목 판정 규칙 자체는 tests/test_identifiers.py(split_heading_line)·test_chunking_parsers.py(DOCX 폴백)가 고정한다.
여기는 두 층을 고정한다(소실 판정은 eval/chunk_invariants):
  - 조립 단위(TestBuildSections·TestMdSections) — 본문 없는 헤딩의 자리, 이미 나간 상위, 문서 끝
  - 문서 단위(TestCitationDocs) — 합성 4건(sample_docs/corpus_v2/*_16_*, 정본 생성기 _src/build_citation_docs.py)을 실제
    경로로 청킹해 원문 소실 0, 골드 *_ct 문항의 근거(스니펫·그 조 구간)와 문서에 없는 조의 가짜 이름표를 본다. #233 전 코드
    (fd86876)로는 골드 9문항 전부 실패했다(2026-10-18 실측, 실패 사유별: 정답 스니펫이 어느 조각에도 없음 7 · 스니펫은 있으나
    그 조 구간에 옆 조 내용이 섞임 2). 수정 후 남는 알려진 실패는 아래 KNOWN_FAIL(#260).
    가짜 이름표('(이하 …)에 따른' 인용 줄 → 문서에 없는 '12조')는 골드가 아니라 여기서만 본다 — 그 문장은 제12조를 실제로
    언급하므로 "제12조 내용" 질문에 그 문장으로 답하는 것이 오답이라 할 수 없어, 생성 골드로는 결함이 갈리지 않았다(설계 11).
"""

import importlib.util
import json
import re
from pathlib import Path

import pytest

from eval.chunk_invariants import chunks_lost, norm, sections_lost
from rag.chunking import chunk_file
from rag.chunking.markdown import md_sections
from rag.chunking.sections import build_sections
from rag.identifiers import chunk_identifiers, cut_identifier_span, effective_headings, normalize_identifiers

ROOT = Path(__file__).resolve().parent.parent


def _h(text, level):
    return (None, text, level)


def _b(text):
    return (None, text, None)


class TestBuildSections:
    """고아(본문 없이 밀린 헤딩)는 다음 본문 앞에 문서 순서로 되살린다(#137 그대로). #233은 가짜 고아·문서 끝만 고친다(설계 9-재)."""

    def test_이미_경로로_나간_장은_고아가_아니다(self):
        # 예전: '제2조(삭제)' 뒤에 제2장이 오면 이미 제1조 경로로 나간 '제1장'까지 고아로 잡아 제2장 첫 본문에 다시 붙였다
        items = [
            _h('규정', 1),
            _h('제1장 총칙', 2),
            _h('제1조(목적)', 3),
            _b('목적 본문'),
            _h('제2조(삭제)', 3),
            _h('제2장 복무', 2),
            _h('제3조(근무)', 3),
            _b('근무 본문'),
        ]
        secs = build_sections(items)
        assert [(s.heading_path, s.body) for s in secs] == [
            (['규정', '제1장 총칙', '제1조(목적)'], '목적 본문'),
            (['규정', '제2장 복무', '제3조(근무)'], '제2조(삭제)\n근무 본문'),
        ]
        assert sections_lost(items, secs) == []

    def test_상위_번호_제목은_다음_본문_앞에(self):
        # 실제로 생기는 고아의 거의 전부가 이 꼴이다 — docling이 장 없는 PDF에서 '2.'와 '2.1'을 같은 레벨로 잡는다(실측은
        # build_sections docstring)
        items = [
            _h('정책', 1),
            _h('1. 목적', 2),
            _b('목적 본문'),
            _h('2. 반품 규정', 2),
            _h('2.1 신청 기한', 2),
            _b('7일 이내'),
        ]
        assert [(s.heading_path, s.body) for s in build_sections(items)][1] == (
            ['정책', '2.1 신청 기한'],
            '2. 반품 규정\n7일 이내',
        )

    def test_알려진_한계_삭제_조는_다음_조_경로_아래로(self):
        # #260·설계 9-재: 배치를 바꾸면 위 상위 제목 고아가 함께 흔들려 고치지 않았다 — 바뀌면 의도한 변경인지 확인할 것
        items = [_h('규정', 1), _h('제3조(삭제)', 2), _h('제4조(대피)', 2), _b('대피 본문')]
        assert [(s.heading_path, s.body) for s in build_sections(items)] == [
            (['규정', '제4조(대피)'], '제3조(삭제)\n대피 본문')
        ]

    def test_문서_끝_헤딩은_본문으로_경로는_나간_앞부분(self):
        items = [_h('규정', 1), _h('제1장 총칙', 2), _h('제6조(목적)', 3), _b('본문'), _h('제7조(삭제)', 3)]
        assert [(s.heading_path, s.body) for s in build_sections(items)] == [
            (['규정', '제1장 총칙', '제6조(목적)'], '본문'),
            (['규정', '제1장 총칙'], '제7조(삭제)'),  # 같은 글이 경로·본문에 두 번 들어가지 않는다
        ]

    def test_문서_끝에_고아와_끝_헤딩이_같이_남으면_문서_순서로(self):
        items = [
            _h('규정', 1),
            _h('제1조(목적)', 2),
            _b('본문'),
            _h('제2장 보칙', 2),
            _h('제8조(삭제)', 3),
            _h('제9조(삭제)', 3),
        ]
        secs = build_sections(items)
        assert (secs[-1].heading_path, secs[-1].body) == (['규정'], '제2장 보칙\n제8조(삭제)\n제9조(삭제)')
        assert sections_lost(items, secs) == []

    def test_되살린_고아는_문서_순서로(self):
        # 빠진 순서로 붙이면 늦게 빠진 상위 'A'가 먼저 빠진 하위 'a1' 뒤에 왔다
        items = [_h('T', 1), _h('A', 2), _h('a1', 3), _h('a2', 3), _h('B', 2), _b('본문')]
        assert [(s.heading_path, s.body) for s in build_sections(items)] == [(['T', 'B'], 'A\na1\na2\n본문')]

    def test_헤딩만_있는_문서는_섹션_0개(self):
        # 결정 3 — 빈 양식은 색인하지 않는다(rag/ingestion이 "추출된 텍스트 없음"으로 실패시킨다)
        assert build_sections([_h('규정', 1), _h('제1조(목적)', 2), _h('제2조(정의)', 2)]) == []

    @pytest.mark.parametrize(
        'items',
        [
            [_h('T', 1), _h('A', 2), _h('a1', 3), _h('a2', 3), _h('B', 2), _b('x'), _h('b1', 3), _h('C', 2)],
            [
                _h('T', 1),
                _b('x'),
                _h('A', 2),
                _h('a1', 3),
                _b('y'),
                _h('a2', 3),
                _h('a3', 3),
                _h('B', 2),
                _h('b1', 3),
                _b('z'),
            ],
            [_b('머리말'), _h('A', 2), _h('B', 1), _h('C', 3), _b('x'), _h('D', 2)],
        ],
    )
    def test_헤딩_배치가_어떻든_글은_남는다(self, items):
        assert sections_lost(items, build_sections(items)) == []


class TestMdSections:
    def test_본문_없는_조는_중간이든_끝이든_남는다(self, tmp_path):
        # 예전 md 조립은 자기 스택이라 본문 없는 헤딩이 같은 레벨 헤딩에 밀리거나 문서 끝이면 사라졌다
        p = tmp_path / 'r.md'
        p.write_text(
            '# 규정\n\n## 제1조(목적)\n\n목적 본문\n\n## 제2조(삭제)\n\n## 제3조(정의)\n\n정의 본문\n\n'
            '## 제4조(삭제)\n',
            encoding='utf-8',
        )
        assert [(s.heading_path, s.body) for s in md_sections(p)] == [
            (['규정', '제1조(목적)'], '목적 본문'),
            (['규정', '제3조(정의)'], '제2조(삭제)\n정의 본문'),
            (['규정'], '제4조(삭제)'),
        ]

    def test_헤딩_사이_문단은_줄로_나뉜_한_덩어리(self, tmp_path):
        # #267부터 docling — 문단은 요소 하나씩 나오고 조립이 줄바꿈 하나로 잇는다(PDF·DOCX와 같은 모양, 정규식 시절엔 빈 줄까지 뒀다)
        p = tmp_path / 'r.md'
        p.write_text('# 제목\n\n\n첫 문단\n\n둘째 문단\n\n\n## 절\n\n\n', encoding='utf-8')
        assert [(s.heading_path, s.body) for s in md_sections(p)] == [
            (['제목'], '첫 문단\n둘째 문단'),
            (['제목'], '절'),
        ]


# ----- 문서 단위: 합성 4건 + 골드 *_ct -------------------------------------------------------------

_spec = importlib.util.spec_from_file_location(
    'build_citation_docs', ROOT / 'sample_docs' / 'corpus_v2' / '_src' / 'build_citation_docs.py'
)
_SRC = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_SRC)
# 문서별 원문 줄 — 생성기의 문단 목록이 정본이다(문서를 다시 읽어 만들면 파서 결함이 정답에 섞인다)
SOURCE_LINES = {
    'aromanica_16_시향관운영지침.docx': _SRC.AROMANICA,
    'summers_16_매장안전관리규정.docx': [t for _, t in _SRC.SUMMERS],
    'homeplus_16_PB시식행사운영기준.md': [
        re.sub(r'^#+\s*', '', line) for line in _SRC.HOMEPLUS.splitlines() if line.strip()
    ],
    'goodpeople_16_후원물품보관규정.pdf': [t for _, t in _SRC.GOODPEOPLE],
}
GOLD = [
    g
    for g in map(json.loads, open(ROOT / 'eval' / 'gold_set_v2.jsonl', encoding='utf-8'))
    if re.fullmatch(r'[a-z]+_ct\d{3}', g['id'])
]
# 알려진 실패(#260): 본문 없는 삭제 조가 다음 조 본문 앞에 붙어(build_sections 고아 배치, #137) 그 조 구간에 옆 조 내용이
# 섞인다 — 문서 aromanica는 묶음 안에서 '제8조(시행일) / 제7조(삭제) / 제8조 본문' 순서가 되어 제8조 줄도 끊긴다.
# 배치를 바꾸면 상위 번호 제목 고아가 함께 흔들려 #233에서 고치지 않았다(설계 9-재, 실측은 build_sections docstring).
# strict — 고치는 순간 XPASS로 실패하니 이 목록에서 지울 것.
KNOWN_FAIL = {
    'aromanica_ct003',
    'summers_ct001',
    'homeplus_ct001',
    'goodpeople_ct001',
    'goodpeople_ct002',
    'aromanica_16_시향관운영지침.docx',
}
_xfail_260 = pytest.mark.xfail(strict=True, reason='#260 삭제 조가 다음 조 본문 앞에 붙음')


def _mark(value, key):
    return pytest.param(value, id=key, marks=_xfail_260 if key in KNOWN_FAIL else ())


_DOC_OF = {name.split('_')[0]: name for name in SOURCE_LINES}


@pytest.fixture(scope='module')
def chunks():
    pytest.importorskip('docling')
    return {name: chunk_file(ROOT / 'sample_docs' / 'corpus_v2' / name.split('_')[0] / name) for name in SOURCE_LINES}


def _article_spans(chunks_, query):
    """질문의 조 번호 이름표가 걸린 조각마다 그 조 구간 — 운영 오리기(rag/search/retriever prepare_evidence)와 같은 함수."""
    token = next(t for t in normalize_identifiers(query) if t.endswith('조'))
    out = []
    for c in chunks_:
        heads = effective_headings((c.meta or {}).get('headings') or [], c.text)
        if token in chunk_identifiers(c.heading_path, heads):
            out.append(cut_identifier_span(c.text, heads, token))
    return out


class TestCitationDocs:
    def test_골드는_문서_넷에_아홉_문항(self):
        assert (
            len(GOLD) == 9
            and all(g['has_evidence'] for g in GOLD)
            and {g['id'].split('_')[0] for g in GOLD} == set(_DOC_OF)
        )

    @pytest.mark.parametrize('name', [_mark(n, n) for n in SOURCE_LINES])
    def test_원문_소실_0(self, name, chunks):
        assert chunks_lost(SOURCE_LINES[name], chunks[name]) == []

    @pytest.mark.parametrize('g', [_mark(g, g['id']) for g in GOLD])
    def test_근거_스니펫이_있고_그_조_구간에_다른_조_내용이_없다(self, g, chunks):
        cs = chunks[g['expected_docs'][0]]
        assert any(norm(g['expected_chunks'][0]['snippet']) in norm(c.text) for c in cs)
        if any(t.endswith('조') for t in normalize_identifiers(g['query'])):
            spans = _article_spans(cs, g['query'])
            assert spans, '조 번호 이름표가 걸린 조각이 없다'
            assert not [m for m in g.get('must_not_contain', []) for s in spans if norm(m) in norm(s)]

    @pytest.mark.parametrize(
        'name,query',
        [
            ('aromanica_16_시향관운영지침.docx', '시향관 운영 지침 제12조 내용 알려줘'),
            ('homeplus_16_PB시식행사운영기준.md', 'PB 시식 행사 운영 기준 제9조에는 뭐라고 되어 있어요?'),
            ('goodpeople_16_후원물품보관규정.pdf', '후원 물품 보관 규정 제15조 내용이 뭐예요?'),
        ],
    )
    def test_문서에_없는_조는_이름표가_없다(self, name, query, chunks):
        # "(이하 …)에 따라" 인용 줄이 본문 보충(effective_headings)에서 헤딩으로 잡혀 없는 조의 이름표가 생겼다(fd86876 실측 3건)
        assert _article_spans(chunks[name], query) == []

    def test_인용_줄은_제목이_되지_않고_진짜_제목은_된다(self, chunks):
        cs = chunks['aromanica_16_시향관운영지침.docx']
        paths = {h for c in cs for h in c.heading_path}
        heads = {h for c in cs for h in effective_headings((c.meta or {}).get('headings') or [], c.text)}
        assert not [h for h in paths | heads if h.startswith(('제4조 및', '제6조 제2항', '제12조'))]
        assert {
            '제2조(시향 예약(단체))',
            '제 5 조1(분실 신고)',
            '제7조(삭제)',
        } <= heads  # 괄호 중첩·'의' 생략(#250)·삭제 조
        ids = {
            t
            for c in cs
            for t in chunk_identifiers(c.heading_path, effective_headings((c.meta or {}).get('headings') or [], c.text))
        }
        assert '5조의1' in ids and '12조' not in ids

    def test_PDF_이미_나간_앞_장이_다음_장에_다시_붙지_않는다(self, chunks):
        # 가짜 고아 — '제 2 조(삭제)' 뒤에 제2장이 오면 이미 나간 '제 1 장 총칙'이 제2장 첫 본문에 다시 붙었다
        cs = chunks['goodpeople_16_후원물품보관규정.pdf']
        ch2 = [c for c in cs if '제 2 장 보관' in c.heading_path]
        assert ch2 and not any('제 1 장 총칙' in c.text for c in ch2)
