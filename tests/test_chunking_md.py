"""MD 파서 = docling + 보정 3개(#267, 정의점 rag/chunking._md_sections).

docling(2.126) MD 백엔드를 그대로 쓰면 표 칸 '\\|'의 내용을 버리고, '<br>'이 든 행을 표 밖으로 흩뜨리고, 링크 주소를 빼고, 인라인
서식으로 문장을 조각낸다(#267 실측). 그 보정과, 정규식 파서가 못 하던 구조 인식(밑줄식 헤딩·닫는 '#'·표 블록)을 고정한다.
운영에는 원문 소실 가드를 두지 않았다(#267 결정 4) — 대신 여기서 저장소의 MD 문서 전부에 "원문 소실 0"을 건다. docling이 또 모르는
데서 내용을 버리면 이 테스트가 먼저 깨진다. 합성 문서 생성의 정본은 sample_docs/corpus_v2/_src/build_md_docs.py.
"""
import json
import re
from pathlib import Path

import pytest

from eval.chunk_invariants import chunks_lost
from rag.chunking import _md_sections, chunk_file

pytest.importorskip('docling')

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / 'sample_docs' / 'corpus_v2'
MD_DOCS = sorted(p for p in CORPUS.glob('*/*.md'))
GOLD = [json.loads(line) for line in (ROOT / 'eval' / 'gold_set_v2.jsonl').read_text(encoding='utf-8').splitlines() if '_mk0' in line]


def _md(tmp_path, text: str) -> Path:
    p = tmp_path / 'doc.md'
    p.write_text(text, encoding='utf-8')
    return p


def _body(tmp_path, text: str) -> str:
    return '\n'.join(s.body for s in _md_sections(_md(tmp_path, text)))


# ── 보정 ─────────────────────────────────────────────────────────────────────

def test_표_칸_안_이스케이프_세로선은_칸을_버리지_않는다(tmp_path):
    # docling 그대로면 '| AE | Mono 재킷 | 모노랩 |' — 셋째 칸('홀로그램 확인')이 출력 어디에도 없었다
    body = _body(tmp_path, '# 제목\n\n| 모델 | 브랜드 | 추가 항목 |\n| --- | --- | --- |\n| AE\\|Mono | 모노랩 | 홀로그램 확인 |\n')
    assert '| AE\\|Mono | 모노랩 | 홀로그램 확인 |' in body


@pytest.mark.parametrize('br', ['<br>', '<br/>', '<br />', '<BR>'])
def test_표_칸_안_줄바꿈_태그는_행_안에_남는다(tmp_path, br):
    # docling 그대로면 '<br>' 뒤 값이 표 밖 글 한 줄로 나가고 행에서 빠졌다. 변형은 '<br>' 하나로 적는다
    body = _body(tmp_path, f'# 제목\n\n| 경로 | 서류 |\n| --- | --- |\n| 택배 | 신청서{br}보증서 |\n')
    assert '| 택배 | 신청서<br>보증서 |' in body
    assert body.index('| 경로 | 서류 |') < body.index('보증서')


def test_링크는_글_뒤에_주소를_붙인다(tmp_path):
    body = _body(tmp_path, '# 제목\n\n신청서는 [양식](https://example.invalid/form)에서 받는다.\n')
    assert body == '신청서는 양식(https://example.invalid/form)에서 받는다.'


@pytest.mark.parametrize('src, want', [
    ('요청은 **이메일 접수**로만 받는다.', '요청은 이메일 접수로만 받는다.'),       # 서식 뒤 조사는 붙인다(원문에 공백 없음)
    ('요청은 **이메일** 접수로만 받는다.', '요청은 이메일 접수로만 받는다.'),       # 원문에 공백이 있으면 띄운다
    ('**주의** 미성년자는 동의서가 필요하다.', '주의 미성년자는 동의서가 필요하다.'),
])
def test_인라인_서식으로_쪼개진_문장은_원문_띄어쓰기대로_한_줄(tmp_path, src, want):
    assert _body(tmp_path, f'# 제목\n\n{src}\n') == want


def test_코드_블록은_들여쓰기를_지킨다(tmp_path):
    # _docling_norm은 공백을 접는다 — 코드에 태우면 YAML 구조가 깨졌다(#267 리뷰 A: 'root:\n child:\n leaf: 1')
    assert _body(tmp_path, '# 제목\n\n```yaml\nroot:\n  child:\n    leaf: 1\n```\n') == 'root:\n  child:\n    leaf: 1'


# ── 구조 인식(정규식 파서가 못 하던 것) ──────────────────────────────────────────

def test_밑줄식_헤딩도_헤딩이다(tmp_path):
    secs = _md_sections(_md(tmp_path, '안내\n====\n\n머리말.\n\n해지 응대\n--------\n\n본문.\n'))
    assert [(s.heading_path, s.body) for s in secs] == [(['안내'], '머리말.'), (['안내', '해지 응대'], '본문.')]


def test_닫는_샾은_제목에_남지_않는다(tmp_path):
    assert _md_sections(_md(tmp_path, '# 기준 #\n\n## 1. 경로 ##\n\n본문.\n'))[0].heading_path == ['기준', '1. 경로']


def test_전부_1로_쓴_번호_목록은_순서대로_번호가_붙는다(tmp_path):
    assert _body(tmp_path, '# 제목\n\n1. 가\n1. 나\n1. 다\n') == '1. 가\n2. 나\n3. 다'


def test_앞머리_메타는_빈_헤딩을_만들지_않는다(tmp_path):
    secs = _md_sections(_md(tmp_path, '---\n문서번호: X-1\n시행일: 2027-03-01\n---\n\n# 제목\n\n본문.\n'))
    assert all(h for s in secs for h in s.heading_path)
    assert '2027-03-01' in '\n'.join(s.body for s in secs)


def test_큰_MD_표는_행_단위로_나뉘고_묶음마다_열_이름이_있다():
    # #245 B6: 정규식 시절에는 글로 분할기를 타 행 한가운데서 잘리고 뒤 조각에 열 이름이 없었다
    chunks = chunk_file(CORPUS / 'aromanica' / 'aromanica_18_향료원료입고기준.md')
    batches = [c for c in chunks if '| AQ-50' in c.text]
    assert len(batches) >= 2
    header = '| 원료 코드 | 원료명 | 산지 | 최소 순도 | 보관 온도 | 입고 검사 |'
    assert all(header in c.text for c in batches)
    assert all(line.endswith('|') for c in batches for line in c.text.split('\n') if line.startswith('|'))   # 행 중간에서 안 잘림
    assert {(c.meta or {}).get('table_key') for c in batches} != {None}                                      # #257 표 전체 확장 대상


# ── 원문 소실 0 — 저장소의 MD 전부 ───────────────────────────────────────────────

_FENCE = re.compile(r'^(`{3,}|~{3,})')
_RULE = re.compile(r'^(=+|-+|\*{3,}|\|?[\s:|-]*-{3,}[\s:|-]*\|?)$')     # 밑줄식 헤딩 밑줄·가로줄·표 구분 행·앞머리 경계
_HTML_TAG = re.compile(r'</?(details|summary)>')
_MARK = re.compile(r'^\s*(#{1,6}\s+|>\s*|[-*+]\s+|\d+[.)]\s+)')


def _plain(line: str) -> str:
    """마크다운 줄 → 글자만(표기는 docling이 문법대로 걷어낸다). 링크는 운영 보정과 같은 '글(주소)', 그림은 대체 글."""
    line = line.strip()
    if _FENCE.match(line) or _RULE.match(line):
        return ''
    line = _HTML_TAG.sub('', line)
    line = re.sub(r'!\[([^\]]*)\]\([^)]*\)', r'\1', line)
    line = re.sub(r'\[([^\]]*)\]\(([^)]*)\)', r'\1(\2)', line)
    line = _MARK.sub('', line).replace('**', '').rstrip('#').strip()
    return line


@pytest.mark.parametrize('path', MD_DOCS, ids=[p.name for p in MD_DOCS])
def test_원문_글자는_조각_어딘가에_남는다(path):
    lines = [_plain(line) for line in path.read_text(encoding='utf-8').split('\n')]
    assert chunks_lost([line for line in lines if line], chunk_file(path)) == []


# ── 합성 골드 ────────────────────────────────────────────────────────────────

def test_골드는_합성_MD_세_문서를_덮는다():
    assert len(GOLD) == 27
    assert {g['expected_docs'][0] for g in GOLD} == {p.name for p in MD_DOCS if '_19_' in p.name}


@pytest.fixture(scope='module')
def mk_chunks():
    return {p.name: chunk_file(p) for p in MD_DOCS if '_19_' in p.name}


@pytest.mark.parametrize('g', GOLD, ids=[g['id'] for g in GOLD])
def test_정답_스니펫이_조각_하나_안에_있다(g, mk_chunks):
    """스니펫은 원문 표기 — 표지('**'·목록 기호)와 공백은 걷어내고 비교한다."""
    key = lambda s: re.sub(r'\s+', '', s.replace('**', ''))
    for e in g['expected_chunks']:
        assert any(key(e['snippet']) in key(c.text) for c in mk_chunks[e['filename']]), e['snippet']
