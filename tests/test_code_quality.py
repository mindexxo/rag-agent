"""코드 품질 래칫 게이트 — AGENTS.md "코드 품질 기준"을 기계로 강제한다.

기준선 tests/quality_baseline.json에 지금의 위반을 동결하고, **그보다 나빠지면 실패**한다. 네 가지를 본다:
  - ruff: 파일 × 규칙 코드별 위반 수(설정은 pyproject.toml). 줄 번호가 아니라 개수라서 무관한 수정으로 줄이 밀려도 안 깨진다.
  - format: `ruff format`이 고칠 파일 목록. 목록에 없는 파일(새 파일 포함)이 포맷에서 벗어나면 실패.
  - size: 함수 코드줄 > FUNC_MAX, 모듈 코드줄 > MODULE_MAX(Ruff에 규칙이 없어 여기서 잰다). 코드줄 = 주석·docstring·빈 줄 제외
    — 설명을 크기 벌점으로 세지 않는다(설명의 양은 history 항목과 리뷰가 본다).
  - history: 파일별 주석·docstring 안의 히스토리 표지 수(이슈 번호·날짜·경위 낱말). 주석은 이해하기 어려운 곳에만 간결하게 쓰고
    경위는 이슈·PR·커밋에 둔다(AGENTS.md 규약) — 새 파일은 0, 기존 파일은 늘지 않게.
대상 파일은 git이 아는 파일(추적 + 미추적이되 무시되지 않은 것)이다 — 로컬에만 있는 gitignore 파일(eval/results 등)이
사람마다 결과를 다르게 만들지 않게 ruff와 크기 스캔이 같은 목록을 쓴다.

판정
  - 새 위반·악화 → 실패. 리뷰 신호(복잡도·분기·인자·크기)도 **새로 넘으면 막힌다** — 넘을 일이면 먼저 책임을 나눠라.
    정말 불가피하면 갱신 명령에 QUALITY_ALLOW_GROWTH=1을 붙여 기준선을 늘리고 PR에 사유를 적는다(기준선 diff가 리뷰 대상이다).
    파일·함수 이름을 바꾸면 키가 바뀌어 "새 위반"으로 읽힌다 — 같은 방법으로 옮기고 PR에 "이름 변경"이라고 적는다.
  - 개선(위반 감소·해소) → 경고만 낸다(실패 아님). 기준선 조이기는 그 정리를 한 PR이 한다:
        QUALITY_BASELINE_UPDATE=1 pytest tests/test_code_quality.py
    갱신은 기본적으로 **줄이기만** 한다(늘어난 항목은 그대로 실패로 남는다). 개선을 실패로 만들지 않는 이유: 여러 세션이
    동시에 작업하는 저장소라, 무관한 정리마다 기준선 JSON을 고치게 하면 그 파일이 상시 머지 충돌 지점이 된다.
  - 기준선 JSON이 머지 충돌하면 손으로 풀지 말고 한쪽을 받은 뒤 갱신 명령으로 다시 만든다.
  - `# noqa`는 위반을 센 수에서 뺀다 — "명시적으로 끈 것"이라 리뷰에서 사유를 본다. pyproject.toml의 상한·제외 완화도 같다.

ruff는 requirements.txt에 버전 고정돼 있다. 없으면 건너뛰되 CI=1(또는 true)이면 실패한다.
버전이 다르면 실패한다 — 규칙 판정이 버전마다 달라 기준선이 흔들린다. ruff를 올릴 때는 같은 PR에서 기준선을 다시 만든다
(QUALITY_BASELINE_UPDATE=1 QUALITY_ALLOW_GROWTH=1).
"""

import ast
import io
import json
import os
import re
import subprocess
import tokenize
import warnings
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
BASELINE = Path(__file__).resolve().parent / 'quality_baseline.json'
TARGETS = [
    'rag',
    'routers',
    'schemas',
    'eval',
    'tests',
    'main.py',
    'config.py',
    'database.py',
    'text_norm.py',
    'conftest.py',
]
FUNC_MAX = (
    50  # 함수 코드줄 — Google 가이드는 "약 40줄이면 분해 검토"(원줄 기준). 코드줄만 세므로 문장 수 상한(50)에 맞춘다
)
MODULE_MAX = 300  # 모듈 코드줄 — 넘으면 책임이 하나인지 검토(AGENTS.md)
# 이슈 번호는 앞에 글자가 붙어도(PR 표기)·뒤에 한글 조사가 붙어도 잡는다 — 그래서 단어 경계 대신 숫자 경계
HISTORY = re.compile(r'(?<![&#/])#\d{2,4}(?!\d)|(?<!\d)20\d\d-\d\d(?:-\d\d)?(?!\d)|실측|기각|선례|결정 ?\d+')


def _flag(name: str) -> bool:
    return os.environ.get(name, '').strip().lower() in ('1', 'true', 'yes')


def _pinned_version() -> str:
    for line in (ROOT / 'requirements.txt').read_text(encoding='utf-8').splitlines():
        if line.startswith('ruff=='):
            return line.split('==', 1)[1].split()[0]
    raise AssertionError('requirements.txt에 ruff==버전 줄이 없다')


def _ruff_bin() -> str:
    try:
        from ruff.__main__ import find_ruff_bin
    except ImportError:
        if _flag('CI'):
            pytest.fail('ruff 미설치 — pip install -r requirements.txt')
        pytest.skip('ruff 미설치 — pip install -r requirements.txt (CI에서는 실패)')
    ruff = find_ruff_bin()
    version = subprocess.run([ruff, '--version'], capture_output=True, text=True, check=True).stdout.split()[-1]
    if version != (pinned := _pinned_version()):
        pytest.fail(f'ruff {version} ≠ 고정 {pinned} — 판정이 버전마다 달라 기준선과 비교할 수 없다(requirements.txt)')
    return ruff


def _files() -> list[str]:
    """대상 .py — git 추적 + 미추적(무시 제외). 디스크에서 지운 파일은 뺀다."""
    out = subprocess.run(
        ['git', 'ls-files', '--cached', '--others', '--exclude-standard', '--', *TARGETS],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    return sorted({f for f in out if f.endswith('.py') and (ROOT / f).is_file()})


def _rel(filename: str) -> str:
    return os.path.relpath(filename, ROOT)


def _ruff_counts(ruff: str, files: list[str]) -> dict[str, int]:
    proc = subprocess.run(
        [ruff, 'check', '--no-cache', '--force-exclude', '--output-format', 'json', '--exit-zero', *files],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    counts: dict[str, int] = {}
    for v in json.loads(proc.stdout):
        key = f'{_rel(v["filename"])}::{v["code"]}'  # 구문 오류는 code가 None → 새 키라 악화로 걸린다
        counts[key] = counts.get(key, 0) + 1
    return counts


def _unformatted(ruff: str, files: list[str]) -> list[str]:
    proc = subprocess.run(
        [ruff, 'format', '--no-cache', '--force-exclude', '--check', '--output-format', 'json', *files],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if proc.returncode not in (0, 1):  # 1 = 고칠 파일 있음. 그 밖은 실행 오류 — 조용히 0건으로 넘기지 않는다
        pytest.fail(f'ruff format 실행 실패: {proc.stderr}')
    return sorted({_rel(v['filename']) for v in json.loads(proc.stdout or '[]')})


def _code_lines(source: str, tree: ast.Module) -> set[int]:
    """주석·docstring·빈 줄을 뺀 코드 줄 번호."""
    doc: set[int] = set()
    for node in ast.walk(tree):
        is_scope = isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        if is_scope and ast.get_docstring(node, clean=False):
            first = node.body[0]
            doc.update(range(first.lineno, first.end_lineno + 1))
    lines: set[int] = set()
    skip = {tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT, tokenize.ENDMARKER}
    for tok in tokenize.generate_tokens(io.StringIO(source).readline):
        if tok.type not in skip:
            lines.update(n for n in range(tok.start[0], tok.end[0] + 1) if n not in doc)
    return lines


def _functions(node: ast.AST, prefix: str = ''):
    """(정규 이름, 함수 노드) — 모든 문장 블록(if·try·with·for 포함)을 훑는다. 이름은 함수·클래스에서만 점으로 잇는다."""
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield f'{prefix}{child.name}', child
            yield from _functions(child, f'{prefix}{child.name}.')
        elif isinstance(child, ast.ClassDef):
            yield from _functions(child, f'{prefix}{child.name}.')
        else:
            yield from _functions(child, prefix)


def _size_violations(files: list[str]) -> dict[str, int]:
    """{'파일::<module>' 또는 '파일::함수경로': 코드줄} — 상한을 넘는 것만. 같은 이름 정의(property setter·조건부 정의)는 #2부터."""
    out: dict[str, int] = {}
    for rel in files:
        source = (ROOT / rel).read_text(encoding='utf-8')
        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            pytest.fail(f'{rel}: 구문 오류 — {e}')
        code = _code_lines(source, tree)
        if len(code) > MODULE_MAX:
            out[f'{rel}::<module>'] = len(code)
        seen: dict[str, int] = {}
        for qualname, node in _functions(tree):
            seen[qualname] = seen.get(qualname, 0) + 1
            n = sum(1 for line in code if node.lineno <= line <= node.end_lineno)
            if n > FUNC_MAX:
                suffix = f'#{seen[qualname]}' if seen[qualname] > 1 else ''
                out[f'{rel}::{qualname}{suffix}'] = n
    return out


def _explanations(source: str) -> list[str]:
    """주석 본문과 docstring — 코드 안 문자열(프롬프트·SQL 등)은 넣지 않는다."""
    tree = ast.parse(source)
    texts = [
        ast.get_docstring(node, clean=False) or ''
        for node in ast.walk(tree)
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    tokens = tokenize.generate_tokens(io.StringIO(source).readline)
    texts += [tok.string.lstrip('#') for tok in tokens if tok.type == tokenize.COMMENT]
    return texts


def _history_counts(files: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for rel in files:
        n = sum(len(HISTORY.findall(text)) for text in _explanations((ROOT / rel).read_text(encoding='utf-8')))
        if n:
            out[rel] = n
    return out


def _compare(kind: str, base: dict[str, int], now: dict[str, int]) -> tuple[list[str], list[str]]:
    """(악화, 개선) — 악화는 새 키이거나 값이 커진 것, 개선은 사라졌거나 값이 작아진 것."""
    worse = [f'{kind} {k}: {base.get(k, 0)} → {v}' for k, v in sorted(now.items()) if v > base.get(k, 0)]
    better = [f'{kind} {k}: {v} → {now.get(k, 0)}' for k, v in sorted(base.items()) if now.get(k, 0) < v]
    return worse, better


def _tightened(base: dict[str, int], now: dict[str, int], allow_growth: bool) -> dict[str, int]:
    keys = set(base) | (set(now) if allow_growth else set())
    merged = {k: (now.get(k, 0) if allow_growth else min(base.get(k, 0), now.get(k, 0))) for k in keys}
    return {k: v for k, v in sorted(merged.items()) if v}


def test_품질_기준선보다_나빠지지_않는다():
    ruff = _ruff_bin()
    files = _files()
    now = {
        'ruff': _ruff_counts(ruff, files),
        'format': {f: 1 for f in _unformatted(ruff, files)},
        'size': _size_violations(files),
        'history': _history_counts(files),
    }
    base = json.loads(BASELINE.read_text(encoding='utf-8')) if BASELINE.exists() else {}
    base = {kind: base.get(kind, {}) for kind in now}

    if _flag('QUALITY_BASELINE_UPDATE'):
        growth = _flag('QUALITY_ALLOW_GROWTH')
        fresh = {kind: _tightened(base[kind], now[kind], growth) for kind in now}
        BASELINE.write_text(json.dumps(fresh, ensure_ascii=False, indent=1, sort_keys=True) + '\n', encoding='utf-8')
        base = fresh

    worse, better = [], []
    for kind in now:
        w, b = _compare(kind, base[kind], now[kind])
        worse += w
        better += b
    if better:
        warnings.warn(
            f'품질 개선 {len(better)}건 — 이 정리를 한 PR이면 기준선을 조여라: '
            'QUALITY_BASELINE_UPDATE=1 pytest tests/test_code_quality.py',
            stacklevel=1,
        )
    assert not worse, (
        '품질 기준선보다 나빠졌다 — 고쳐라(불가피하면 docstring의 QUALITY_ALLOW_GROWTH):\n  ' + '\n  '.join(worse)
    )


def test_래칫_비교는_새_위반과_악화를_잡고_개선을_따로_센다():
    base = {'a.py::F401': 2, 'b.py::C901': 1}
    worse, better = _compare('ruff', base, {'a.py::F401': 3, 'c.py::B904': 1})
    assert worse == ['ruff a.py::F401: 2 → 3', 'ruff c.py::B904: 0 → 1']
    assert better == ['ruff b.py::C901: 1 → 0']


def test_기준선_갱신은_기본적으로_줄이기만_한다():
    base = {'a': 3, 'b': 2}
    now = {'a': 1, 'b': 5, 'c': 1}
    assert _tightened(base, now, allow_growth=False) == {'a': 1, 'b': 2}  # 늘어난 b·새 c는 기준선에 안 들어간다
    assert _tightened(base, now, allow_growth=True) == {'a': 1, 'b': 5, 'c': 1}


def test_코드줄은_주석과_docstring과_빈_줄을_세지_않는다():
    source = 'def f():\n    """설명\n    둘째 줄"""\n    # 주석\n\n    x = 1\n    return x\n'
    assert _code_lines(source, ast.parse(source)) == {1, 6, 7}


def test_함수_목록은_조건부_정의와_중첩을_모두_훑는다():
    source = (
        'class A:\n    def m(self):\n        def g(): pass\n'
        'try:\n    def h(): pass\nexcept ImportError:\n    def h(): pass\n'
        'if True:\n    def k(): pass\n'
    )
    assert [name for name, _ in _functions(ast.parse(source))] == ['A.m', 'A.m.g', 'h', 'h', 'k']


def test_히스토리_표지는_주석과_docstring에서만_센다():
    source = (
        '"""모듈 설명(#123) 2026-10-25 실측."""\n'
        'PROMPT = "#999 이건 문자열이라 안 센다"\n'
        'x = 1  # 기각한 안, 결정 3\n'
        '# 다음 줄 설명 — 번호 없음\n'
        'y = 2  #12 주석 기호 바로 뒤 숫자는 세지 않는다(포매터가 "# "로 고친다)\n'
        'z = 3  # 이 값은 #287에서 정했다(2026-10-25에), PR#141 참고\n'
        'w = 4  # 결정\n'
        '# 3개를 본다\n'
    )
    texts = _explanations(source)
    assert (
        sum(len(HISTORY.findall(t)) for t in texts) == 8
    )  # #123·날짜·실측·기각·결정 3·#287·날짜·PR#141 — 줄 넘은 '결정/3개'는 아님
