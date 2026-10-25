"""타입 검사 게이트(#308) — 판단 기준 5(AGENTS.md "코드 품질 기준")의 새 파일 몫. 공개 시그니처 힌트 래칫은 ruff ANN(pyproject.toml).

기존 파일은 tests/typecheck_baseline.json에 동결하고, **목록 밖 파일(= 새 파일)은 mypy 오류 0**이어야 한다(설정은 pyproject.toml
[tool.mypy] — 함수 정의에 힌트 필수). 기존 코드에 그대로 켜면 rag/ 51·eval/ 163 오류가 나서(main 31bece0) 리팩토링 뒤로 미룬다.
리팩토링으로 기존 파일이 깨끗해지면 목록에서 빼서 조인다:
    QUALITY_BASELINE_UPDATE=1 pytest tests/test_typecheck.py
갱신은 목록을 줄이기만 한다(사라진 파일·이제 오류 0인 파일을 뺀다). 처음 만들 때만 QUALITY_ALLOW_GROWTH=1로 현재 파일 전부를 넣는다.
파일 이름을 바꾸거나 옮기면 목록 밖이 되어 새 파일로 검사된다(#304에서 쪼개 생기는 모듈과 같은 취급). 반대로 기존 파일에 붙여 넣은
코드는 이 검사를 피한다 — 그 몫은 ruff ANN 래칫(시그니처)뿐이다. 새 파일이 부르는 기존 모듈은 힌트가 없어 Any로 흐른다
(follow_imports=silent) — 타입은 새 파일 안에서만 닫히고, 기존 코드가 깨끗해질수록(#308 2단계) 강해진다.
대상은 tests/test_code_quality.py와 같은 git 기준 파일 목록에서 tests/·루트 conftest.py를 뺀 것 — 테스트 함수는 관례상 힌트를 달지 않는다.
mypy는 requirements.txt에 버전 고정 — 없으면 건너뛰되 CI=1이면 실패, 버전이 다르면 실패.
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
BASELINE = Path(__file__).resolve().parent / 'typecheck_baseline.json'
TARGETS = ['rag', 'routers', 'schemas', 'eval', 'main.py', 'config.py', 'database.py', 'text_norm.py']
ERROR_LINE = re.compile(r'^(?P<path>[^:\s]+\.py):(?:\d+:)? error:', re.M)  # 줄 번호 없는 오류(모듈 경로 충돌 등)도 센다


def _flag(name: str) -> bool:
    return os.environ.get(name, '').strip().lower() in ('1', 'true', 'yes')


def _pinned_version() -> str:
    for line in (ROOT / 'requirements.txt').read_text(encoding='utf-8').splitlines():
        if line.startswith('mypy=='):
            return line.split('==', 1)[1].split()[0]
    raise AssertionError('requirements.txt에 mypy==버전 줄이 없다')


def _require_mypy() -> None:
    try:
        import mypy.version
    except ImportError:
        if _flag('CI'):
            pytest.fail('mypy 미설치 — pip install -r requirements.txt')
        pytest.skip('mypy 미설치 — pip install -r requirements.txt (CI에서는 실패)')
    if mypy.version.__version__ != (pinned := _pinned_version()):
        pytest.fail(f'mypy {mypy.version.__version__} ≠ 고정 {pinned} — 판정이 버전마다 달라 기준선과 비교할 수 없다')


def _files() -> set[str]:
    out = subprocess.run(
        ['git', 'ls-files', '--cached', '--others', '--exclude-standard', '--', *TARGETS],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    return {f for f in out if f.endswith('.py') and (ROOT / f).is_file()}


def _errors_by_file(files: list[str]) -> tuple[dict[str, int], str]:
    """({파일: 오류 수}, mypy 출력). 오류가 없는 파일은 키가 없다."""
    proc = subprocess.run(
        [sys.executable, '-m', 'mypy', '--no-error-summary', *files], cwd=ROOT, capture_output=True, text=True
    )
    if proc.returncode not in (0, 1):  # 2 = 설정·실행 오류 — 조용히 0건으로 넘기지 않는다
        pytest.fail(f'mypy 실행 실패: {proc.stdout}{proc.stderr}')
    counts = _count_errors(proc.stdout)
    if proc.returncode == 1 and not counts:  # 오류라는데 집계가 비면 출력 형식이 바뀐 것 — 조용히 통과시키지 않는다
        pytest.fail(f'mypy가 오류를 냈는데 집계하지 못했다(출력 형식 확인):\n{proc.stdout}{proc.stderr}')
    return counts, proc.stdout


def _count_errors(output: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for m in ERROR_LINE.finditer(output):
        counts[m['path']] = counts.get(m['path'], 0) + 1
    return counts


def test_새_파일은_mypy_오류가_없다():
    _require_mypy()
    files = _files()
    baseline = set(json.loads(BASELINE.read_text(encoding='utf-8'))) if BASELINE.exists() else set()

    if _flag('QUALITY_BASELINE_UPDATE'):
        if _flag('QUALITY_ALLOW_GROWTH'):
            baseline = files
        else:
            kept = sorted(baseline & files)
            errors, _ = _errors_by_file(kept) if kept else ({}, '')
            baseline = {f for f in kept if errors.get(f)}  # 사라진 파일·이제 깨끗한 파일을 뺀다
        BASELINE.write_text(json.dumps(sorted(baseline), ensure_ascii=False, indent=1) + '\n', encoding='utf-8')

    new = sorted(files - baseline)
    if not new:
        return
    errors, output = _errors_by_file(new)
    assert not errors, f'새 파일은 타입 검사를 통과해야 한다(판단 기준 5, #308):\n{output}'


def test_오류_집계는_파일별로_센다():
    out = 'pkg/d.py: error: Source file found twice\npkg/a.py:3: error: x  [arg-type]\npkg/a.py:9: error: y\npkg/b.py:1: note: z\npkg/c.py:2: error: w\n'
    assert _count_errors(out) == {'pkg/d.py': 1, 'pkg/a.py': 2, 'pkg/c.py': 1}
