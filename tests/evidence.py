"""테스트 증거 — pytest JUnit 리포트를 PR "테스트" 절에 붙일 표로 바꾼다(#311).

사람이 "N건 통과"를 손으로 쓰지 않는다 — 이 출력이 증거다(chat-service `scripts/test-evidence.py`와 같은 발상).
스킵·xfail을 통과와 섞지 않고 **사유별로** 보인다: "ruff 미설치"·"docling 미설치" 같은 스킵이 통과로 묻히던 것을 막는 게 목적이다.

    python -m pytest --junitxml=/tmp/junit.xml
    python -m tests.evidence /tmp/junit.xml

경고: 리포트가 STALE_MINUTES보다 오래됨(이전 실행 결과를 붙이는 사고), 커밋 안 된 변경(새 파일 포함)이 있는 상태에서 돈 결과.
종료 코드 1: 실패·에러가 있거나, 테스트가 0건이거나(전부 선택 해제·수집 붕괴), 리포트를 못 읽을 때.
한계: strict가 아닌 xpass는 pytest가 통과로 적어 통과에 섞인다.
"""

import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STALE_MINUTES = 60
PASS, FAIL, SKIP, XFAIL = '통과', '실패', '스킵', 'xfail'
STATUSES = (PASS, FAIL, SKIP, XFAIL)
SOURCE_DIRS = ('rag/', 'routers/', 'schemas/', 'eval/', 'tests/', '.claude/')  # 미추적 새 파일을 "변경"으로 볼 범위


def _status(case: ET.Element) -> tuple[str, str]:
    """(상태, 사유) — failure·error는 실패, xfail은 pytest가 skipped(type=pytest.xfail)로 적는다."""
    node = case.find('failure')
    if node is None:
        node = case.find('error')
    if node is not None:
        message = node.get('message') or node.tag
        return FAIL, message.splitlines()[0][:120]
    if (node := case.find('skipped')) is not None:
        reason = (node.get('message') or '').removeprefix('Skipped: ')
        return (XFAIL if node.get('type') == 'pytest.xfail' else SKIP), reason
    return PASS, ''


def _file_of(case: ET.Element) -> str:
    """classname(점 경로, 예: 패키지.test_모듈.TestY) → 그 테스트 파일 경로. 수집 오류처럼 비어 있으면 '?'."""
    parts = (case.get('classname') or '?').split('.')
    for i in range(len(parts), 0, -1):
        if parts[i - 1].startswith('test_') or parts[i - 1] == 'conftest':
            return '/'.join(parts[:i]) + '.py'
    return '/'.join(parts)


def summarize(report: Path) -> tuple[str, bool]:
    """(마크다운, 실패 여부)."""
    per_file: dict[str, Counter[str]] = defaultdict(Counter)
    failures: list[str] = []
    skip_reasons: Counter[str] = Counter()
    for case in ET.parse(report).getroot().iter('testcase'):
        status, reason = _status(case)
        name = _file_of(case)
        per_file[name][status] += 1
        if status == FAIL:
            failures.append(f'`{name}::{case.get("name")}` — {reason}')
        elif status in (SKIP, XFAIL):
            skip_reasons[f'{status}: {reason or "(사유 없음)"}'] += 1
    empty = not per_file
    return _render(report, per_file, failures, skip_reasons), bool(failures) or empty


def _git(*args: str) -> str:
    try:
        return subprocess.run(['git', *args], cwd=ROOT, capture_output=True, text=True).stdout
    except FileNotFoundError:
        return ''


def _header(report: Path, files: int) -> list[str]:
    age = (time.time() - report.stat().st_mtime) / 60
    commit = _git('rev-parse', '--short', 'HEAD').strip() or '?'
    changed = [line for line in _git('status', '--porcelain').splitlines() if not line.startswith('??')]
    changed += [
        line
        for line in _git('status', '--porcelain').splitlines()
        if line.startswith('?? ') and line[3:].startswith(SOURCE_DIRS)
    ]
    lines = [f'커밋 `{commit}{" + 미커밋 변경" if changed else ""}` · 리포트 {age:.0f}분 전 · 파일 {files}개']
    if age > STALE_MINUTES:
        lines.append(f'> ⚠️ 리포트가 {age:.0f}분 전 것이다 — 지금 코드로 다시 돌린 결과인지 확인하라')
    if not files:
        lines.append('> ⚠️ 테스트가 0건이다 — 선택 해제(-k)나 수집 붕괴를 확인하라')
    return lines


def _render(report: Path, per_file: dict[str, Counter[str]], failures: list[str], skip_reasons: Counter[str]) -> str:
    total: Counter[str] = sum(per_file.values(), Counter())
    lines = _header(report, len(per_file))
    lines += ['', '| 구분 | ' + ' | '.join(STATUSES) + ' |', '|---|' + '---|' * len(STATUSES)]
    lines.append('| **합계** | ' + ' | '.join(str(total[s]) for s in STATUSES) + ' |')
    for name in sorted(n for n, c in per_file.items() if c[FAIL] or c[SKIP]):  # 표에는 눈여겨볼 파일만
        lines.append(f'| `{name}` | ' + ' | '.join(str(per_file[name][s]) for s in STATUSES) + ' |')
    if failures:
        lines += ['', f'**실패 {len(failures)}건**', *[f'- {f}' for f in failures]]
    if skip_reasons:
        lines += ['', '**스킵·xfail 사유** — 통과가 아니다', *[f'- {r} ({n}건)' for r, n in skip_reasons.most_common()]]
    return '\n'.join(lines)


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print('사용: python -m tests.evidence <junit.xml>')
        return 2
    try:
        text, failed = summarize(Path(argv[1]))
    except (OSError, ET.ParseError) as e:
        print(f'리포트를 읽을 수 없다({e}) — pytest가 중단됐으면 다시 돌린다')
        return 1
    print(text)
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
