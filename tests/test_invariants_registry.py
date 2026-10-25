"""불변식 대장(INVARIANTS.md, #312)이 낡지 않았는지 — 강제 칸의 테스트가 실제로 정의돼 있고, ID가 겹치지 않고, 정의점 파일이 있다.

대장의 가치는 "규칙 ↔ 강제 테스트" 연결이다. 테스트가 이름을 바꾸거나 지워지면 연결이 조용히 끊기므로 여기서 잡는다.
이름이 있다고 의미가 맞는 것은 아니다 — 의미는 등록할 때 사람이(리뷰가) 본다. 이 검사는 존재만 본다.
"""

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REGISTRY = ROOT / 'INVARIANTS.md'
TEST_REF = re.compile(r'`(tests/[\w/]+\.py)::([^`\s]+)`')
ROW = re.compile(r'^\| (INV-\d+) \| [^|\n]+ \| `([^`]+)` \| ([^|\n]*) \|', re.M)
FRAGMENT = re.compile(r'^`tests/[\w/]+\.py::[^`\s]+`$')


def _defined(path: Path) -> set[str]:
    """그 파일에 정의된 테스트 경로 — 'test_x'와 'TestY::test_x'."""
    names: set[str] = set()
    for node in ast.parse(path.read_text(encoding='utf-8')).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith('test_'):
            names.add(node.name)
        elif isinstance(node, ast.ClassDef) and node.name.startswith('Test'):
            names |= {
                f'{node.name}::{m.name}'
                for m in node.body
                if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)) and m.name.startswith('test_')
            }
    return names


def _rows() -> list[tuple[str, str, str]]:
    text = REGISTRY.read_text(encoding='utf-8')
    rows = ROW.findall(text)
    lines = len(re.findall(r'^\| INV-', text, re.M))
    assert rows and len(rows) == lines, (
        f'INV 행 {lines}개 중 {len(rows)}개만 파싱됐다 — 칸 안의 |·정의점 형식을 확인하라'
    )
    return rows


def test_강제_칸의_테스트가_실제로_있다():
    cache: dict[str, set[str]] = {}
    missing = []
    for inv, _, enforced in _rows():
        for file, name in TEST_REF.findall(enforced):
            path = ROOT / file
            if not path.is_file():
                missing.append(f'{inv}: {file} 없음')
                continue
            if name not in cache.setdefault(file, _defined(path)):
                missing.append(f'{inv}: {file}::{name}')
    assert not missing, f'INVARIANTS.md가 가리키는 테스트가 없다(이름 변경·삭제?): {missing}'


def test_ID는_겹치지_않고_정의점_파일이_있다():
    rows = _rows()
    ids = [inv for inv, _, _ in rows]
    assert len(ids) == len(set(ids)), [i for i in ids if ids.count(i) > 1]
    gone = [f'{inv}: {where}' for inv, where, _ in rows if not (ROOT / where).exists()]
    assert not gone, f'정의점 파일이 없다: {gone}'


def test_강제_칸은_테스트_목록이거나_대시다():
    """'`a` · `b` (부분)' 꼴 — 조각마다 형식을 본다(유효한 참조 하나에 오타 난 참조가 섞여 조용히 빠지지 않게)."""
    bad = []
    for inv, _, enforced in _rows():
        body = enforced.strip().removesuffix('(부분)').strip()
        if body == '—':
            continue
        bad += [f'{inv}: {frag}' for frag in body.split(' · ') if not FRAGMENT.match(frag.strip())]
    assert not bad, f'강제 칸은 `tests/…::이름`을 \' · \'로 잇거나 — 여야 한다: {bad}'
