"""rag/chunking 패키지 경계 — 하위 모듈로 옮긴 이름을 옛 경로(`rag.chunking.X`)로 패치하는 테스트·eval을 막는다.

`rag.chunking`은 옮긴 이름을 re-export하므로 **읽기**는 그대로 되지만, **패치**는 `__init__`의 바인딩만 바꾸고 실제 호출처
(하위 모듈 전역)에는 닿지 않아 조용히 무효가 된다. 그래서 패치는 이름이 정의된 모듈에 해야 한다.
"""

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PKG = ROOT / 'rag' / 'chunking'


def _reexported() -> dict[str, str]:
    """`__init__`이 하위 모듈에서 들여온 이름 → 정의 모듈."""
    tree = ast.parse((PKG / '__init__.py').read_text(encoding='utf-8'))
    out: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith('rag.chunking.'):
            for a in node.names:
                out[a.asname or a.name] = node.module
    return out


def _chunking_aliases(src: str) -> set[str]:
    """`import rag.chunking as X` / `from rag import chunking as X` 꼴의 별칭 — 패치 대상 표현식의 머리."""
    names = set(re.findall(r'import rag\.chunking as (\w+)', src))
    names |= set(re.findall(r'from rag import chunking(?: as (\w+))?', src)) - {''}
    if re.search(r'from rag import chunking\b(?! as)', src):
        names.add('chunking')
    return names


def _old_path_patches(src: str, aliases: set[str], moved: dict[str, str]) -> list[str]:
    hits = []
    for alias in aliases | {'rag.chunking'}:
        a = re.escape(alias)
        for name in re.findall(rf'setattr\(\s*{a}\s*,\s*[\'"](\w+)[\'"]', src):
            if name in moved:
                hits.append(f'setattr({alias}, {name!r}) → {moved[name]}에 패치하라')
        for name in re.findall(rf'(?m)^\s*{a}\.(\w+)\s*=[^=]', src):
            if name in moved:
                hits.append(f'{alias}.{name} = … → {moved[name]}에 대입하라')
    return hits


def test_옮긴_이름은_옛_경로로_패치하지_않는다() -> None:
    moved = _reexported()
    assert moved, '__init__이 하위 모듈에서 들여온 이름이 없다 — 패키지 분리 상태를 확인하라'
    bad = []
    for f in [*(ROOT / 'tests').glob('*.py'), *(ROOT / 'eval').rglob('*.py')]:
        src = f.read_text(encoding='utf-8')
        aliases = _chunking_aliases(src)
        for hit in _old_path_patches(src, aliases, moved):
            bad.append(f'{f.relative_to(ROOT)}: {hit}')
    assert not bad, '\n'.join(bad)


def _bound_directly() -> dict[str, dict[str, set[str]]]:
    """정의 모듈 → {이름: 그 이름을 `from rag.chunking.X import 이름`으로 직접 바인딩해 **쓰는** 패키지 모듈들}.

    바인딩한 모듈은 자기 전역을 읽으므로 정의 모듈에 패치해도 닿지 않는다 — 그런 이름은 소비 모듈의 전역을 패치해야 한다.
    `__init__`이 `__all__`용으로 재수출만 하는 바인딩은 소비가 아니라 세지 않는다.
    """
    out: dict[str, dict[str, set[str]]] = {}
    for f in PKG.glob('*.py'):
        tree = ast.parse(f.read_text(encoding='utf-8'))
        used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}  # import·__all__ 문자열은 Name이 아니다
        imports = [
            n for n in tree.body if isinstance(n, ast.ImportFrom) and n.module and n.module.startswith('rag.chunking.')
        ]
        for node in imports:
            sub = str(node.module).rsplit('.', 1)[1]
            for name in [a.asname or a.name for a in node.names if (a.asname or a.name) in used]:
                out.setdefault(sub, {}).setdefault(name, set()).add(f.stem)
    return out


def _submodule_aliases(src: str, pkg_aliases: set[str], subs: set[str]) -> dict[str, str]:
    """패치 대상 표현식 → 하위 모듈 이름. `from rag.chunking import split as S`·`import rag.chunking.split as S`·`ch.split`·`rag.chunking.split`."""
    out: dict[str, str] = {}
    for sub in subs:
        for head in pkg_aliases | {'rag.chunking'}:
            out[f'{head}.{sub}'] = sub
        for m in re.finditer(rf'import rag\.chunking\.{sub} as (\w+)', src):
            out[m.group(1)] = sub
        for m in re.finditer(rf'from rag\.chunking import ([^\n]*)\b{sub}\b(?: as (\w+))?', src):
            out[m.group(2) or sub] = sub
    return out


def _defining_module_patches(src: str, aliases: dict[str, str], bound: dict[str, dict[str, set[str]]]) -> list[str]:
    hits = []
    for expr, sub in aliases.items():
        a = re.escape(expr)
        names = re.findall(rf'setattr\(\s*{a}\s*,\s*[\'"](\w+)[\'"]', src) + re.findall(
            rf'(?m)^\s*{a}\.(\w+)\s*=[^=]', src
        )
        for name in names:
            users = bound.get(sub, {}).get(name)
            if users:
                where = '·'.join(f'rag.chunking.{u}' if u != '__init__' else 'rag.chunking' for u in sorted(users))
                hits.append(f'{expr}.{name} 패치는 {where}에 닿지 않는다(직접 바인딩) — 소비 모듈의 전역을 패치하라')
    return hits


def test_직접_바인딩된_이름은_정의_모듈에_패치하지_않는다() -> None:
    bound = _bound_directly()
    assert bound, '하위 모듈 사이의 직접 바인딩이 없다 — 패키지 분리 상태를 확인하라'
    bad = []
    for f in [*(ROOT / 'tests').glob('*.py'), *(ROOT / 'eval').rglob('*.py')]:
        src = f.read_text(encoding='utf-8')
        aliases = _submodule_aliases(src, _chunking_aliases(src), set(bound))
        for hit in _defining_module_patches(src, aliases, bound):
            bad.append(f'{f.relative_to(ROOT)}: {hit}')
    assert not bad, '\n'.join(bad)


def test_하위_모듈은_서로를_절대_경로로만_쓴다() -> None:
    """상대 import는 계층 검사(test_architecture)를 우회한다."""
    for f in PKG.glob('*.py'):
        for node in ast.walk(ast.parse(f.read_text(encoding='utf-8'))):
            if isinstance(node, ast.ImportFrom):
                assert node.level == 0, f'{f.name}:{node.lineno} 상대 import'
