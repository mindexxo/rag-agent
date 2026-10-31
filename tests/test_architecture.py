"""의존 방향(계층) 검사(#310) — AGENTS.md "아키텍처" 절이 정의점이고, 이 테스트가 그 문장을 코드와 대조한다.

검사 단위는 rag/의 모듈 하나(`rag/cache.py` → `cache`)이고, 하위 패키지는 통째로 한 단위(`rag/llm/` → `llm`)다.
기준선이 없는 하드 규칙이다 — 도입 시점(main 211d5d2)에 전부 통과했다. 실패하면 코드를 고치거나, 의도한 변화면
AGENTS.md 문장을 고친다(문서가 정의점이므로 문서를 고치는 것이 곧 결정의 기록이다).

  ① rag/·config.py는 routers/·eval/을 import하지 않는다(함수 안 포함). config.py는 프로젝트 모듈을 import하지 않는다.
  ② rag/의 모든 단위가 AGENTS.md 계층 절에 나온다 — 새 모듈이 어느 층인지 정하지 않고 들어오지 못하게.
  ③ 0층 단위의 톱레벨 rag import는 AGENTS.md가 적은 예외뿐이다.
  ④ 진입점(rag/streaming.py·rag/worker.py)을 다른 rag 모듈이 톱레벨에서 import하지 않는다(지연은 ⑤의 문서화된 목록만).
  ④′ 조립점(rag/service.py)을 아는 rag 모듈은 진입점뿐이다 — '아래에서 위로만'.
  ⑤ 함수 안 지연 import(rag 모듈 대상)는 AGENTS.md 목록과 정확히 같다 — 지연 import는 순환을 숨기므로 늘어날 때 보이게.
  ⑥ rag 톱레벨 import에 순환이 없다. 상대 import는 금지(검사를 우회한다). 파싱이 비면 조용히 통과하지 않게 최소 가드를 둔다.
"""

import ast
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAG = ROOT / 'rag'
LAYERS = ('0층', '중간층', '조립점', '진입점')
PROJECT = {'rag', 'routers', 'schemas', 'eval', 'database', 'text_norm', 'main', 'config'}


def _unit(path: Path) -> str:
    rel = path.relative_to(RAG).parts
    return rel[0] if len(rel) > 1 else path.stem


def _imported_modules(node: ast.AST) -> list[str]:
    """import 문 하나가 가리키는 모듈 경로들. `from rag import x`는 `rag.x`로 편다."""
    if isinstance(node, ast.Import):
        return [a.name for a in node.names]
    if isinstance(node, ast.ImportFrom) and node.level > 0:
        raise AssertionError(
            f'상대 import(줄 {node.lineno}) — 이 저장소는 절대 import만 쓴다. 상대 import는 이 검사를 우회한다'
        )
    if isinstance(node, ast.ImportFrom) and node.module:
        if node.module == 'rag':
            return [f'rag.{a.name}' for a in node.names]
        return [node.module]
    return []


def _imports(tree: ast.Module) -> tuple[list[str], list[str]]:
    """(톱레벨, 함수 안) import 모듈 경로 — 모듈 수준 if/try 안은 톱레벨로 센다."""
    top: list[str] = []
    lazy: list[str] = []

    def visit(node: ast.AST, in_func: bool) -> None:
        for child in ast.iter_child_nodes(node):
            is_func = isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda))
            (lazy if in_func else top).extend(_imported_modules(child))
            visit(child, in_func or is_func)

    visit(tree, False)
    return top, lazy


def _rag_unit(module: str) -> str | None:
    parts = module.split('.')
    return parts[1] if parts[0] == 'rag' and len(parts) > 1 else None


def _graph() -> tuple[dict[str, set[str]], dict[str, set[str]], dict[str, list[str]]]:
    """(톱레벨 간선, 지연 간선, 단위별 전체 import 경로). 자기 단위 안의 import(rag/llm 내부)는 간선이 아니다."""
    top: dict[str, set[str]] = defaultdict(set)
    lazy: dict[str, set[str]] = defaultdict(set)
    every: dict[str, list[str]] = defaultdict(list)
    for path in sorted(RAG.rglob('*.py')):
        unit = _unit(path)
        t, z = _imports(ast.parse(path.read_text(encoding='utf-8')))
        every[unit] += t + z
        for edges, mods in ((top, t), (lazy, z)):
            edges[unit].update(u for m in mods if (u := _rag_unit(m)) and u != unit)
    return top, lazy, every


def _agents_section() -> str:
    text = (ROOT / 'AGENTS.md').read_text(encoding='utf-8')
    start = text.index('## 아키텍처')
    return text[start : text.index('\n## ', start + 1)]


def _bullet(section: str, label: str) -> str:
    """'- **라벨' 불릿 하나 — 다음 불릿이나 빈 줄(문단 끝)에서 끊는다."""
    start = section.index(f'- **{label}')
    ends = [e for e in (section.find('\n- ', start + 1), section.find('\n\n', start)) if e > 0]
    return section[start : min(ends, default=len(section))]


def _units_in(text: str) -> set[str]:
    return set(re.findall(r'`rag/(\w+)(?:\.py|/)', text))


def _documented_lazy(section: str) -> dict[str, set[str]]:
    """'`a`·`b`(→x, y)' 꼴을 {a: {x, y}, b: {x, y}}로."""
    start = section.index('이걸 하는 모듈:')
    block = section[start : section.index('\n순환 회피가 아닌', start)]  # 마커가 사라지면 조용히 넓어지지 않고 실패한다
    out: dict[str, set[str]] = defaultdict(set)
    for names, targets in re.findall(r'((?:`\w+`·?)+)\(→([^)]+)\)', block):
        for name in re.findall(r'`(\w+)`', names):
            out[name] |= {t.strip() for t in targets.split(',')}
    return dict(out)


def test_rag와_config는_역방향으로_import하지_않는다():
    _, _, every = _graph()
    bad = [f'rag/{u}: {m}' for u, mods in every.items() for m in mods if m.split('.')[0] in ('routers', 'eval')]
    t, z = _imports(ast.parse((ROOT / 'config.py').read_text(encoding='utf-8')))
    bad += [f'config.py: {m}' for m in t + z if m.split('.')[0] in PROJECT]
    assert not bad, bad


def test_rag의_모든_모듈이_계층_절에_있다():
    actual = {_unit(p) for p in RAG.rglob('*.py')}
    section = _agents_section()
    layered = set().union(*(_units_in(_bullet(section, label)) for label in LAYERS))
    missing = sorted(actual - layered)
    assert not missing, f'AGENTS.md "아키텍처" 절에 층을 정해 적어라: {missing}'


def test_0층은_문서화된_예외_말고_rag를_톱레벨에서_쓰지_않는다():
    section = _agents_section()
    leaf_block = _bullet(section, '0층')
    exceptions = {
        m: _units_in(rest) for m, rest in re.findall(r'예외[^:]*: `rag/(\w+)(?:\.py|/)`는([^\n]*)', leaf_block)
    }
    assert {'models', 'tokens', 'llm'} <= _units_in(leaf_block), '0층 불릿 파싱 실패 — 표기(백틱·rag/ 경로)를 확인하라'
    top, _, _ = _graph()
    bad = {
        u: sorted(top[u] - exceptions.get(u, set())) for u in _units_in(leaf_block) if top[u] - exceptions.get(u, set())
    }
    assert not bad, f'0층이 rag 모듈을 톱레벨에서 쓴다(층을 옮기거나 AGENTS.md 예외에 적어라): {bad}'


def test_진입점은_다른_rag_모듈이_톱레벨에서_import하지_않는다():
    entry = _units_in(_bullet(_agents_section(), '진입점'))
    assert entry == {'streaming', 'worker'}, f'진입점 불릿 파싱 결과가 다르다: {entry}'
    top, _, _ = _graph()
    bad = sorted(f'{u} → {v}' for u, vs in top.items() for v in vs & entry)
    assert not bad, bad


def test_조립점은_진입점만_import한다():
    """'의존은 아래에서 위로만' — 조립점(service)을 아는 rag 모듈은 진입점뿐이다(지연 포함)."""
    section = _agents_section()
    hub = _units_in(_bullet(section, '조립점'))
    entry = _units_in(_bullet(section, '진입점'))
    assert hub == {'service'}, f'조립점 불릿 파싱 결과가 다르다: {hub}'
    top, lazy, _ = _graph()
    bad = sorted(f'{u} → {v}' for g in (top, lazy) for u, vs in g.items() for v in vs & hub if u not in entry)
    assert not bad, bad


def test_지연_import는_AGENTS_목록과_같다():
    _, lazy, _ = _graph()
    actual = {u: vs for u, vs in lazy.items() if vs}
    assert actual == _documented_lazy(_agents_section()), (
        '함수 안 rag import가 AGENTS.md "이걸 하는 모듈" 목록과 다르다 — 순환을 숨기는 지연 import는 문서에 이유와 함께 적는다'
    )


def test_rag_톱레벨_import에_순환이_없다():
    top, _, _ = _graph()
    assert not (cycle := _find_cycle(top)), f'톱레벨 순환: {" → ".join(cycle)}'


def _find_cycle(edges: dict[str, set[str]]) -> list[str]:
    """첫 순환 경로 하나(없으면 빈 목록) — 방문 중(경로 위) 노드로 돌아오는 간선을 찾는 DFS."""
    done: set[str] = set()

    def dfs(node: str, path: list[str]) -> list[str]:
        for nxt in sorted(edges.get(node, ())):
            if nxt in path:
                return path[path.index(nxt) :] + [nxt]
            if nxt not in done and (found := dfs(nxt, path + [nxt])):
                return found
        done.add(node)
        return []

    for start in sorted(edges):
        if start not in done and (found := dfs(start, [start])):
            return found
    return []


def test_검사기_자체_파싱():
    tree = ast.parse(
        'import a\nfrom rag import x, y\nif T:\n    from rag.z import q\ndef f():\n    from rag.w import r\n'
    )
    assert _imports(tree) == (['a', 'rag.x', 'rag.y', 'rag.z'], ['rag.w'])
    lazy = _documented_lazy('이걸 하는 모듈: `a`(→b), `c`·`d`(→e, f).\n순환 회피가 아닌')
    assert lazy == {'a': {'b'}, 'c': {'e', 'f'}, 'd': {'e', 'f'}}
    assert _find_cycle({'a': {'b'}, 'b': {'c'}, 'c': {'a'}}) == ['a', 'b', 'c', 'a']
    assert _find_cycle({'a': {'b'}, 'b': set()}) == []
    try:
        _imported_modules(ast.parse('from .x import y').body[0])
    except AssertionError:
        pass
    else:
        raise AssertionError('상대 import가 통과했다')
