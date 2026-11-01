"""불변식 대장(INVARIANTS.md)의 부분 칸 보강 — 코드를 읽어 판정하는 정적 검사(DB·OpenSearch 없음).

각 테스트는 INV 번호로 시작한다. 탐지기를 직접 짠 검사는 합성 소스로 "위반을 실제로 잡는다"를 같이 단언한다
(탐지기가 무뎌져 조용히 통과하는 것을 막는다).
"""

import ast
import json
import re
import tomllib
from collections import Counter
from pathlib import Path

import pytest

from rag.models import Base

ROOT = Path(__file__).resolve().parent.parent


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding='utf-8'), filename=str(path))


def _rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def _py_files(*dirs: str) -> list[Path]:
    return [p for d in dirs for p in sorted((ROOT / d).rglob('*.py'))]


def _names(node: ast.AST) -> set[str]:
    """노드 안의 이름·속성 이름 전부."""
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)} | {
        n.attr for n in ast.walk(node) if isinstance(n, ast.Attribute)
    }


# ── INV-01 테넌트 조건 정적 검사 ──────────────────────────────────────────────────────────────────

# tenant_id 컬럼이 있는 매핑 클래스 — 모델이 늘면 자동으로 따라온다.
TENANT_MODELS = {m.class_.__name__ for m in Base.registry.mappers if 'tenant_id' in m.columns}
TENANT_TABLES = {m.local_table.name for m in Base.registry.mappers if 'tenant_id' in m.columns}
SCOPE_HELPERS = {'owned_filter'}  # tenant 조건을 담은 WHERE 표현식을 돌려주는 함수

# tenant 조건 없이 테넌트 테이블을 읽거나 쓰는 질의가 있는 함수 — 이유가 있는 것만. 새 항목은 리뷰 대상이다.
UNSCOPED_ALLOWED: dict[str, str] = {
    'rag/documents.py::retire_documents': '호출부가 테넌트로 확정한 id 목록만 받는다',
    'rag/documents.py::_reuse_failed_row': '같은 함수 호출부가 테넌트로 찾은 행 id',
    'rag/ingestion.py::index_pending_document': 'outbox 행의 id — 등재 시점에 테넌트로 확정됐다',
    'rag/ingestion.py::_load_index_target': 'outbox 행의 id — 등재 시점에 테넌트로 확정됐다',
    'rag/ingestion.py::_commit_ready': '이미 읽은 문서의 폴더 id',
    'rag/outbox.py::_set_row': 'outbox 행 id로만 갱신(워커 내부)',
    'rag/outbox.py::_apply': '처리 시점 상태 확인 — payload id는 등재 시점에 테넌트로 확정됐다',
    'rag/outbox.py::_record_failure': 'outbox 행 id로만 갱신(워커 내부)',
    'rag/outbox.py::_mark_terminal': 'outbox 행 id로만 갱신(워커 내부)',
    'rag/outbox.py::pending_row_ids': '테스트·eval 전용 행 조회',
    'rag/search/os_index.py::sync_meta_documents_now': 'META payload id — 호출부 사전 스코프에 기댄다(표면 테스트가 덮는다)',
    'rag/search/os_index.py::sync_meta_faqs_now': 'META payload id — 호출부 사전 스코프에 기댄다',
    'rag/turn/turn_state.py::sweep_stale_generating': '전역 치유 — 상태만 바꾸고 아무것도 노출하지 않는다',
}


def _is_tenant_query(call: ast.Call) -> bool:
    """select/update/delete(테넌트 모델…) 또는 x.get(테넌트 모델, id)."""
    f = call.func
    if isinstance(f, ast.Name) and f.id in ('select', 'update', 'delete'):
        return bool(_names(ast.Tuple(elts=list(call.args), ctx=ast.Load())) & TENANT_MODELS)
    return (
        isinstance(f, ast.Attribute)
        and f.attr == 'get'
        and bool(call.args)
        and isinstance(call.args[0], ast.Name)
        and call.args[0].id in TENANT_MODELS
    )


def _scoped_vars(fn: ast.AST) -> set[str]:
    """tenant 조건을 담은 식을 대입받은 지역 이름(where = owned_filter(...) 등) — 고정점까지."""
    found: set[str] = set()
    assigns = [n for n in ast.walk(fn) if isinstance(n, ast.Assign)]
    while True:
        new = {
            n.id
            for a in assigns
            if _names(a.value) & ({'tenant_id'} | SCOPE_HELPERS | found)
            for t in a.targets
            for n in ast.walk(t)
            if isinstance(n, ast.Name)
        } - found
        if not new:
            return found
        found |= new


def _unscoped_queries(tree: ast.Module) -> list[tuple[str, int]]:
    """(함수 이름, 줄) — 질의 식(체인 전체)에도, 그 결과를 대입받은 이름을 다시 쓰는 뒤 문장에도 tenant 조건이 없는 곳."""
    out: list[tuple[str, int]] = []
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        parent = {id(c): p for p in ast.walk(fn) for c in ast.iter_child_nodes(p)}
        marks = {'tenant_id'} | SCOPE_HELPERS | _scoped_vars(fn)
        stmts = [n for n in ast.walk(fn) if isinstance(n, ast.stmt) and n is not fn]
        for call in (c for c in ast.walk(fn) if isinstance(c, ast.Call) and _is_tenant_query(c)):
            expr: ast.AST = call
            while not isinstance(parent.get(id(expr)), ast.stmt):
                expr = parent[id(expr)]
            stmt = parent[id(expr)]
            if _names(expr) & marks:
                continue
            target = stmt.targets[0] if isinstance(stmt, ast.Assign) else None
            if isinstance(target, ast.Name) and any(
                s.lineno > stmt.lineno and target.id in _names(s) and _names(s) & marks for s in stmts
            ):
                continue
            out.append((fn.name, call.lineno))
    return out


def test_INV01_테넌트_테이블_질의는_tenant_조건을_건다():
    """INV-01 테넌트 격리는 WHERE 절로만 보장한다 — rag/·routers/의 테넌트 테이블 질의(select·update·delete·get)마다
    tenant 조건이 질의 식이나 그 결과를 이어 쓰는 문장에 있어야 한다. 없는 곳은 UNSCOPED_ALLOWED에 사유와 함께만."""
    assert {'Document', 'Message', 'Faq', 'Conversation'} <= TENANT_MODELS, TENANT_MODELS
    files = _py_files('rag', 'routers')
    found = {f'{_rel(p)}::{fn}': line for p in files for fn, line in _unscoped_queries(_tree(p))}
    assert found, '허용 목록 항목조차 못 찾았다 — 탐지기가 질의를 못 읽는다'
    new = {k: v for k, v in found.items() if k not in UNSCOPED_ALLOWED}
    assert not new, f'tenant 조건 없는 테넌트 테이블 질의(.where(Model.tenant_id == tenant_id)를 걸어라): {new}'
    stale = sorted(set(UNSCOPED_ALLOWED) - set(found))
    assert not stale, f'UNSCOPED_ALLOWED에 더는 해당하지 않는 항목이 남았다 — 지워라: {stale}'


def _sql_literals(tree: ast.Module) -> list[tuple[str, int]]:
    """text()/sql_text()에 넘긴 SQL 문자열(f-string은 상수 조각만)."""
    out = []
    for call in ast.walk(tree):
        if not (
            isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id in ('text', 'sql_text')
        ):
            continue
        if not call.args:
            continue
        arg = call.args[0]
        parts = [arg] if isinstance(arg, ast.Constant) else getattr(arg, 'values', [])
        out.append(
            (''.join(p.value for p in parts if isinstance(p, ast.Constant) and isinstance(p.value, str)), call.lineno)
        )
    return out


def test_INV01_원시_SQL도_테넌트_테이블을_읽으면_tenant_id를_건다():
    """INV-01 원시 SQL(text) 경로 — 테넌트 테이블 이름이 나오는 문장은 `:tenant_id` 바인드를 쓴다."""
    found = [(p, s, line) for p in _py_files('rag', 'routers') for s, line in _sql_literals(_tree(p))]
    assert len(found) >= 3, f'원시 SQL을 {len(found)}개만 찾았다 — 탐지기를 확인하라'
    table_re = re.compile(r'\b(?:FROM|JOIN)\s+(' + '|'.join(sorted(TENANT_TABLES)) + r')\b', re.I)
    bad = [f'{_rel(p)}:{line}' for p, s, line in found if table_re.search(s) and ':tenant_id' not in s]
    assert bad == [], f'테넌트 테이블을 읽는 원시 SQL에 :tenant_id 조건이 없다: {bad}'


@pytest.mark.parametrize(
    ('src', 'caught'),
    [
        ('async def f(s, i):\n    return await s.execute(select(Document).where(Document.id == i))', True),
        ('async def f(s, i):\n    return await s.get(Message, i)', True),
        (
            'async def f(s, t, i):\n    stmt = select(Faq).where(Faq.id == i)\n    stmt = stmt.where(Faq.tenant_id == t)',
            False,
        ),
        (
            'async def f(s, o):\n    w = owned_filter(*o)\n    return await s.execute(select(Conversation).where(w))',
            False,
        ),
        (
            'async def f(s, t, i):\n    m = await s.get(Message, i)\n    if m.tenant_id != t:\n        return None',
            False,
        ),
    ],
)
def test_INV01_탐지기는_조건_없는_질의를_잡고_스코프된_질의는_통과시킨다(src, caught):
    assert bool(_unscoped_queries(ast.parse(src))) is caught


# ── INV-79 구조 결정 필드는 required ─────────────────────────────────────────────────────────────


def test_INV79_구조_결정_필드는_기본값이_없고_옵셔널은_문서화된_둘뿐이다():
    """INV-79 llm_schemas의 safe·intent·standalone에 기본값을 주지 않는다 — 기본값이 있으면 JSON 스키마 required에서
    빠져 모델이 필드를 생략해도 합법이 된다. 옵셔널은 모듈 docstring이 허용한 reason(RouteDecision)·variants뿐."""
    from pydantic import BaseModel

    from rag.turn import llm_schemas

    models = [
        v
        for v in vars(llm_schemas).values()
        if isinstance(v, type) and issubclass(v, BaseModel) and v.__module__ == llm_schemas.__name__
    ]
    assert len(models) >= 4, models
    decisive = [(m, f) for m in models for f in m.model_fields if f in ('safe', 'intent', 'standalone')]
    assert len(decisive) == 4, decisive  # RouteDecision.safe·intent, CondenseResult·CondenseMultiResult.standalone
    for m, f in decisive:
        assert m.model_fields[f].is_required(), f'{m.__name__}.{f}에 기본값이 있다'
        assert f in m.model_json_schema().get('required', []), f'{m.__name__}.{f}가 JSON 스키마 required에 없다'
    optional = {(m.__name__, f) for m in models for f, info in m.model_fields.items() if not info.is_required()}
    assert optional == {('RouteDecision', 'reason'), ('CondenseMultiResult', 'variants')}, optional


# ── INV-84 BM25 질의는 lex_clause로만 ────────────────────────────────────────────────────────────

FULLTEXT_KEYS = {'match', 'match_phrase', 'match_phrase_prefix', 'match_bool_prefix', 'multi_match'}
FULLTEXT_KEYS |= {'query_string', 'simple_query_string', 'combined_fields'}


def _fulltext_clauses(tree: ast.Module) -> list[tuple[str, int]]:
    """(감싼 함수 이름, 줄) — 키가 전문 검색 질의 종류인 dict 리터럴."""
    out = []
    for fn in [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))] or [tree]:
        for d in ast.walk(fn):
            if isinstance(d, ast.Dict) and any(
                isinstance(k, ast.Constant) and k.value in FULLTEXT_KEYS for k in d.keys
            ):
                out.append((getattr(fn, 'name', '<module>'), d.lineno))
    return sorted(set(out))


def test_INV84_rag의_전문_검색_절은_lex_clause_하나에서만_만든다():
    """INV-84 BM25 질의는 os_search.lex_clause()로만 만든다 — 기본 match는 2.18 hybrid를 500으로 죽인다.
    rag/ 어디에도 lex_clause 밖에서 match 계열 절을 dict로 짓는 곳이 없어야 한다."""
    found = [(f'{_rel(p)}::{fn}', line) for p in _py_files('rag') for fn, line in _fulltext_clauses(_tree(p))]
    owners = {k for k, _ in found}
    assert owners == {'rag/search/os_search.py::lex_clause'}, found


def test_INV84_탐지기는_함수_밖과_다른_종류의_전문_검색_절도_잡는다():
    src = "Q = {'multi_match': {}}\ndef f(q):\n    return {'bool': {'must': [{'match_phrase': {'text': q}}]}}"
    assert {fn for fn, _ in _fulltext_clauses(ast.parse(src))} == {'f'}
    assert _fulltext_clauses(ast.parse("Q = {'match': {'text': 'x'}}")) == [('<module>', 1)]


# ── INV-74 AGENTS.md의 셀 수 있는 숫자 ───────────────────────────────────────────────────────────

AGENTS = (ROOT / 'AGENTS.md').read_text(encoding='utf-8')


def _claim(pattern: str) -> tuple[str, ...]:
    m = re.search(pattern, AGENTS)
    assert m, f'AGENTS.md에서 {pattern!r} 문장을 못 찾았다 — 문장을 바꿨으면 이 테스트의 패턴도 같이 고쳐라'
    return m.groups()


def test_INV74_turn_패키지_모듈_수가_AGENTS와_같다():
    (claimed,) = _claim(r'턴 수명 모듈 (\d+)개')
    actual = [p for p in (ROOT / 'rag' / 'turn').glob('*.py') if p.name != '__init__.py']
    assert int(claimed) == len(actual), f'AGENTS.md는 {claimed}개, rag/turn/에는 {len(actual)}개'


def test_INV74_정본_골드_건수가_AGENTS와_같다():
    no_ev, trap = _claim(r'no_evidence (\d+) / trap (\d+)')
    rows = [
        json.loads(ln)
        for ln in (ROOT / 'eval' / 'gold_set_v2.jsonl').read_text(encoding='utf-8').splitlines()
        if ln.strip()
    ]
    counts = Counter(r['type'] for r in rows)
    assert (int(no_ev), int(trap)) == (counts['no_evidence'], counts['trap']), counts


def test_INV74_품질_상한_숫자가_설정과_같다():
    """AGENTS.md "코드 품질 기준" 2번의 상한 = pyproject.toml(ruff) + tests/test_code_quality.py(크기)."""
    from tests import test_code_quality as q

    claims = _claim(
        r'복잡도 (\d+), 분기 (\d+), 문장 (\d+), 인자 (\d+), 중첩 (\d+), 함수 코드줄 (\d+), 모듈 코드줄 (\d+)'
    )
    lint = tomllib.loads((ROOT / 'pyproject.toml').read_text(encoding='utf-8'))['tool']['ruff']['lint']
    pl = lint['pylint']
    actual = (
        lint['mccabe']['max-complexity'],
        pl['max-branches'],
        pl['max-statements'],
        pl['max-args'],
        pl['max-nested-blocks'],
        q.FUNC_MAX,
        q.MODULE_MAX,
    )
    assert tuple(map(int, claims)) == actual
