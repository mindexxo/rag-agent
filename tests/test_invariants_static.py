"""불변식 대장(INVARIANTS.md)의 빈 강제 칸을 채우는 정적·순수 검사(#315) — DB·OpenSearch 없이 돈다.

각 테스트는 INV 번호로 시작하고, 규칙의 정의점은 대장의 정의점 칸이 가리키는 파일 docstring이다.
여기서는 그 문장을 코드(AST·문자열 상수·가짜 클라이언트)와 대조만 한다 — 규칙을 바꿀 일이면 정의점을 고친다.

검사 대상을 파싱하지 못하거나 0개로 읽으면 조용히 통과하지 않게, 테스트마다 "최소 하나는 찾았다"는 가드를 둔다.
"""

import ast
import re
from pathlib import Path

import pytest

from rag.search import os_client as C

ROOT = Path(__file__).resolve().parent.parent
RAG = ROOT / 'rag'


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding='utf-8'), filename=str(path))


def _py_files(*targets: str) -> list[Path]:
    """대상 디렉터리(재귀)·파일 목록 → .py 경로. 디렉터리가 비면 가드가 잡게 그대로 돌려준다."""
    out: list[Path] = []
    for t in targets:
        p = ROOT / t
        out += sorted(p.rglob('*.py')) if p.is_dir() else [p]
    return out


def _rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def _toplevel_import_roots(tree: ast.Module) -> list[tuple[str, int]]:
    """함수 밖 import의 최상위 패키지 이름과 줄 — 모듈 수준 if/try·클래스 본문 안은 톱레벨로 센다
    (tests/test_architecture.py `_imports`와 같은 기준). 상대 import는 그 테스트가 따로 막으므로 여기선 건너뛴다."""
    found: list[tuple[str, int]] = []

    def visit(node: ast.AST) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                continue
            if isinstance(child, ast.Import):
                found.extend([(a.name.split('.')[0], child.lineno) for a in child.names])
            elif isinstance(child, ast.ImportFrom) and child.level == 0 and child.module:
                found.append((child.module.split('.')[0], child.lineno))
            visit(child)

    visit(tree)
    return found


def _docstring_nodes(tree: ast.Module) -> set[int]:
    """모듈·클래스·함수 docstring과 식 문장으로 놓인 문자열 상수의 id — 설명 문장은 '코드가 여는 경로'가 아니다."""
    return {
        id(n.value)
        for n in ast.walk(tree)
        if isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str)
    }


def _code_strings(tree: ast.Module) -> list[tuple[str, int]]:
    """docstring을 뺀 문자열 상수(f-string 조각 포함)와 줄."""
    skip = _docstring_nodes(tree)
    return [
        (n.value, n.lineno)
        for n in ast.walk(tree)
        if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in skip
    ]


# ── INV-30 ────────────────────────────────────────────────────────────────────────────────────────


def test_INV30_turn_state는_commit을_호출하지_않는다():
    """INV-30 turn_state 함수는 commit하지 않는다 — 호출부가 커밋 시점을 정한다(rag/turn/turn_state.py 모듈 docstring
    "세션 규약"). `x.commit(...)` 형태의 호출(Attribute attr=='commit')이 모듈 어디에도 없어야 한다."""
    tree = _tree(RAG / 'turn' / 'turn_state.py')
    funcs = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    assert funcs, 'turn_state.py에서 함수를 하나도 못 읽었다 — 검사가 빈 채로 통과하면 안 된다'
    commits = [
        n.lineno
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == 'commit'
    ]
    assert commits == [], f'rag/turn/turn_state.py가 commit을 호출한다(줄 {commits}) — 커밋은 호출자 소관이다'


# ── INV-73 ────────────────────────────────────────────────────────────────────────────────────────

HEAVY_THIRD_PARTY = {'opensearchpy', 'litellm'}


def test_INV73_opensearchpy와_litellm은_rag_어느_모듈도_톱레벨에서_import하지_않는다():
    """INV-73 opensearchpy·litellm은 rag/ 어느 모듈의 톱레벨에서도 import하지 않는다 — 함수 안 지연 import만 허용
    (AGENTS.md "순환 회피가 아닌 지연 import": 매핑 상수만 보는 테스트가 opensearchpy 없이 os_client를 import할 수
    있게, litellm은 118MB라 vLLM·TEI만 쓰는 프로세스가 비용을 안 내게). `rag.llm.litellm`(우리 모듈)은 최상위 이름이
    `rag`라 대상이 아니다."""
    files = _py_files('rag')
    assert len(files) > 20, f'rag/ 파일을 {len(files)}개만 읽었다'
    bad = [
        f'{_rel(p)}:{line} import {root}'
        for p in files
        for root, line in _toplevel_import_roots(_tree(p))
        if root in HEAVY_THIRD_PARTY
    ]
    assert bad == [], '톱레벨 import 금지(함수 안으로 내려라): ' + ', '.join(bad)


# ── INV-80 ────────────────────────────────────────────────────────────────────────────────────────

PROM_METRIC_TYPES = {'Counter', 'Histogram', 'Gauge', 'Summary', 'Info', 'Enum'}
# 라벨 이름을 '_'로 자른 토큰 중 하나라도 여기 있으면 고카디널리티로 본다. 근거: rag/metrics.py 모듈 docstring
# "라벨 규율" — tenant_id·conversation_id·filename 금지(카디널리티 폭발), 테넌트별 지표는 routers/stats.py 소관.
# 같은 성격(엔터티 id·사용자·자유 텍스트)인 message·document·faq·chunk·folder·user·query·session과 'id' 토큰 일반을 더했다.
# 허용 어휘(현재): route·reason·stage·ext·op·result — 전부 닫힌 작은 집합이다.
HIGH_CARDINALITY_TOKENS = {
    'tenant',
    'conversation',
    'message',
    'document',
    'doc',
    'faq',
    'chunk',
    'folder',
    'user',
    'session',
    'filename',
    'file',
    'path',
    'query',
    'id',
}


def _prometheus_names(tree: ast.Module) -> tuple[set[str], set[str]]:
    """(from prometheus_client import로 들어온 지표 클래스 이름, prometheus_client 모듈 별칭).
    collections.Counter 같은 동명이인을 지표로 오인하지 않으려고 import 출처로 가른다."""
    classes: set[str] = set()
    modules: set[str] = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom) and n.module == 'prometheus_client':
            classes |= {a.asname or a.name for a in n.names if a.name in PROM_METRIC_TYPES}
        elif isinstance(n, ast.Import):
            modules |= {a.asname or a.name for a in n.names if a.name == 'prometheus_client'}
    return classes, modules


def _is_metric_call(node: ast.AST, classes: set[str], modules: set[str]) -> bool:
    if not isinstance(node, ast.Call):
        return False
    f = node.func
    if isinstance(f, ast.Name):
        return f.id in classes
    return (
        isinstance(f, ast.Attribute)
        and f.attr in PROM_METRIC_TYPES
        and isinstance(f.value, ast.Name)
        and f.value.id in modules
    )


def _labelnames(call: ast.Call) -> list[str] | None:
    """labelnames 키워드 또는 세 번째 위치 인자. 없으면 [], 리터럴이 아니면 None(정적으로 검증 불가)."""
    node = next((k.value for k in call.keywords if k.arg == 'labelnames'), None)
    if node is None and len(call.args) >= 3:
        node = call.args[2]
    if node is None:
        return []
    try:
        value = ast.literal_eval(node)
    except ValueError:
        return None
    return [value] if isinstance(value, str) else list(value)


def _metric_definitions() -> list[tuple[str, list[str] | None]]:
    """rag/·routers/·main.py에서 prometheus_client 지표 정의를 전부 찾는다 — (위치, labelnames)."""
    found = []
    for p in _py_files('rag', 'routers', 'main.py'):
        tree = _tree(p)
        classes, modules = _prometheus_names(tree)
        if not (classes or modules):
            continue
        found += [
            (f'{_rel(p)}:{n.lineno}', _labelnames(n)) for n in ast.walk(tree) if _is_metric_call(n, classes, modules)
        ]
    return found


def test_INV80_Prometheus_라벨에_고카디널리티_값을_쓰지_않는다():
    """INV-80 Prometheus 라벨에 고카디널리티 값(tenant_id·conversation_id·filename 류)을 쓰지 않는다 — rag/metrics.py
    "라벨 규율". 지표 정의는 rag/metrics.py만이 아니라 rag/·routers/·main.py 전부에서 찾는다(다른 곳에 새로 생겨도 걸리게).
    labelnames가 리터럴이 아니면 정적으로 검증할 수 없으므로 그것도 실패로 본다."""
    defs = _metric_definitions()
    assert sum(loc.startswith('rag/metrics.py') for loc, _ in defs) >= 7, f'지표 정의를 덜 읽었다: {defs}'
    opaque = [loc for loc, labels in defs if labels is None]
    assert opaque == [], f'labelnames가 리터럴이 아니라 검증 불가: {opaque}'
    bad = [
        f'{loc} {label}'
        for loc, labels in defs
        for label in labels or []
        if set(label.lower().split('_')) & HIGH_CARDINALITY_TOKENS
    ]
    assert bad == [], '고카디널리티 라벨(카디널리티 폭발 — 테넌트별 값은 routers/stats.py 소관): ' + ', '.join(bad)


# ── INV-81 ────────────────────────────────────────────────────────────────────────────────────────


def _imports_schemas_query(tree: ast.Module) -> list[int]:
    lines = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Import) and any(
            a.name == 'schemas.query' or a.name.startswith('schemas.query.') for a in n.names
        ):
            lines.append(n.lineno)
        elif isinstance(n, ast.ImportFrom) and n.level == 0:
            if n.module == 'schemas.query' or (n.module == 'schemas' and any(a.name == 'query' for a in n.names)):
                lines.append(n.lineno)
    return lines


def test_INV81_죽은_레거시_schemas_query를_아무도_import하지_않는다():
    """INV-81 `schemas/query.py`의 죽은 레거시(QueryRequest·QueryResponse)를 쓰지 않는다 — AGENTS.md "이름이 비슷해서
    헷갈리는 것"(참조처 0, 실제로 쓰는 건 schemas/kms.py의 KmsQueryRequest). rag/·routers/·schemas/·eval/·main.py에서
    `schemas.query` import(함수 안 포함)가 0이어야 한다. schemas/query.py 자신은 제외."""
    files = [p for p in _py_files('rag', 'routers', 'schemas', 'eval', 'main.py') if p != ROOT / 'schemas' / 'query.py']
    assert len(files) > 50, f'대상 파일을 {len(files)}개만 읽었다'
    bad = [f'{_rel(p)}:{line}' for p in files for line in _imports_schemas_query(_tree(p))]
    assert bad == [], f'죽은 레거시 schemas.query를 import한다 — schemas/kms.py를 써라: {bad}'


# ── INV-82 ────────────────────────────────────────────────────────────────────────────────────────

GOLD_V2_SPLIT = re.compile(r'(?:^|/)gold_v2(?:/|$)')  # 테넌트별 분할본 디렉터리 — 'gold_set_v2'는 안 걸린다
GOLD_FILE = re.compile(r'gold[\w-]*\.jsonl')  # 골드 이름을 단 jsonl
SPLIT_READER = 'eval/validate_gold_v2.py'
# 정본 밖 골드를 읽는 것이 허용된 곳 — 사유와 함께. 여기 없는 모듈이 새 골드 파일을 열면 실패한다.
LEGACY_GOLD_READERS = {
    # v1 단일 테넌트(demo) 골드. run_all의 어느 축도 이 둘을 부르지 않는다(검색축은 eval/retrieval_v2.py) — eval/retrieval.py는
    # 지금 resolve_gold·score_one 정의점이라 남아 있고, main()만 v1 실행 경로다. eval/gate.py는 eval/refusal.py가 대체한 옛 축.
    'gold_set_v1.jsonl': {'eval/gate.py', 'eval/retrieval.py'},
}
CANONICAL = 'gold_set_v2.jsonl'


def _eval_strings() -> dict[str, list[tuple[str, int]]]:
    files = _py_files('eval')
    assert len(files) > 30, f'eval/ 파일을 {len(files)}개만 읽었다'
    return {_rel(p): _code_strings(_tree(p)) for p in files}


def test_INV82_분할본_gold_v2는_validate_gold_v2만_읽는다():
    """INV-82 측정 축은 정본 골드만 읽는다 — AGENTS.md "데이터·평가": 테넌트별 분할 eval/gold_v2/*.jsonl은 구축용 초안이며
    어느 측정 축도 읽지 않고, 읽는 것은 eval/validate_gold_v2.py 하나다. eval/ 아래 코드 문자열(docstring 제외)에서
    gold_v2 경로 조각을 쓰는 모듈이 정확히 그 하나여야 한다."""
    readers = {
        mod: lines
        for mod, strings in _eval_strings().items()
        if (lines := [line for s, line in strings if GOLD_V2_SPLIT.search(s)])
    }
    assert set(readers) == {SPLIT_READER}, f'분할본(eval/gold_v2/)을 읽는 모듈: {readers} — 측정 축은 정본만 읽는다'


def test_INV82_eval이_여는_골드_파일은_정본과_허용_목록뿐이다():
    """INV-82(보조) eval/ 코드가 이름으로 여는 골드 jsonl은 정본(eval/gold_set_v2.jsonl)과 LEGACY_GOLD_READERS 뿐이다.
    gold_private(사내 실문서 골드)는 디렉터리라 이 패턴에 안 걸리고 AGENTS.md가 정본 병합 예외로 둔 것이다.
    축별 전용 셋(intent_set_v1·cache_set_v1 등)은 골드가 아닌 별도 질문셋이라 대상이 아니다."""
    allowed = {CANONICAL: None, **LEGACY_GOLD_READERS}
    seen: dict[str, set[str]] = {}
    for mod, strings in _eval_strings().items():
        for s, line in strings:
            for name in GOLD_FILE.findall(s):
                seen.setdefault(name, set()).add(mod)
                assert name in allowed, f'{mod}:{line} 허용 목록 밖의 골드 파일 {name}'
                readers = allowed[name]
                assert readers is None or mod in readers, f'{mod}:{line} {name}은 {sorted(readers)}만 읽는다'
    assert CANONICAL in seen, '정본 골드를 여는 모듈을 하나도 못 찾았다 — 패턴이 틀렸다'


# ── INV-28 ────────────────────────────────────────────────────────────────────────────────────────

OUTBOX_MODEL = 'SearchIndexOutbox'
# 스키마 접두(public.search_index_outbox)·따옴표 식별자("public"."search_index_outbox")도 같은 테이블이다.
OUTBOX_TABLE_SQL = re.compile(
    r'\b(delete\s+from|truncate(\s+table)?)\s+(only\s+)?("?\w+"?\s*\.\s*)?"?search_index_outbox\b', re.I
)


def _delete_calls(tree: ast.Module) -> list[ast.Call]:
    """sqlalchemy `delete(...)`와 `x.delete(...)`(session.delete 등) 호출 전부."""
    return [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and (
            (isinstance(n.func, ast.Name) and n.func.id == 'delete')
            or (isinstance(n.func, ast.Attribute) and n.func.attr == 'delete')
        )
    ]


def _mentions_outbox(tree: ast.Module) -> bool:
    """SearchIndexOutbox를 이름(`SearchIndexOutbox`)으로든 속성(`models.SearchIndexOutbox`)으로든 쓰는가."""
    return any(
        (isinstance(n, ast.Name) and n.id == OUTBOX_MODEL) or (isinstance(n, ast.Attribute) and n.attr == OUTBOX_MODEL)
        for n in ast.walk(tree)
    )


def _outbox_deletes(tree: ast.Module, label: str) -> list[str]:
    hits = [f'{label}:{line} SQL' for s, line in _code_strings(tree) if OUTBOX_TABLE_SQL.search(s)]
    if _mentions_outbox(tree):
        # outbox 모델을 다루는 모듈에서는 delete 호출 자체를 금지한다 — session.delete(row)는 정적으로 대상 형식을
        # 알 수 없으므로 보수적으로 본다(지금 이런 모듈은 rag/outbox.py 하나이고 delete 호출이 0이다).
        hits += [f'{label}:{c.lineno} {ast.unparse(c.func)}(...)' for c in _delete_calls(tree)]
    return hits


def test_INV28_outbox_행을_지우는_코드가_없다():
    """INV-28 outbox 행은 지우지 않는다 — done·failed 행이 이력이자 관측 지점이다(rag/outbox.py·rag/models.py
    SearchIndexOutbox docstring, 보존 정리는 나중에 스케줄러로). rag/·routers/·main.py에서
        ① SearchIndexOutbox를 쓰는 모듈의 delete 호출(sqlalchemy delete(SearchIndexOutbox)·session.delete(row))
        ② 'DELETE FROM search_index_outbox'·TRUNCATE 원시 SQL 문자열
    이 없어야 한다. SearchIndexOutbox를 쓰는 모듈이 하나도 안 잡히면 검사가 빈 것이므로 실패."""
    files = _py_files('rag', 'routers', 'main.py')
    users = [_rel(p) for p in files if _mentions_outbox(_tree(p))]
    assert 'rag/outbox.py' in users, f'SearchIndexOutbox 사용처를 못 찾았다: {users}'
    bad = [hit for p in files for hit in _outbox_deletes(_tree(p), _rel(p))]
    assert bad == [], f'outbox 행 삭제(이력 소실): {bad}'


@pytest.mark.parametrize(
    'src',
    [
        'from rag.models import SearchIndexOutbox\nstmt = delete(SearchIndexOutbox)',
        'from rag import models\nstmt = delete(models.SearchIndexOutbox)',
        'from rag import models\nasync def f(s, row: models.SearchIndexOutbox):\n    await s.delete(row)',
        'q = "DELETE FROM search_index_outbox WHERE id = 1"',
        'q = "delete from public.search_index_outbox"',
        'q = \'DELETE FROM "public"."search_index_outbox"\'',
        'q = "TRUNCATE TABLE public.search_index_outbox"',
    ],
)
def test_INV28_탐지기는_속성_참조와_스키마_접두_삭제도_잡는다(src):
    """INV-28(검사기 자체) 위 검사가 공허하지 않다 — 모델 속성 참조(`models.SearchIndexOutbox`)·스키마 접두 SQL도 잡는다."""
    assert _outbox_deletes(ast.parse(src), '<합성>'), src


# ── INV-83 ────────────────────────────────────────────────────────────────────────────────────────


class _FakeIndices:
    """exists=True인 기존 인덱스 — get_mapping은 주어진 properties를 돌려주고, 쓰기 호출은 전부 기록한다."""

    def __init__(self, have: dict):
        self.have = have
        self.calls: list[tuple[str, dict | None]] = []

    async def exists(self, *a, **k):
        return True

    async def get_mapping(self, *a, **k):
        return {'kms': {'mappings': {'properties': self.have}}}

    async def put_mapping(self, *a, body=None, **k):
        self.calls.append(('put_mapping', body))

    async def create(self, *a, body=None, **k):
        self.calls.append(('create', body))

    async def put_settings(self, *a, body=None, **k):
        self.calls.append(('put_settings', body))

    async def delete(self, *a, **k):
        self.calls.append(('delete', None))


@pytest.mark.asyncio
async def test_INV83_기존_인덱스에는_빠진_추가_필드만_정의_그대로_덧붙인다(monkeypatch):
    """INV-83 기존 인덱스에는 덧붙이는 필드만 추가하고 매핑 변경은 건드리지 않는다 — rag/search/os_client.py ensure_index·
    ADDITIVE_FIELDS(매핑 변경은 새 인덱스+전량 재색인 절차). 기존 매핑이 정의와 **다른 타입**이어도(검색 필드 text를
    keyword로, 추가 필드 하나는 이미 다른 모양으로) ensure_index가 보내는 것은 put_mapping 한 번이고, 그 properties는
    ① ADDITIVE_FIELDS 중 기존에 없는 것만 ② MAPPING 정의와 같은 값 ③ 기존 필드는 하나도 없음 — create·설정 변경·삭제는 없다."""
    props = C.MAPPING['mappings']['properties']
    already = C.ADDITIVE_FIELDS[0]
    have = {'text': {'type': 'keyword'}, C.NORI_FIELD: {'type': 'keyword'}, already: {'type': 'keyword'}}
    fake = _FakeIndices(have)
    monkeypatch.setattr(C, 'client', lambda: type('_Client', (), {'indices': fake})())
    await C.ensure_index()
    assert [name for name, _ in fake.calls] == ['put_mapping'], fake.calls
    sent = fake.calls[0][1]['properties']
    assert set(sent) == set(C.ADDITIVE_FIELDS) - set(have)
    assert not set(sent) & set(have), '이미 있는 필드를 다시 보낸다 — 기존 필드 매핑 변경 시도'
    assert all(sent[f] == props[f] for f in sent), '추가 필드가 MAPPING 정의와 다른 모양으로 나간다'


def test_INV83_추가_필드는_재색인_없이_붙여도_되는_모양뿐이다():
    """INV-83(보조) ADDITIVE_FIELDS는 재색인 없이 붙여도 안전한 것만 — rag/search/os_client.py 주석: 검색하지 않는 보관 필드
    (index: False)이거나 새 keyword 정확 일치 필드다. 분석되는 검색 필드(analyzer를 단 text·knn_vector)가 여기 들어오면
    옛 청크는 그 필드가 빈 채로 검색돼 조용히 리콜이 깎인다 — 그건 재색인 절차로 가야 한다."""
    props = C.MAPPING['mappings']['properties']
    assert C.ADDITIVE_FIELDS, 'ADDITIVE_FIELDS가 비었다'
    unsafe = [f for f in C.ADDITIVE_FIELDS if not (props[f].get('index') is False or props[f].get('type') == 'keyword')]
    assert unsafe == [], f'재색인 없이 붙이면 안 되는 필드가 ADDITIVE_FIELDS에 있다: {unsafe}'
