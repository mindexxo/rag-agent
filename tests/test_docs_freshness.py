"""문서·주석이 코드와 어긋나는 것을 기계가 잡는다 (#88 하네스 1단계).

이 저장소는 규약을 코드 옆(docstring·주석)에 두고 grep으로 찾게 설계했다(AGENTS.md 참조).
그 방식의 유일한 실패 모드는 **코드가 바뀔 때 옆의 문장이 따라 안 바뀌는 것**이다. 그러면 문서가
조용히 거짓말을 하고, 사람과 에이전트 모두 그 거짓말을 근거로 판단한다. 이 파일이 그걸 막는다.

부채 청소용이 아니라 래칫이다 — #88 조사 시점에 살아있는 위반은 1건(eval/refusal.py의 gold
카운트 60/48, 실제 58/50)뿐이었고 그건 같은 PR에서 고쳤다. 여기 있는 4종은 "이미 한 번
어긋난 적이 있는 종류"만 검사한다. 새 검사를 추가할 때도 같은 기준을 쓴다 — 가정이 아니라
실제로 어긋난 이력이 있을 때.

에러 메시지에는 반드시 **무엇을 어떻게 고쳐야 하는지**를 적는다. 이 테스트의 독자는 대개
에이전트이고, 메시지가 그대로 수정 지침이 되어야 사람 손을 덜 탄다.
"""
from __future__ import annotations

import ast
import json
import re
from collections import Counter
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent


def _scan_files() -> list[Path]:
    """문서 문장이 들어있는 파일 — 코드의 docstring·주석과 루트 마크다운."""
    files: list[Path] = []
    for pkg in ('rag', 'routers', 'schemas', 'eval', 'tests'):
        files += sorted((_ROOT / pkg).glob('*.py'))
    files += [_ROOT / n for n in ('config.py', 'main.py', 'database.py', 'schema.sql')]
    files += [_ROOT / n for n in ('AGENTS.md', 'CLAUDE.md', 'README.md')]
    # 이 파일 자신은 제외한다 — 없어진 경로를 '검사 대상 데이터'로 적어두므로(화이트리스트·에러
    # 메시지) 스스로를 검사하면 반드시 실패한다.
    here = Path(__file__).resolve()
    return [f for f in files if f.exists() and f.resolve() != here]


_SCAN = _scan_files()


def _rel(p: Path) -> str:
    return str(p.relative_to(_ROOT))


# ---------------------------------------------------------------- ① 경로 참조

# 저장소 안을 가리키는 경로 토큰만 잡는다. 확장자를 요구해 산문과 구분한다.
_PATH_RE = re.compile(
    r'\b(?:rag|routers|schemas|eval|tests|sample_docs|docs_internal)/[\w/\-.]*'
    r'\.(?:py|sql|md|jsonl|json|yml|yaml|txt)\b'
)

# gitignore된 디렉터리 — 저장소만으로는 실존을 검증할 수 없다(생성물·내부 기획문서).
_UNVERIFIABLE_PREFIXES = (
    'docs_internal/',    # .gitignore:36 — 내부 기획문서
    'eval/results/',     # .gitignore:26 — 측정 산출물
)

# 이 저장소는 폐기된 것을 '구 X'·'옛 X'로 표기한다(실측 관용구, 여러 모듈에서 사용).
# 그렇게 표시된 참조는 없어진 게 정상이므로 위반이 아니다.
#
# 마커가 참조 **바로 앞**에 붙은 경우만 인정한다("구 rag/foo.py"). 사이에 다른 단어가 끼면
# ("옛 판정은 rag/foo.py") 통과하지 못하고 _PATH_WHITELIST 등록이 필요하다 — 의도한 것이다.
# 문장 전체에서 '구'·'옛'을 찾으면 무관한 서술이 진짜 위반을 덮어버린다.
_PAST_TENSE_RE = re.compile(r'(?:구|옛)\s*$')


def _is_past_tense(text: str, start: int) -> bool:
    """참조 직전에 폐기 마커가 붙어 있는가."""
    return bool(_PAST_TENSE_RE.search(text[max(0, start - 4):start]))


# 위 두 규칙으로도 걸러지지 않는 과거형 서술만 등록한다. 등록 시 사유를 반드시 적는다.
# 이 딕셔너리는 늘어나는 방향이 아니라 줄어드는 방향으로 관리한다.
_PATH_WHITELIST: dict[tuple[str, str], str] = {
    ('rag/citation_tail.py', 'rag/answer_check.py'):
        '#61에서 삭제된 파일. citation_tail.py가 과거형으로 폐기 사유를 설명하는 문장이라 오도가 아니다.',
}


class TestPathReferences:
    """문장 안의 파일 경로가 실제로 존재해야 한다."""

    def test_문장_속_경로는_실존해야_한다(self):
        broken = []
        for f in _SCAN:
            text = f.read_text(encoding='utf-8')
            for m in _PATH_RE.finditer(text):
                ref = m.group()
                if ref.startswith(_UNVERIFIABLE_PREFIXES):
                    continue
                if (_ROOT / ref).exists():
                    continue
                if _is_past_tense(text, m.start()):
                    continue
                if (_rel(f), ref) in _PATH_WHITELIST:
                    continue
                broken.append((_rel(f), ref))
        broken = sorted(set(broken))

        assert not broken, '\n'.join(
            f"{src}: 경로 '{ref}'가 저장소에 없다. 파일이 이동·삭제됐으면 이 문장을 새 경로로 "
            f"고쳐라. 이미 없어진 파일을 과거형으로 설명하는 문장이라면 "
            f"tests/test_docs_freshness.py의 _PATH_WHITELIST에 ('{src}', '{ref}') 키로 사유를 "
            f"적어 등록해라 — 사유 없는 등록은 리뷰에서 반려한다."
            for src, ref in broken
        )


# ---------------------------------------------------------------- ② 심볼 참조

# 이 저장소 관용구: "config.py의 condense_history_budget_tokens" 처럼 '<파일>의 <심볼>'로 쓴다.
# 심볼 부분은 ASCII만 받는다 — \w는 한글도 매칭해서 조사가 이름에 붙어버린다("REQUIRED_KEYS를").
# 끝의 `*`는 계열 표기다("TAIL_EXAMPLE_*") — 접두어가 일치하는 정의가 하나라도 있으면 통과.
_SYMBOL_RE = re.compile(r'([\w/]+\.py)의\s+([A-Za-z_][A-Za-z0-9_.]*)(\*?)')


def _defined_names(path: Path) -> set[str]:
    """top-level 및 클래스 본문에 정의된 이름 전체."""
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, ast.Assign):
            names |= {t.id for t in node.targets if isinstance(t, ast.Name)}
    return names


class TestSymbolReferences:
    """'<파일>.py의 <심볼>' 참조가 그 파일에 실제로 있어야 한다.

    '<파일>의 <말>' 관용구는 산문에도 흔히 쓰여서("worker.py의 arq cron", "streaming.py의 FE
    대응") 정의 유무만으로 판정하면 오탐이 쏟아진다. 오탐이 쌓이면 린터는 무시당하고, 무시당하는
    린터는 없는 것보다 나쁘다 — 그래서 **정밀도를 택했다**: AST 정의 목록에도 없고 파일 본문에
    단어로도 등장하지 않는 이름만 위반으로 본다(= 그 파일에 그런 것이 존재하지 않는다는 뜻).

    정의 목록을 AST로 뽑는 이유는 따로 있다 — 문자열 검색만 쓰면 정의를 지워도 import 줄이나
    호출부에 이름이 남아 통과한다(tests/test_eval_condense_trim.py에서 실제로 겪은 함정).
    """

    def test_문장_속_심볼은_그_파일에_있어야_한다(self):
        broken = []
        for f in _SCAN:
            # 백틱은 마크다운 장식이므로 제거하고 문장만 본다.
            text = f.read_text(encoding='utf-8').replace('`', '')
            for m in _SYMBOL_RE.finditer(text):
                ref_path, symbol, wildcard = m.group(1), m.group(2), m.group(3)
                target = _ROOT / ref_path
                if not target.exists():
                    continue          # 경로 자체 문제는 ①이 잡는다
                if _is_past_tense(text, m.start()):
                    continue          # '구 X의 Y' — 폐기된 구조를 설명하는 문장
                # dotted 참조·문장 끝 마침표는 마지막 조각만 대조한다.
                leaf = symbol.rstrip('.').split('.')[-1]
                if not leaf:
                    continue
                defined = _defined_names(target)
                if wildcard:          # 'TAIL_EXAMPLE_*' — 계열이 하나라도 있으면 된다
                    if not any(n.startswith(leaf) for n in defined):
                        broken.append((_rel(f), ref_path, symbol + '*'))
                    continue
                if leaf in defined:
                    continue
                # 본문 등장은 단어 경계로 확인한다 — 부분 문자열로 보면 'HISTORY'가
                # 'HISTORY_ISOLATED'에 걸려 통과해버린다(짧은 이름·접두어가 전부 구멍).
                if re.search(rf'\b{re.escape(leaf)}\b', target.read_text(encoding='utf-8')):
                    continue
                broken.append((_rel(f), ref_path, symbol))

        assert not broken, '\n'.join(
            f"{src}: '{ref_path}의 {symbol}' 참조가 어긋난다 — {ref_path}에 '{symbol}' 정의가 "
            f"없다. 심볼 이름이 바뀌었으면 이 문장을 새 이름으로 고치고, 코드가 다른 파일로 "
            f"옮겨졌으면 경로를 고쳐라."
            for src, ref_path, symbol in broken
        )


# ------------------------------------------------- ③ eval docstring ↔ 데이터

# eval 축 문서가 gold 건수를 문장으로 주장한다 — 데이터가 바뀌면 조용히 거짓이 된다.
# (실제 사고: #88 조사에서 refusal.py가 60/48이라 주장, 실제 58/50)
_DOCSTRING_COUNT_CLAIMS = (
    ('eval/refusal.py', 'eval/gold_set_v2.jsonl', 'type', ('no_evidence', 'trap')),
    # 캐시 셋 40쌍 확장(#113) — kind별 건수를 docstring이 주장하므로 기계 검증
    ('eval/cache_eval.py', 'eval/cache_set_v1.jsonl', 'kind',
     ('negation', 'numeric', 'condition', 'temporal', 'exception',
      'para_surface', 'para_deep')),
)


class TestEvalDocstringCounts:
    """eval 모듈 docstring이 주장하는 건수 = gold 데이터의 실제 건수."""

    @pytest.mark.parametrize('module,data,field,labels', _DOCSTRING_COUNT_CLAIMS)
    def test_docstring_건수_주장이_데이터와_일치한다(self, module, data, field, labels):
        doc = ast.get_docstring(ast.parse((_ROOT / module).read_text(encoding='utf-8')))
        assert doc, f'{module}에 모듈 docstring이 없다 — 축의 판정 규약은 docstring이 정의점이다.'

        rows = [json.loads(l) for l in (_ROOT / data).read_text(encoding='utf-8').splitlines() if l.strip()]
        actual = Counter(r[field] for r in rows)

        for label in labels:
            m = re.search(rf'{re.escape(label)}\((\d+)\)', doc)
            assert m, (
                f"{module} docstring에 '{label}(건수)' 형태의 주장이 없다. 이 검사는 그 형식을 "
                f"전제한다 — 문장을 지웠다면 tests/test_docs_freshness.py의 "
                f"_DOCSTRING_COUNT_CLAIMS에서 이 항목도 함께 빼라."
            )
            claimed = int(m.group(1))
            assert claimed == actual[label], (
                f"{module} docstring은 {label}({claimed})이라 적었지만 {data}의 실제 "
                f"{field}={label} 건수는 {actual[label]}건이다. gold를 고친 쪽이 맞다면 "
                f"{module}의 숫자를 {actual[label]}로 갱신해라. docstring이 맞다면 gold가 잘못 "
                f"수정된 것이니 변경 이력을 확인하고, eval/gold_v2/*.jsonl 분할본도 같이 "
                f"어긋났는지 `python -m eval.validate_gold_v2`로 점검해라."
            )


# ----------------------------------------------- ④ 저장소 내부 숫자 불변식

# "두 파일이 같은 값이어야 한다"고 양쪽에 주석으로 적어둔 쌍 — 강제하는 코드가 없었다.
_NUMBER_INVARIANTS = (
    ('concurrency_limit_default', 'concurrency_limit'),
    ('user_concurrency_default', 'user_concurrency'),
)


def _config_default(field: str) -> int:
    tree = ast.parse((_ROOT / 'config.py').read_text(encoding='utf-8'))
    cls = next((n for n in ast.walk(tree)
                if isinstance(n, ast.ClassDef) and n.name == 'Settings'), None)
    assert cls is not None, (
        'config.py에서 Settings 클래스를 찾지 못했다. 클래스명이 바뀌었으면 '
        'tests/test_docs_freshness.py의 _config_default도 같이 고쳐라.'
    )
    node = next((n for n in cls.body
                 if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)
                 and n.target.id == field), None)
    assert node is not None, (
        f"config.py의 Settings에 '{field}' 필드가 없다. 필드명이 바뀌었으면 "
        f'tests/test_docs_freshness.py의 _NUMBER_INVARIANTS에서 이 항목의 이름을 갱신하고, '
        f'설정이 삭제됐다면 대응하는 schema.sql 컬럼도 함께 정리했는지 확인해라.'
    )
    return node.value.value


def _schema_default(column: str) -> int:
    text = (_ROOT / 'schema.sql').read_text(encoding='utf-8')
    # 주석 처리된 마이그레이션 예시(-- ALTER TABLE ...)에 걸리지 않도록 CREATE 본문 형태만 본다.
    m = re.search(rf'^\s*{column}\s+INTEGER\s+NOT NULL\s+DEFAULT\s+(\d+)', text, re.MULTILINE)
    assert m, f'schema.sql에서 {column}의 DEFAULT를 찾지 못했다 — 컬럼 정의 형식이 바뀌었는지 확인해라.'
    return int(m.group(1))


class TestInternalNumberInvariants:
    """config 기본값과 schema.sql DEFAULT가 같아야 한다 (양쪽 주석이 그렇게 약속했다)."""

    @pytest.mark.parametrize('field,column', _NUMBER_INVARIANTS)
    def test_동시성_기본값이_config와_schema에서_일치한다(self, field, column):
        cfg, sql = _config_default(field), _schema_default(column)
        assert cfg == sql, (
            f"config.py의 {field}={cfg}인데 schema.sql의 tenant_quotas.{column} DEFAULT는 "
            f"{sql}이다. 두 값은 '같은 값으로 유지할 것'이라고 양쪽 주석에 명시돼 있다 — "
            f"한쪽을 바꿨으면 다른 쪽도 같이 고쳐라. 이미 DB에 반영된 환경이 있으면 "
            f"schema.sql 하단의 주석 처리된 ALTER TABLE 패턴을 따라 마이그레이션 문장도 남겨라."
        )
