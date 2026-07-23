"""gold v2 검증기 — 스키마·쿼터·파일명 정합 + gold 오염 검사.

오염 검사: has_evidence 문항의 expected_points·snippet이 해당 테넌트의 실제
소스 텍스트에 존재하는지 전수 대조 (생성 에이전트가 수치를 지어냈으면 여기서 걸린다).

사용: python3 -m eval.validate_gold_v2 [tenant ...]
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / 'sample_docs' / 'corpus_v2'
GOLD = ROOT / 'eval' / 'gold_v2'

TENANTS = ['summers', 'homeplus', 'adererror', 'aromanica', 'goodpeople', 'harim']
QUOTA = {'single_fact': 20, 'paraphrase': 20, 'rare_lexical': 10, 'multi_doc': 10,
         'multi_turn': 15, 'no_evidence': 10, 'trap': 8, 'smalltalk': 4}
SAFETY_TYPES = {'safety', 'prompt_injection', 'pii'}   # 합계 3 (type='safety' + safety_tags 방식 수용)
REQUIRED_KEYS = {'id', 'query', 'conversation', 'expected_docs', 'expected_chunks',
                 'expected_points', 'has_evidence', 'type', 'safety_tags'}


def _norm(s: str) -> str:
    # '|' 무시: xlsx 청크는 마크다운 표(| 구분)로 렌더되므로 매칭에서 제외
    return re.sub(r'[\s|]+', '', s)


def _numeric_tokens(s: str) -> list[str]:
    """포인트 속 수치 토큰만 추출 — 서술형 표현은 채점기의 임베딩 폴백 영역이라
    리터럴 검증하지 않고, 지어낸 숫자(진짜 오염 위험)만 잡는다."""
    return re.findall(r'\d[\d,.:~]*', s)


def tenant_source_text(tenant: str) -> str:
    """테넌트의 전체 소스 텍스트 (md 원본 + txt + xlsx 렌더) — 오염 대조용."""
    parts = []
    src = CORPUS / '_src' / tenant
    for f in sorted(src.glob('*.md')):
        if '.rev2' not in f.name:
            parts.append(f.read_text())
    for f in sorted((CORPUS / tenant).iterdir()):
        if '.rev2' in f.name:
            continue
        if f.suffix in ('.md', '.txt'):
            parts.append(f.read_text())
        elif f.suffix == '.xlsx':
            from rag.xlsx_chunking import chunk_xlsx
            parts.extend(c.text for c in chunk_xlsx(f))
    return _norm('\n'.join(parts))


def uploaded_filenames(tenant: str) -> set[str]:
    return {f.name for f in (CORPUS / tenant).iterdir() if '.rev2' not in f.name}


def validate(tenant: str) -> list[str]:
    errors = []
    path = GOLD / f'{tenant}.jsonl'
    if not path.exists():
        return [f'{tenant}: 파일 없음']
    rows = []
    for i, line in enumerate(path.read_text().splitlines(), 1):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as e:
            errors.append(f'{tenant}:{i} JSON 파싱 실패: {e}')
    if len(rows) != 100:
        errors.append(f'{tenant}: {len(rows)}줄 (기대 100)')

    valid_files = uploaded_filenames(tenant)
    source = tenant_source_text(tenant)
    ids = Counter(r.get('id') for r in rows)
    types = Counter(r.get('type') for r in rows)

    for dup, n in ids.items():
        if n > 1:
            errors.append(f'{tenant}: id 중복 {dup} ×{n}')

    for r in rows:
        rid = r.get('id', '?')
        missing = REQUIRED_KEYS - set(r)
        if missing:
            errors.append(f'{rid}: 키 누락 {missing}')
            continue
        for d in r['expected_docs']:
            for fn in ([d] if isinstance(d, str) else d):   # str 또는 대체 그룹(list)
                if fn not in valid_files:
                    errors.append(f'{rid}: 업로드에 없는 파일명 {fn}')
        if r['type'] == 'multi_turn' and not r['conversation']:
            errors.append(f'{rid}: multi_turn인데 conversation 없음')
        if r['has_evidence']:
            if not r['expected_docs']:
                errors.append(f'{rid}: has_evidence인데 expected_docs 없음')
            source_nc = source.replace(',', '')          # xlsx 숫자셀은 콤마 없이 렌더됨
            for p in r['expected_points']:
                for tok in _numeric_tokens(_norm(p)):
                    if tok not in source and tok.replace(',', '') not in source_nc:
                        errors.append(f'{rid}: point의 수치가 소스에 없음(오염 의심): {p!r} ({tok})')
            for c in r['expected_chunks'] or []:
                if _norm(c.get('snippet', '')) not in source:
                    errors.append(f'{rid}: snippet이 소스에 없음: {c.get("snippet")!r}')
        else:
            if r['expected_docs'] or r['expected_points']:
                errors.append(f'{rid}: has_evidence=false인데 기대값 존재')

    for t, want in QUOTA.items():
        if types.get(t, 0) != want:
            errors.append(f'{tenant}: {t} {types.get(t, 0)}개 (기대 {want})')
    n_safety = sum(types.get(t, 0) for t in SAFETY_TYPES)
    if n_safety != 3:
        errors.append(f'{tenant}: safety 합계 {n_safety} (기대 3)')
    return errors


def main() -> None:
    targets = sys.argv[1:] or TENANTS
    total_err = 0
    for tenant in targets:
        errs = validate(tenant)
        print(f'{tenant}: {"OK" if not errs else f"{len(errs)}건 문제"}')
        for e in errs[:20]:
            print(f'  - {e}')
        if len(errs) > 20:
            print(f'  ... 외 {len(errs) - 20}건')
        total_err += len(errs)
    sys.exit(1 if total_err else 0)


if __name__ == '__main__':
    main()
