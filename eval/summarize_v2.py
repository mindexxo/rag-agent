"""기준선 v2 결과 집계 — 모드 × 테넌트 × 타입.

사용: python3 -m eval.summarize_v2
입력: eval/results/generation_oracle.jsonl, generation_retrieved.jsonl
"""
import json
from collections import defaultdict
from pathlib import Path

RESULT_DIR = Path(__file__).resolve().parent / 'results'
TENANTS = ['summers', 'homeplus', 'adererror', 'aromanica', 'goodpeople', 'harim']


def tenant_of(row_id: str) -> str:
    p = row_id.split('_')[0]
    return p if p in TENANTS else 'demo'


def _avg(rows, key):
    vals = [r['scores'][key] for r in rows if r['scores'][key] is not None]
    return sum(vals) / len(vals) if vals else 0.0


def table(rows, group_fn, label):
    groups = defaultdict(list)
    for r in rows:
        groups[group_fn(r)].append(r)
    print(f'{label:<16}{"n":>5}{"EPCov":>8}{"Cite":>7}')
    for g in sorted(groups):
        rs = groups[g]
        print(f'{g:<16}{len(rs):>5}{_avg(rs, "expected_points_coverage"):>8.3f}'
              f'{_avg(rs, "citation_accuracy"):>7.3f}')
    print(f'{"(전체)":<16}{len(rows):>5}{_avg(rows, "expected_points_coverage"):>8.3f}'
          f'{_avg(rows, "citation_accuracy"):>7.3f}')


def main() -> None:
    for mode in ('oracle', 'retrieved'):
        path = RESULT_DIR / f'generation_{mode}.jsonl'
        rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        print(f'\n{"=" * 44}\nmode: {mode}  ({len(rows)}문항)\n{"=" * 44}')
        print('\n[테넌트별]')
        table(rows, lambda r: tenant_of(r['id']), 'tenant')
        print('\n[타입별]')
        table(rows, lambda r: r['type'], 'type')
        # 테넌트 × 취약 타입 교차 (paraphrase·multi_turn만 — 관심 축)
        print('\n[테넌트 × paraphrase/multi_turn EPCov]')
        cross = defaultdict(list)
        for r in rows:
            if r['type'] in ('paraphrase', 'multi_turn'):
                cross[(tenant_of(r['id']), r['type'])].append(r)
        for (t, ty) in sorted(cross):
            rs = cross[(t, ty)]
            print(f'  {t:<12}{ty:<12}{_avg(rs, "expected_points_coverage"):>7.3f} (n={len(rs)})')


if __name__ == '__main__':
    main()
