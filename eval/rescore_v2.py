"""저장된 측정 결과의 citation_accuracy 재채점 — gold 보정(대체 출처 그룹) 반영.

답변 재생성 없이 결과 jsonl의 answer + 최신 gold로 인용 점수만 다시 계산한다.
사용: python3 -m eval.rescore_v2
"""
import json
from pathlib import Path

from eval.generation import citation_accuracy, expected_points_coverage, expected_points_hits

ROOT = Path(__file__).resolve().parent
GOLD = {json.loads(l)['id']: json.loads(l)
        for l in (ROOT / 'gold_set_v2.jsonl').read_text().splitlines() if l.strip()}


def main() -> None:
    for mode in ('oracle', 'retrieved'):
        path = ROOT / 'results' / f'generation_{mode}.jsonl'
        rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        changed = 0
        for r in rows:
            g = GOLD[r['id']]
            new_cite = citation_accuracy(r['answer'], g.get('expected_docs', []))
            # #221 이후 행의 expected_points_coverage는 규칙 + 2단계 심판 값이다 — 규칙만으로 다시 계산해 덮으면
            # 심판 반영분이 조용히 지워진다. 그런 행은 규칙을 다시 계산하고, 최종은 "새 규칙 적중 ∪ 감사의 covered"로
            # 다시 짓는다(심판은 다시 부르지 않는다 — 새로 미적중이 된 포인트는 판정 없이 미적중으로 남는다).
            points = g.get('expected_points', [])
            judged = r.get('epcov_judge') not in (None, 'rule-only')
            before = (r['scores']['citation_accuracy'], r['scores'].get('expected_points_coverage'))
            if judged and points:
                rule = expected_points_hits(r['answer'], points)
                covered = {a['k'] for a in r['scores'].get('expected_points_audit') or [] if a.get('label') == 'covered'}
                r['scores']['expected_points_coverage_rule'] = sum(rule) / len(points)
                r['scores']['expected_points_coverage'] = sum(h or k in covered for k, h in enumerate(rule)) / len(points)
            else:
                r['scores']['expected_points_coverage'] = expected_points_coverage(r['answer'], points)
            r['scores']['citation_accuracy'] = new_cite
            if before != (new_cite, r['scores']['expected_points_coverage']):
                changed += 1
        path.write_text('\n'.join(json.dumps(r, ensure_ascii=False) for r in rows) + '\n')
        print(f'{mode}: {len(rows)}행 재채점, 점수 변경 {changed}건')


if __name__ == '__main__':
    main()
