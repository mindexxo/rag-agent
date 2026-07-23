"""저장된 측정 결과의 citation_accuracy 재채점 — gold 보정(대체 출처 그룹) 반영.

답변 재생성 없이 결과 jsonl의 answer + 최신 gold로 인용 점수만 다시 계산한다.
사용: python3 -m eval.rescore_v2
"""
import json
from pathlib import Path

from eval.generation import citation_accuracy, expected_points_coverage

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
            new_ep = expected_points_coverage(r['answer'], g.get('expected_points', []))
            if (new_cite != r['scores']['citation_accuracy']
                    or new_ep != r['scores']['expected_points_coverage']):
                changed += 1
            r['scores']['citation_accuracy'] = new_cite
            r['scores']['expected_points_coverage'] = new_ep
        path.write_text('\n'.join(json.dumps(r, ensure_ascii=False) for r in rows) + '\n')
        print(f'{mode}: {len(rows)}행 재채점, 점수 변경 {changed}건')


if __name__ == '__main__':
    main()
