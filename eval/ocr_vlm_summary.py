"""#296 요약 — 표 이미지 하네스(#246) 실행 디렉터리 여러 개와 스캔 대조(#224) compare.json을 엔진 나란히 표로 만든다.

사용: python -m eval.ocr_vlm_summary --runs eval/results/ocr296/triton eval/results/ocr296/vlm_qwen ... [--scan eval/results/scan224/compare.json]
      → stdout 마크다운. 셀 = 자리·글자 완전 일치(띄어쓰기 포함 — Triton은 띄어쓰기 소실이 여기서 깎인다, #253). 존재 = 띄어쓰기 무시하고 글자가 문서 어딘가에 있는가.
숫자 오류 = 기대가 숫자인 칸의 오독('오독' kind 중 is_numeric), 추가 = 기대에 없는 칸이 생김(환각 후보).
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from eval.table_parsing.expected import is_numeric


def _load(run: Path):
    rows = [json.loads(l) for l in (run / 'results.jsonl').read_text().splitlines() if l.strip()]
    return [r for r in rows if r['layer'] == 'l1' and r['path'] == 'img']


def table_matrix(runs: list[Path]) -> str:
    per = {}
    for run in runs:
        agg = defaultdict(lambda: dict(c=[0, 0], n=[0, 0], p=[0, 0], num_err=0, extra=0, clean=True, sec=0.0))
        tot = dict(c=[0, 0], n=[0, 0], p=[0, 0], num_err=0, extra=0, clean_axes=0, sec=0.0)
        for r in _load(run):
            a = agg[r['axis']]
            a['c'][0] += r['cells'][0]
            a['c'][1] += r['cells'][1]
            a['n'][0] += r['numeric'][0]
            a['n'][1] += r['numeric'][1]
            a['p'][0] += r['presence'][0]
            a['p'][1] += r['presence'][1]
            a['num_err'] += sum(1 for m in r['mismatches'] if m['kind'] == '오독' and is_numeric(m['expected']))
            a['extra'] += sum(1 for m in r['mismatches'] if m['kind'] == '추가')
            a['clean'] &= r['clean']
            a['sec'] = max(a['sec'], r['seconds'])
        for a in agg.values():
            tot['c'][0] += a['c'][0]
            tot['c'][1] += a['c'][1]
            tot['n'][0] += a['n'][0]
            tot['n'][1] += a['n'][1]
            tot['p'][0] += a['p'][0]
            tot['p'][1] += a['p'][1]
            tot['num_err'] += a['num_err']
            tot['extra'] += a['extra']
            tot['clean_axes'] += a['clean']
            tot['sec'] += a['sec']
        per[run.name] = (agg, tot)
    names = [r.name for r in runs]
    axes = sorted({ax for agg, _ in per.values() for ax in agg})
    out = ['| 축 | ' + ' | '.join(names) + ' |', '|---|' + '---|' * len(names)]
    for ax in axes:
        cells = []
        for n in names:
            a = per[n][0].get(ax)
            if a is None:
                cells.append('—')
                continue
            mark = '✅' if a['clean'] else '❌'
            cells.append(
                f"{mark} {a['c'][0]}/{a['c'][1]} · 존재 {a['p'][0]}/{a['p'][1]} (숫자 {a['n'][0]}/{a['n'][1]}{', 숫자오독 ' + str(a['num_err']) if a['num_err'] else ''}{', 추가 ' + str(a['extra']) if a['extra'] else ''})"
            )
        out.append(f'| {ax} | ' + ' | '.join(cells) + ' |')
    out.append(
        '| **합계** | '
        + ' | '.join(
            f"셀 **{t['c'][0]}/{t['c'][1]}** ({t['c'][0] / max(t['c'][1], 1):.1%}) · 존재(띄어쓰기 무시) **{t['p'][0]}/{t['p'][1]}** ({t['p'][0] / max(t['p'][1], 1):.1%}) · 숫자 {t['n'][0]}/{t['n'][1]} · 숫자오독 {t['num_err']} · 추가 {t['extra']} · 완전일치 축 {t['clean_axes']}/{len(axes)} · {t['sec']:.0f}s"
            for _, t in (per[n] for n in names)
        )
        + ' |'
    )
    return '\n'.join(out)


def scan_matrix(compare_json: Path) -> str:
    rows = json.loads(compare_json.read_text())
    modes = list(dict.fromkeys(r['mode'] for r in rows))
    out = ['| 문서 · 조건 | ' + ' | '.join(modes) + ' |', '|---|' + '---|' * len(modes)]
    files = list(dict.fromkeys((r['file'], r['condition']) for r in rows))
    for f, cond in files:
        cells = []
        for m in modes:
            r = next((x for x in rows if x['file'] == f and x['mode'] == m), None)
            cells.append(
                '—'
                if r is None
                else f"제목{'✓' if r['title_ok'] else '✗'} 헤딩 {r['heading_recall']:.2f} 문장 {r['sent_exact']:.2f}/{r['sent_nospace']:.2f} 글자 {r['char_acc']:.3f} 띄어쓰기 {r['space_kept']:.2f}"
            )
        out.append(f'| {f[:26]} · {cond} | ' + ' | '.join(cells) + ' |')
    avg = []
    for m in modes:
        rs = [x for x in rows if x['mode'] == m]
        avg.append(
            f"제목 {sum(x['title_ok'] for x in rs)}/{len(rs)} 헤딩 {sum(x['heading_recall'] for x in rs) / len(rs):.2f} 문장 {sum(x['sent_exact'] for x in rs) / len(rs):.2f}/{sum(x['sent_nospace'] for x in rs) / len(rs):.2f} 글자 {sum(x['char_acc'] for x in rs) / len(rs):.3f} 띄어쓰기 {sum(x['space_kept'] for x in rs) / len(rs):.2f}"
        )
    out.append('| **평균** | ' + ' | '.join(avg) + ' |')
    return '\n'.join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='*', default=[])
    ap.add_argument('--scan')
    a = ap.parse_args()
    if a.runs:
        print('## 표 이미지 22축 (L1 셀 단위, #246 하네스)\n')
        print(table_matrix([Path(r) for r in a.runs]))
    if a.scan:
        print('\n## 스캔 PDF 5건 (#224 대조)\n')
        print(scan_matrix(Path(a.scan)))


if __name__ == '__main__':
    main()
