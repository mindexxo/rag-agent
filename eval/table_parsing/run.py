"""#246 실행기 — 합성 표 46개 × 4경로를 워커와 같은 경로로 변환해 두 층을 채점하고 매트릭스를 쓴다.

용법:
    python -m eval.table_parsing.run                      # 전체(변환 약 4분 + 모델 로딩, OCR·VLM 서버 필요)
    python -m eval.table_parsing.run --only S01_basic S02_colspan --paths img pdf
    python -m eval.table_parsing.run --out eval/results/table_parsing/run2
    python -m eval.table_parsing.run --engine vlm --paths img   # #296: 쪽 전체를 VLM(VLM_CAPTION_*)으로 — eval/vlm_engine.py

산출물(<out>/): docs/(생성 문서·png), results.jsonl(건별·즉시 기록), matrix_l1.md·matrix_l2.md(축 × 경로),
mismatches.md(틀린 칸 기대↔실제 나란히 + 실제 markdown 행). 변환은 `_convert_local`(docling 인프로세스 — 워커의 변환
프로세스와 같은 변환기) 한 번으로 L1을 채점하고, 그 문서를 `chunk_file`에 끼워 L2를 낸다(두 번 변환하지 않는다 —
OCR 시간이 두 배가 된다). L2는 분할기(SentenceSplitter 512·겹침 50) 흔적까지 **색인에 들어간 그대로** 잰다(결정 a).
"""
import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import rag.chunking as ch
import rag.xlsx_chunking as xc
from eval.table_parsing.build import build_all
from eval.table_parsing.expected import expected_cells, expected_rows, expected_tables, flat
from eval.table_parsing.score import (Score, all_row_texts, all_texts, chunk_tables, docling_tables, match_rows,
                                      match_tables, score_cells, score_rows)
from eval.table_parsing.specs import BY_ID, PATH_LABEL, paths_of

PATHS = ('img', 'pdf', 'docx', 'xlsx')
_ENGINE = 'triton'          # --engine 값(결과 행 자기서술, #296)


def _score_pair(spec, path, doc, chunks):
    tables_l1 = expected_tables(spec, path, 'l1')
    tables = expected_tables(spec, path, 'l2')
    actuals = docling_tables(doc)
    pool = all_texts(actuals)
    l1 = [score_cells(expected_cells(t), a, flat(t), pool)
          for t, a in zip(tables_l1, match_tables([expected_cells(t) for t in tables_l1], actuals))]
    exp_rows = [expected_rows(spec, t, path) for t in tables]
    found = chunk_tables(chunks, xlsx=path == 'xlsx')
    pool2 = all_row_texts(found)
    actual_l2 = match_rows(exp_rows, found)
    l2 = [score_rows(e, a, pool2) for e, a in zip(exp_rows, actual_l2)]
    return l1, l2, actual_l2


def _convert_and_chunk(path_file: Path, path: str):
    doc = ch._convert_local(path_file, **({'max_num_pages': 100} if path in ('img', 'pdf') else {}))
    orig_ch, orig_xc = ch.docling_convert, xc.docling_convert
    ch.docling_convert = xc.docling_convert = lambda p, **k: doc      # 같은 변환 결과로 청킹(두 번 변환 안 함)
    try:
        chunks = ch.chunk_file(path_file, description='표') if path == 'xlsx' else ch.chunk_file(path_file)
    except Exception as e:                                             # noqa: BLE001 — 청킹 실패도 결과다(L2만)
        chunks = e
    finally:
        ch.docling_convert, xc.docling_convert = orig_ch, orig_xc
    return doc, chunks


def _failed(spec, kind: str, err: Exception, path: str = 'docx') -> list[Score]:
    out = []
    for tbl in expected_tables(spec, path, 'l1'):
        s = score_cells(expected_cells(tbl), None, flat(tbl))
        s.mismatches.append(dict(r=-1, c=-1, kind=kind, expected='', actual=f'{type(err).__name__}: {str(err)[:120]}'))
        out.append(s)
    return out


def _row(spec, path, k, s: Score, layer, seconds, actual_rows=None):
    return dict(id=spec['id'], axis=spec['axis'], path=path, table=k, layer=layer, seconds=round(seconds, 1), engine=_ENGINE,
                detected=s.detected, dims_ok=s.dims_ok, clean=s.clean, exp_dims=s.exp_dims, act_dims=s.act_dims,
                cells=[s.cells_ok, s.cells_total], numeric=[s.numeric_ok, s.numeric_total],
                spans=[s.span_ok, s.span_total], presence=[s.presence_ok, s.presence_total],
                header_rows=s.header_rows, kinds=s.kinds, mismatches=s.mismatches, actual_rows=actual_rows)


def _matrix(rows, layer) -> str:
    """축 × 경로 — 칸: 셀 일치/전체 (숫자 일치/전체) + 오류 유형. 같은 축의 표들(짝 2개·나란히·중첩)을 합친다."""
    agg = defaultdict(lambda: dict(c=[0, 0], n=[0, 0], kinds=set(), present=[0, 0], clean=True))
    for r in rows:
        if r['layer'] != layer:
            continue
        a = agg[(r['axis'], r['path'])]
        a['c'][0] += r['cells'][0]; a['c'][1] += r['cells'][1]
        a['n'][0] += r['numeric'][0]; a['n'][1] += r['numeric'][1]
        a['present'][0] += r['presence'][0]; a['present'][1] += r['presence'][1]
        a['kinds'] |= set(r['kinds'])
        a['clean'] &= r['clean']
    axes = sorted({r['axis'] for r in rows})
    lines = ['| 축 | ' + ' | '.join(PATH_LABEL[p] for p in PATHS) + ' |', '|---|' + '---|' * len(PATHS)]
    for ax in axes:
        cells = []
        for p in PATHS:
            a = agg.get((ax, p))
            if a is None:
                cells.append('—')
                continue
            c, n, pr = a['c'], a['n'], a['present']
            mark = '✅' if a['clean'] else ('⚠️' if pr[0] == pr[1] else '❌')
            kinds = f" {'·'.join(sorted(a['kinds']))}" if a['kinds'] else ''
            cells.append(f"{mark} {c[0]}/{c[1]} (숫자 {n[0]}/{n[1]}){kinds}")
        lines.append(f'| {ax} | ' + ' | '.join(cells) + ' |')
    lines += ['', '✅ 표 검출·행열 수·병합 범위·모든 칸 일치 · ⚠️ 글자(띄어쓰기 무시)는 문서 어딘가에 다 있으나 자리·병합·구조가 다름 '
                  '· ❌ 글자 유실/오독. 괄호는 숫자·금액·날짜 칸 완전 일치.']
    if layer == 'l2':
        lines.append('L2는 청크 분할기 흔적(긴 표의 경계 절단·겹침 중복)까지 색인에 들어간 그대로 잰다 — 긴 표 축의 수치는 표 렌더가 아니라 분할 결과다.')
    return '\n'.join(lines)


def _mismatch_report(rows) -> str:
    out = ['# 틀린 칸 — 기대 ↔ 실제', '']
    for r in rows:
        if r['clean']:
            continue
        out.append(f"## {r['id']} · {r['path']} · 표{r['table']} · L{1 if r['layer'] == 'l1' else 2} — "
                   f"{'·'.join(r['kinds'])} (기대 {r['exp_dims']} / 실제 {r['act_dims']}, 헤더행 {r['header_rows']}, "
                   f"존재 {r['presence'][0]}/{r['presence'][1]})")
        for m in r['mismatches'][:40]:
            out.append(f"- ({m['r']},{m['c']}) {m['kind']}: `{m['expected']}` → `{m['actual']}`")
        if r['layer'] == 'l2' and r.get('actual_rows') is not None:
            out += ['', '실제 markdown 행:', '```']
            out += ['| ' + ' | '.join(x) + ' |' for x in (r['actual_rows'] or [['(없음)']])[:12]]
            out.append('```')
        out.append('')
    return '\n'.join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='eval/results/table_parsing/run1')
    ap.add_argument('--only', nargs='*')
    ap.add_argument('--paths', nargs='*', default=list(PATHS))
    ap.add_argument('--engine', choices=('triton', 'vlm'), default='triton',
                    help="triton=현행(레이아웃+TableFormer+원격 RapidOCR) | vlm=쪽 전체 VLM(#296, PDF·표 이미지만 영향)")
    args = ap.parse_args()
    global _ENGINE
    if args.engine == 'vlm':
        from eval import vlm_engine
        _ENGINE = vlm_engine.label()
        vlm_engine.install()
        print(f'engine={vlm_engine.label()} scale={vlm_engine.SCALE} max_tokens={vlm_engine.MAX_TOKENS}', flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    made = build_all(out / 'docs', set(args.only) if args.only else None, set(args.paths))
    rows = []
    with open(out / 'results.jsonl', 'w') as f:
        def emit(r):
            rows.append(r)
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
            f.flush()

        for spec_id, path, file in made:
            spec = BY_ID[spec_id]
            t = time.time()
            try:
                doc, chunks = _convert_and_chunk(file, path)
            except Exception as e:                                   # noqa: BLE001 — 변환 실패도 결과다
                sec = time.time() - t
                for k, s in enumerate(_failed(spec, '변환실패', e, path)):
                    emit(_row(spec, path, k, s, 'l1', sec))
                    emit(_row(spec, path, k, s, 'l2', sec))
                print(f'{spec_id:16} {path:5} 변환 실패 {type(e).__name__}: {str(e)[:80]}', flush=True)
                continue
            sec = time.time() - t
            try:
                if isinstance(chunks, Exception):
                    raise chunks
                l1, l2, act_rows = _score_pair(spec, path, doc, chunks)
            except Exception as e:                                   # noqa: BLE001 — 청킹·채점 실패를 결과에 남긴다
                tables = expected_tables(spec, path, 'l1')
                actuals = docling_tables(doc)
                l1 = [score_cells(expected_cells(tb), a, flat(tb), all_texts(actuals))
                      for tb, a in zip(tables, match_tables([expected_cells(tb) for tb in tables], actuals))]
                l2 = _failed(spec, '청킹실패', e, path)
                act_rows = [None] * len(l2)
            for k, s1 in enumerate(l1):                       # L1·L2 표 수가 다를 수 있다(DOCX 중첩: L1 2개, L2 1개)
                emit(_row(spec, path, k, s1, 'l1', sec))
            for k, s2 in enumerate(l2):
                emit(_row(spec, path, k, s2, 'l2', sec, act_rows[k]))
            print(f"{spec_id:16} {path:5} {sec:5.1f}s L1 {sum(s.cells_ok for s in l1)}/{sum(s.cells_total for s in l1)} "
                  f"L2 {sum(s.cells_ok for s in l2)}/{sum(s.cells_total for s in l2)} "
                  f"{'·'.join(k for s in l1 for k in s.kinds) or 'ok'}", flush=True)
    (out / 'matrix_l1.md').write_text('# L1 — docling 추출 정확도(셀 격자 vs 정의)\n\n' + _matrix(rows, 'l1') + '\n')
    (out / 'matrix_l2.md').write_text('# L2 — 색인 결과 정확도(청크 markdown vs 기대 렌더)\n\n' + _matrix(rows, 'l2') + '\n')
    (out / 'mismatches.md').write_text(_mismatch_report(rows))
    print('결과:', out)


if __name__ == '__main__':
    main()
