"""실문서 파싱 충실도 검증 실행기 (그림 캡션 · 문장 포함률).

    python -m eval.doc_fidelity.run --tenant inticube --phase extract     # 그림 전수 추출(docling 재변환 — VLM 사용)
    python -m eval.doc_fidelity.run --tenant inticube --phase judge       # Claude 1차 판정(rubric.py)
    python -m eval.doc_fidelity.run --tenant inticube --phase coverage    # 문장 포함률 + 유실 후보 Claude 판정
    python -m eval.doc_fidelity.run --phase synthetic                     # 합성 표 이미지(V12·V23·V15) 캡션 ↔ 정답
    python -m eval.doc_fidelity.run --tenant inticube --phase review      # 재판정 HTML(review.html)
    python -m eval.doc_fidelity.run --tenant inticube --phase summarize [--human human.json]

산출물은 eval/results/doc_fidelity/<tenant>/ — **gitignore**(실문서 그림·원문이 들어 있다). 커밋하는 것은 이 코드뿐.
필요: DB·OpenSearch(색인 청크), VLM(extract·synthetic), claude CLI(judge·coverage). 모듈 역할:
  pictures(추출·색인 대조) · rubric(판정 기준 정의점) · judge(Claude 판정) · coverage(문장 포함률) · synthetic · review(HTML·집계)
  · sources(결과 경로·DB·색인 조회) · report_phases(review·docview·summarize 단계)
"""

import argparse
import asyncio
import json
from collections import Counter, defaultdict
from pathlib import Path

from eval.doc_fidelity import coverage as cov
from eval.doc_fidelity.pictures import extract_document, norm, record_dict
from eval.doc_fidelity.report_phases import chunk_view, phase_docview, phase_review, phase_summarize
from eval.doc_fidelity.review import load_jsonl
from eval.doc_fidelity.sources import ROOT, dump_jsonl, index_chunks, ready_documents

__all__ = ['chunk_view']  # 테스트가 이 모듈에서 가져간다


async def phase_extract(tenant: str, out: Path, files: list[str] | None = None) -> None:
    crops = out / 'crops'
    crops.mkdir(parents=True, exist_ok=True)
    rows = []
    for d in await ready_documents(tenant):
        if not d.filename.lower().endswith(('.pdf', '.docx', '.xlsx')):
            continue
        if files and not d.filename.startswith(tuple(files)):
            continue
        chunks = await index_chunks(d.id)
        recs = [
            record_dict(r)
            for r in extract_document(d.id, d.filename, Path(d.blob_path), [c['text'] for c in chunks], crops)
        ]
        chunks_norm = [norm(c['text']) for c in chunks]
        for r in recs:
            if r['in_index'] is False:  # 색인 쪽 캡션을 사람이 볼 수 있게 가장 비슷한 청크를 붙인다
                i, _ = cov.best_chunk(norm(r['caption']), chunks_norm)
                r['nearest_chunk'] = chunks[i]['text'][:1500] if i >= 0 else ''
        rows += recs
        print(
            f'{d.filename[:40]}: 그림 {len(recs)} · 캡션 {sum(r["captioned"] for r in recs)} · '
            f'색인 불일치 {sum(r["in_index"] is False for r in recs)}',
            flush=True,
        )
    dump_jsonl(out / 'pictures.jsonl', rows)
    print(f'→ {out / "pictures.jsonl"} ({len(rows)}장)')


async def phase_judge(out: Path, concurrency: int, retry_errors: bool = False) -> None:
    from eval.doc_fidelity.judge import judge_all

    if retry_errors and (out / 'judged.jsonl').exists():  # 판정 실패(error)한 그림만 다시 — 나머지 판정은 그대로 둔다
        prev = load_jsonl(out / 'judged.jsonl')
        redo = [{k: v for k, v in r.items() if k != 'verdict'} for r in prev if (r.get('verdict') or {}).get('error')]
        done = {
            r['picture_id']: r
            for r in await judge_all(redo, out / 'crops', concurrency=concurrency, sink=out / 'judged.partial.jsonl')
        }
        rows = [done.get(r['picture_id'], r) for r in prev]
    else:
        rows = await judge_all(
            load_jsonl(out / 'pictures.jsonl'),
            out / 'crops',
            concurrency=concurrency,
            sink=out / 'judged.partial.jsonl',
        )  # 진행분 — 끝나면 judged.jsonl로 대체
    dump_jsonl(out / 'judged.jsonl', rows)
    print(f'→ {out / "judged.jsonl"} 등급 {dict(Counter((r["verdict"] or {}).get("grade") for r in rows))}')


async def phase_coverage(tenant: str, out: Path, concurrency: int) -> None:
    from eval.claude_client import ClaudeCliClient
    from eval.doc_fidelity.judge import DEFAULT_MODEL, JSON_OBJECT

    client = ClaudeCliClient(model=DEFAULT_MODEL)
    sem = asyncio.Semaphore(concurrency)
    rows = []
    for d in await ready_documents(tenant):
        fmt = d.filename.lower().rsplit('.', 1)[-1]
        if fmt not in cov.EXTRACTORS:
            continue
        chunks = await index_chunks(d.id)
        units = cov.classify(cov.EXTRACTORS[fmt](d.id, Path(d.blob_path)), chunks)

        async def judge_loss(u: dict, chunks: list[dict] = chunks) -> dict:
            async with sem:
                try:
                    raw = await client.acomplete(
                        [
                            {
                                'role': 'user',
                                'content': cov.LOSS_PROMPT.format(
                                    unit=u['text'],
                                    chunk=chunks[u['best_chunk']]['text'][:2000] if u['best_chunk'] >= 0 else '(없음)',
                                ),
                            }
                        ]
                    )
                except RuntimeError as e:  # 호출 실패는 그 단위만 '판정 실패'로 — 실행 전체를 멈추지 않는다
                    return {**u, 'loss_label': '판정 실패', 'loss_note': f'호출 실패: {str(e)[:200]}'}
            m = JSON_OBJECT.search(raw or '')
            try:
                j = json.loads(m.group(0)) if m else {}
            except ValueError:
                j = {}
            return {**u, 'loss_label': j.get('label', '판정 실패'), 'loss_note': j.get('note', raw[:200])}

        judged = await asyncio.gather(*(judge_loss(u) for u in units if u['class'] == '유실 후보'))
        it = iter(judged)
        units = [next(it) if u['class'] == '유실 후보' else u for u in units]
        for u in units:
            u['filename'] = d.filename
        rows += units
        c = Counter(u['class'] for u in units)
        print(f'{d.filename[:40]}: 단위 {len(units)} {dict(c)}', flush=True)
    dump_jsonl(out / 'coverage.jsonl', rows)


async def _index_texts(recs: list[dict]) -> dict[str, tuple[str, str]]:
    """파일명 → (색인 본문, 색인 헤딩) 정규화 글. 그림으로 안 잡히고 글자(OCR)·헤딩으로 들어간 경우를 보는 데 쓴다."""
    doc_ids: dict[str, int] = {}
    for r in recs:
        doc_ids.setdefault(r['filename'], r['document_id'])
    texts = {}
    for name, did in doc_ids.items():
        chunks = await index_chunks(did)
        texts[name] = (
            norm('\n'.join(c['text'] for c in chunks)),
            norm('\n'.join(h for c in chunks for h in c.get('heading_path') or [])),
        )
    return texts


def _synthetic_row(sid: str, fmt: str, r: dict | None, merged: bool, index_text: tuple[str, str]) -> dict:
    """정의 하나의 대조 결과 — 짝지은 그림의 캡션 채점과, 정답 글자가 색인 본문·헤딩에 들어갔는지."""
    from eval.doc_fidelity.picture_specs import PICTURES
    from eval.doc_fidelity.synthetic import score_picture

    sc = score_picture(PICTURES[sid], (r or {}).get('caption') or '')
    body, heads = index_text
    texts = PICTURES[sid].get('texts', [])
    return {
        'id': sid,
        'doc': fmt,
        'found': r is not None,
        'merged': merged,
        'index_body_hit': sum(norm(t) in body for t in texts),
        'index_heading_hit': sum(norm(t) in heads and norm(t) not in body for t in texts),
        'picture_id': (r or {}).get('picture_id'),
        'area_ratio': (r or {}).get('area_ratio'),
        'captioned': bool((r or {}).get('caption')),
        'judge_grade': ((r or {}).get('verdict') or {}).get('grade'),
        **sc,
    }


async def phase_synthetic_pictures(out: Path) -> None:
    """합성 그림 문서(picture_specs.DOC_NAMES)의 추출·판정 결과를 정답과 대조. extract(--files homeplus_1)·judge 뒤에 돈다."""
    from eval.doc_fidelity.picture_specs import DOC_NAMES, PICTURES
    from eval.doc_fidelity.synthetic import match_pictures

    # 기준은 최신 추출(pictures.jsonl)이다. 판정(judged.jsonl)은 같은 그림 id가 있을 때만 붙인다 — 판정 파일을 기준으로 삼으면
    # 문서를 다시 올린 뒤에도 옛 추출(지운 문서 id)을 채점한다
    verdicts = (
        {r['picture_id']: r.get('verdict') for r in load_jsonl(out / 'judged.jsonl')}
        if (out / 'judged.jsonl').exists()
        else {}
    )
    recs = [{**r, 'verdict': verdicts.get(r['picture_id'])} for r in load_jsonl(out / 'pictures.jsonl')]
    index_text = await _index_texts(recs)
    rows = []
    for fmt, name in DOC_NAMES.items():
        specs = [(k, v) for k, v in PICTURES.items() if v['doc'] == fmt]
        matched = match_pictures(specs, [r for r in recs if r['filename'] == name], out / 'crops')
        uses = Counter((r or {}).get('picture_id') for _, r in matched if r)
        for sid, r in matched:
            merged = bool(r) and uses[r['picture_id']] > 1
            rows.append(_synthetic_row(sid, fmt, r, merged, index_text.get(name, ('', ''))))
    dump_jsonl(out / 'synthetic_pictures.jsonl', rows)
    for r in rows:
        print(
            f"{r['id']:12} {'검출' if r['found'] else '미검출'}{'(합쳐짐)' if r.get('merged') else ''} 면적 {r['area_ratio']} 캡션 {'O' if r['captioned'] else 'X'} "
            f"글자 {r['texts_hit']}/{r['texts']} 숫자 {r['numbers_hit']}/{r['numbers']} 짝 {r['pairs_hit']}/{r['pairs']} "
            f"순서 {r['sequence_ok']} 판정 {r['judge_grade']} | 색인 본문 {r['index_body_hit']}/{r['texts']} 헤딩 {r['index_heading_hit']}"
        )


def phase_ab_review(variants: list[str], tenant: str = 'inticube') -> None:
    """두 변형 캡션 블라인드 비교 페이지. variants = [기준, 후보]. 실문서(inticube)는 캡션 대상이던 그림만,
    합성(homeplus)은 변형 캡션이 있는 그림 전부(XLSX처럼 운영에서 캡션이 안 붙는 그림도 프롬프트 비교에는 넣는다)."""
    from eval.doc_fidelity.review import write_ab_html

    base, cand = variants
    cap = {}
    for r in load_jsonl(ROOT / 'prompt_ab.jsonl'):
        cap[(r['picture_id'], r['variant'])] = r['caption']
    items = [
        {
            'picture_id': r['picture_id'],
            'crop': r['crop'],
            'filename': r['filename'],
            'base': cap[(r['picture_id'], base)],
            'cand': cap[(r['picture_id'], cand)],
        }
        for r in load_jsonl(ROOT / tenant / 'pictures.jsonl')
        if (r.get('captioned') or tenant != 'inticube')
        and r.get('crop')
        and (r['picture_id'], base) in cap
        and (r['picture_id'], cand) in cap
    ]
    out = ROOT / tenant
    p = write_ab_html(items, out / 'ab_review.html', out / 'ab_key.json')
    print(f'→ {p} ({len(items)}장, 기준 {base} / 후보 {cand}, 정답 매핑 {out / "ab_key.json"})')


def phase_prompt_ab(only: list[str] | None) -> None:
    """캡션 프롬프트 변형 비교(prompt_ab.py). 합성 그림(homeplus)은 정답 자동 채점표, 실문서(inticube)는 변형별 캡션만."""
    from eval.doc_fidelity import prompt_ab as ab
    from eval.doc_fidelity.picture_specs import PICTURES

    variants = {k: v for k, v in ab.VARIANTS.items() if not only or k in only}
    syn = [r for r in load_jsonl(ROOT / 'homeplus' / 'synthetic_pictures.jsonl') if r.get('picture_id')]
    spec_of = {}
    for r in syn:
        spec_of.setdefault(r['picture_id'], []).append(r['id'])
    real = [r for r in load_jsonl(ROOT / 'inticube' / 'pictures.jsonl') if r.get('crop')]
    targets = [(pid, ROOT / 'homeplus' / 'crops' / f"{pid.replace('-', '_')}.png") for pid in spec_of] + [
        (r['picture_id'], ROOT / 'inticube' / 'crops' / r['crop']) for r in real
    ]
    rows = ab.run(targets, variants, ROOT / 'prompt_ab.jsonl')
    table = defaultdict(lambda: Counter())
    for r in rows:
        if r['variant'] not in variants or r['picture_id'] not in spec_of:
            continue
        for sid in spec_of[r['picture_id']]:
            sc = ab.score_variant(sid, PICTURES[sid], r['caption'])
            t = table[r['variant']]
            for k in (
                'texts',
                'texts_hit',
                'numbers',
                'numbers_hit',
                'pairs',
                'pairs_hit',
                'edges',
                'edges_hit',
                'english',
                'english_hit',
            ):
                t[k] += sc.get(k) or 0
            t['sequence_ok'] += bool(sc.get('sequence_ok'))
            t['blank_hallucinated'] += bool(sc.get('blank_hallucinated'))
    lines = [
        '| 변형 | 글자 | 숫자 | 짝 | 흐름도 화살표 | 영문 유지 | 순서 OK | 빈 칸 환각 |',
        '|---|---|---|---|---|---|---|---|',
    ]
    for name in variants:
        t = table[name]
        lines.append(
            f"| {name} | {t['texts_hit']}/{t['texts']} | {t['numbers_hit']}/{t['numbers']} | {t['pairs_hit']}/{t['pairs']} | "
            f"{t['edges_hit']}/{t['edges']} | {t['english_hit']}/{t['english']} | {t['sequence_ok']} | {t['blank_hallucinated']} |"
        )
    (ROOT / 'prompt_ab_summary.md').write_text('\n'.join(lines))
    print('\n'.join(lines))


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--tenant', default='inticube')
    ap.add_argument(
        '--phase',
        required=True,
        choices=[
            'extract',
            'judge',
            'coverage',
            'synthetic',
            'synthetic-pictures',
            'review',
            'docview',
            'prompt-ab',
            'ab-review',
            'summarize',
        ],
    )
    ap.add_argument('--concurrency', type=int, default=3)
    ap.add_argument('--human', type=Path)
    ap.add_argument('--retry-errors', action='store_true', help='judge: 판정 실패한 그림만 다시')
    ap.add_argument('--variants', nargs='*', help='prompt-ab: 돌릴 변형 이름(기본 전부)')
    ap.add_argument('--files', nargs='*', help='이 파일명 접두로 시작하는 문서만(합성 그림 문서 등)')
    args = ap.parse_args()
    out = ROOT / args.tenant.split(',')[0]
    out.mkdir(parents=True, exist_ok=True)
    if args.phase == 'extract':
        await phase_extract(args.tenant, out, args.files)
    elif args.phase == 'judge':
        await phase_judge(out, args.concurrency, args.retry_errors)
    elif args.phase == 'coverage':
        await phase_coverage(args.tenant, out, args.concurrency)
    elif args.phase == 'synthetic':
        from eval.doc_fidelity.synthetic import run

        dump_jsonl(ROOT / 'synthetic' / 'synthetic.jsonl', run(ROOT / 'synthetic'))
    elif args.phase == 'synthetic-pictures':
        await phase_synthetic_pictures(out)
    elif args.phase == 'ab-review':
        phase_ab_review(args.variants or ['V0_현행', 'V11_현행+빈칸+화살표+차트V6'], args.tenant)
    elif args.phase == 'prompt-ab':
        phase_prompt_ab(args.variants)
    elif args.phase == 'docview':
        await phase_docview(args.tenant, out, args.files)
    elif args.phase == 'review':
        await phase_review(out)
    else:
        phase_summarize(out, args.human)


if __name__ == '__main__':
    asyncio.run(main())
