"""실문서 파싱 충실도 검증 실행기 (#247 그림 캡션 · #231 문장 포함률).

    python -m eval.doc_fidelity.run --tenant inticube --phase extract     # 그림 전수 추출(docling 재변환 — VLM 사용)
    python -m eval.doc_fidelity.run --tenant inticube --phase judge       # Claude 1차 판정(rubric.py)
    python -m eval.doc_fidelity.run --tenant inticube --phase coverage    # 문장 포함률 + 유실 후보 Claude 판정
    python -m eval.doc_fidelity.run --phase synthetic                     # 합성 표 이미지(V12·V23·V15) 캡션 ↔ 정답
    python -m eval.doc_fidelity.run --tenant inticube --phase review      # 재판정 HTML(review.html)
    python -m eval.doc_fidelity.run --tenant inticube --phase summarize [--human human.json]

산출물은 eval/results/doc_fidelity/<tenant>/ — **gitignore**(실문서 그림·원문이 들어 있다, #247 결정 3). 커밋하는 것은 이 코드뿐.
필요: DB·OpenSearch(색인 청크), VLM(extract·synthetic), claude CLI(judge·coverage). 모듈 역할:
  pictures(추출·색인 대조) · rubric(판정 기준 정의점) · judge(Claude 판정) · coverage(문장 포함률) · synthetic · review(HTML·집계)
"""

import argparse
import asyncio
import json
from collections import Counter, defaultdict
from pathlib import Path

from eval.doc_fidelity import coverage as cov
from eval.doc_fidelity.pictures import PLACEHOLDER, extract_document, norm, record_dict
from eval.doc_fidelity.review import load_jsonl, select_for_human, summarize, write_html

ROOT = Path('eval/results/doc_fidelity')


def _dump(path: Path, rows) -> None:
    path.write_text('\n'.join(json.dumps(r, ensure_ascii=False) for r in rows))


async def _documents(tenant: str) -> list:
    from sqlalchemy import select

    from database import AsyncSessionLocal
    from rag.models import Document

    async with AsyncSessionLocal() as s:
        return (
            (
                await s.execute(
                    select(Document)
                    .where(Document.tenant_id == tenant, Document.status == 'ready', Document.is_active.is_(True))
                    .order_by(Document.id)
                )
            )
            .scalars()
            .all()
        )


async def _chunks(document_id: int) -> list[dict]:
    from config import settings
    from rag.search import os_client

    resp = await os_client.client().search(
        index=settings.opensearch_index,
        body={
            'size': 2000,
            '_source': ['text', 'heading_path', 'page'],
            'query': {'term': {'document_id': document_id}},
            'sort': [{'chunk_id': 'asc'}],
        },
    )
    return [h['_source'] for h in resp['hits']['hits']]


async def phase_extract(tenant: str, out: Path, files: list[str] | None = None) -> None:
    crops = out / 'crops'
    crops.mkdir(parents=True, exist_ok=True)
    rows = []
    for d in await _documents(tenant):
        if not d.filename.lower().endswith(('.pdf', '.docx', '.xlsx')):
            continue
        if files and not d.filename.startswith(tuple(files)):
            continue
        chunks = await _chunks(d.id)
        recs = [
            record_dict(r)
            for r in extract_document(d.id, d.filename, Path(d.blob_path), [c['text'] for c in chunks], crops)
        ]
        chunks_norm = [norm(c['text']) for c in chunks]
        for r in recs:
            if r['in_index'] is False:  # 결정 5 — 색인 쪽 캡션을 사람이 볼 수 있게 가장 비슷한 청크를 붙인다
                i, _ = cov.best_chunk(norm(r['caption']), chunks_norm)
                r['nearest_chunk'] = chunks[i]['text'][:1500] if i >= 0 else ''
        rows += recs
        print(
            f'{d.filename[:40]}: 그림 {len(recs)} · 캡션 {sum(r["captioned"] for r in recs)} · '
            f'색인 불일치 {sum(r["in_index"] is False for r in recs)}',
            flush=True,
        )
    _dump(out / 'pictures.jsonl', rows)
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
    _dump(out / 'judged.jsonl', rows)
    print(f'→ {out / "judged.jsonl"} 등급 {dict(Counter((r["verdict"] or {}).get("grade") for r in rows))}')


async def phase_coverage(tenant: str, out: Path, concurrency: int) -> None:
    from eval.claude_client import ClaudeCliClient
    from eval.doc_fidelity.judge import _JSON, DEFAULT_MODEL

    client = ClaudeCliClient(model=DEFAULT_MODEL)
    sem = asyncio.Semaphore(concurrency)
    rows = []
    for d in await _documents(tenant):
        fmt = d.filename.lower().rsplit('.', 1)[-1]
        if fmt not in cov.EXTRACTORS:
            continue
        chunks = await _chunks(d.id)
        units = cov.classify(cov.EXTRACTORS[fmt](d.id, Path(d.blob_path)), chunks)

        async def judge_loss(u, chunks=chunks):
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
                except RuntimeError as e:  # 호출 실패는 그 단위만 '판정 실패'로 — 실행 전체를 멈추지 않는다(리뷰 지적)
                    return {**u, 'loss_label': '판정 실패', 'loss_note': f'호출 실패: {str(e)[:200]}'}
            m = _JSON.search(raw or '')
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
    _dump(out / 'coverage.jsonl', rows)


async def phase_synthetic_pictures(out: Path) -> None:
    """합성 그림 문서(picture_specs.DOC_NAMES)의 추출·판정 결과를 정답과 대조. extract(--files homeplus_1)·judge 뒤에 돈다."""
    from eval.doc_fidelity.picture_specs import DOC_NAMES, PICTURES
    from eval.doc_fidelity.synthetic import match_pictures, score_picture

    # 기준은 최신 추출(pictures.jsonl)이다. 판정(judged.jsonl)은 같은 그림 id가 있을 때만 붙인다 — 판정 파일을 먼저 읽던 첫 구현은
    # 문서를 다시 올린 뒤에도 옛 추출(지운 문서 id)을 채점했다(10/18, 경계 재배치 확인 중 발견)
    verdicts = (
        {r['picture_id']: r.get('verdict') for r in load_jsonl(out / 'judged.jsonl')}
        if (out / 'judged.jsonl').exists()
        else {}
    )
    recs = [{**r, 'verdict': verdicts.get(r['picture_id'])} for r in load_jsonl(out / 'pictures.jsonl')]
    index_text = {}
    for r in recs:  # 그림으로 안 잡히고 글자(OCR)·헤딩으로 들어간 경우를 보려고 문서 색인 글을 둔다
        index_text.setdefault(r['filename'], r['document_id'])
    for name, did in list(index_text.items()):
        chunks = await _chunks(did)
        index_text[name] = (
            norm('\n'.join(c['text'] for c in chunks)),
            norm('\n'.join(h for c in chunks for h in c.get('heading_path') or [])),
        )
    rows = []
    for fmt, name in DOC_NAMES.items():
        specs = [(k, v) for k, v in PICTURES.items() if v['doc'] == fmt]
        matched = match_pictures(specs, [r for r in recs if r['filename'] == name], out / 'crops')
        uses = Counter((r or {}).get('picture_id') for _, r in matched if r)
        for sid, r in matched:
            sc = score_picture(PICTURES[sid], (r or {}).get('caption') or '')
            body, heads = index_text.get(name, ('', ''))
            texts = PICTURES[sid].get('texts', [])
            as_text = {
                'body': sum(norm(t) in body for t in texts),
                'heading': sum(norm(t) in heads and norm(t) not in body for t in texts),
            }
            rows.append(
                {
                    'id': sid,
                    'doc': fmt,
                    'found': r is not None,
                    'merged': bool(r) and uses[r['picture_id']] > 1,
                    'index_body_hit': as_text['body'],
                    'index_heading_hit': as_text['heading'],
                    'picture_id': (r or {}).get('picture_id'),
                    'area_ratio': (r or {}).get('area_ratio'),
                    'captioned': bool((r or {}).get('caption')),
                    'judge_grade': ((r or {}).get('verdict') or {}).get('grade'),
                    **sc,
                }
            )
    _dump(out / 'synthetic_pictures.jsonl', rows)
    for r in rows:
        print(
            f"{r['id']:12} {'검출' if r['found'] else '미검출'}{'(합쳐짐)' if r.get('merged') else ''} 면적 {r['area_ratio']} 캡션 {'O' if r['captioned'] else 'X'} "
            f"글자 {r['texts_hit']}/{r['texts']} 숫자 {r['numbers_hit']}/{r['numbers']} 짝 {r['pairs_hit']}/{r['pairs']} "
            f"순서 {r['sequence_ok']} 판정 {r['judge_grade']} | 색인 본문 {r['index_body_hit']}/{r['texts']} 헤딩 {r['index_heading_hit']}"
        )


def chunk_view(r: dict, chunks: list[dict]) -> list[dict]:
    """그림 하나가 실제로 들어간 색인 청크들(문서 순서).

    캡션이 있으면: 그 캡션에만 있는 줄(정규화 4자 이상, 3개 넘는 청크에 나오는 줄은 뺀다)을 덮는 **최소 청크**를 탐욕적으로 고른다.
    줄 하나라도 든 청크를 전부 보이던 첫 구현은 타임레포트 화면 캡처마다 청크 7~11개를 붙였다 — 'Hansol Time Report'·'Light
    Mode' 같은 공통 줄이 모든 캡처에 있다(10/18 실측). 긴 캡션은 분할기가 두 청크로 나누므로 2개가 정상일 수 있다.
    캡션이 없으면: 자리표시가 든 청크. 청크의 page는 **첫 줄의 쪽**이라 그림 쪽 바로 앞 쪽에서 시작한 청크도 본다.
    PDF는 쪽을 그림 쪽·바로 앞 쪽으로 한정한다.
    """
    page = r.get('page')
    near = [c for c in chunks if page is None or c.get('page') is None or page - 1 <= c['page'] <= page]
    lines = [n for n in (norm(x) for x in (r.get('caption') or '').split('\n')) if len(n) >= 4]
    if not lines:
        hits = [c for c in near if PLACEHOLDER in c['text']]
        return [
            {
                'heading_path': c.get('heading_path') or [],
                'page': c.get('page'),
                'text': c['text'],
                'note': f'이 범위의 자리표시 {sum(c["text"].count(PLACEHOLDER) for c in hits)}개 중 하나',
            }
            for c in hits
        ]
    df = {n: sum(n in norm(c['text']) for c in chunks) for n in lines}
    want = {n for n in lines if 0 < df[n] <= 3} or {n for n in lines if df[n] > 0}
    picked, covered = [], set()
    while want - covered:
        best = max(near, key=lambda c: len({n for n in want - covered if n in norm(c['text'])}), default=None)
        gain = {n for n in want - covered if best is not None and n in norm(best['text'])}
        if not gain:
            break
        picked.append(best)
        covered |= gain
    order = {id(c): i for i, c in enumerate(chunks)}
    return [
        {'heading_path': c.get('heading_path') or [], 'page': c.get('page'), 'text': c['text'], 'note': ''}
        for c in sorted(picked, key=lambda c: order[id(c)])
    ]


def phase_ab_review(variants: list[str], tenant: str = 'inticube') -> None:
    """두 변형 캡션 블라인드 비교 페이지(결정 15). variants = [기준, 후보]. 실문서(inticube)는 캡션 대상이던 그림만,
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


async def phase_docview(tenant: str, out: Path, files: list[str] | None) -> None:
    """문서 전체 청킹 보기. --tenant는 쉼표로 여럿(합쳐서 한 페이지 → eval/results/doc_fidelity/docview.html).
    --files가 없으면 그 테넌트에서 그림이 있는 문서(pictures.jsonl)만."""
    from eval.doc_fidelity.review import write_docview_html

    tenants = tenant.split(',')
    docs, caps = [], defaultdict(list)
    for t in tenants:
        pics_path = ROOT / t / 'pictures.jsonl'
        pics = load_jsonl(pics_path) if pics_path.exists() else []
        for r in pics:
            if r.get('caption'):
                caps[r['filename']].append(r['caption'])
        for d in await _documents(t):
            if (files and d.filename.startswith(tuple(files))) or (
                not files and d.filename in {r['filename'] for r in pics}
            ):
                docs.append((f'[{t}] {d.filename}', await _chunks(d.id)))
                caps[f'[{t}] {d.filename}'] = caps.get(d.filename, [])
    target = (ROOT if len(tenants) > 1 else out) / 'docview.html'
    print(f'→ {write_docview_html(docs, caps, target)} (문서 {len(docs)}건)')


async def phase_review(out: Path) -> None:
    from eval.doc_fidelity.review import write_coverage_html

    picked = select_for_human(load_jsonl(out / 'judged.jsonl'))
    docs = {d: await _chunks(d) for d in {r['document_id'] for r in picked}}
    for r in picked:  # 사용자가 실제로 받는 것 = 색인 청크(OpenSearch 조회값) — 캡션 옆에 둔다
        r['chunk_view'] = chunk_view(r, docs[r['document_id']])
    p = write_html(picked, out / 'review.html')
    print(f'→ {p} ({len(picked)}장 — {dict(Counter(r["review_reason"] for r in picked))})')
    if (out / 'coverage.jsonl').exists():
        lost = [u for u in load_jsonl(out / 'coverage.jsonl') if u.get('loss_label') in ('유실', '판정 실패')]
        chunks = {d: await _chunks(d) for d in {u['document_id'] for u in lost}}
        print(f'→ {write_coverage_html(lost, chunks, out / "coverage_review.html")} ({len(lost)}건)')


def coverage_table(rows: list[dict]) -> list[str]:
    by = defaultdict(Counter)
    for u in rows:
        cls = u['class'] if u['class'] != '유실 후보' else f"유실 후보·{u.get('loss_label', '?')}"
        by[(u['fmt'], u['element'])][cls] += 1
    lines = [
        '| 형식 | 요소 | 단위 | 본문 | 표 | 헤딩 | 부분 | 유실 후보→유실 | →변형 | →잡음 | 포함률 |',
        '|---|---|---|---|---|---|---|---|---|---|---|',
    ]
    for (fmt, el), c in sorted(by.items()):
        total = sum(c.values())
        lost = c['유실 후보·유실'] + c['유실 후보·판정 실패']
        lines.append(
            f"| {fmt} | {el} | {total} | {c['본문']} | {c['표']} | {c['헤딩']} | {c['부분']} | {lost} | "
            f"{c['유실 후보·변형']} | {c['유실 후보·잡음']} | {1 - lost / total:.3f} |"
        )
    return lines


def phase_summarize(out: Path, human_path: Path | None) -> None:
    human = json.loads(human_path.read_text()) if human_path and human_path.exists() else None
    pics = summarize(load_jsonl(out / 'judged.jsonl'), human)
    lines = ['# 실문서 파싱 충실도 요약 (#247 그림 · #231 문장)', '', '## 그림 캡션']
    if pics['judge_errors']:
        lines.append(
            f"> ⚠️ 판정 실패 {pics['judge_errors']}장 — 판정 수·오류율에서 빠졌다. `--phase judge --retry-errors`로 다시 돌릴 것"
        )
    lines += ['```', json.dumps(pics, ensure_ascii=False, indent=1), '```']
    if (out / 'coverage.jsonl').exists():
        lines += ['', '## 문장 포함률', *coverage_table(load_jsonl(out / 'coverage.jsonl'))]
    syn = ROOT / 'synthetic' / 'synthetic.jsonl'
    if syn.exists():
        lines += ['', '## 합성 표 이미지(그림 분류) 캡션', '| 정의 | 분류 | 칸 값 | 숫자 칸 |', '|---|---|---|---|']
        lines += [
            f"| {r['id']} | {r['classified']} | {r['values_hit']}/{r['values']} | {r['numeric_hit']}/{r['numeric']} |"
            for r in load_jsonl(syn)
        ]
    (out / 'summary.md').write_text('\n'.join(lines))
    print('\n'.join(lines))


async def main():
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

        _dump(ROOT / 'synthetic' / 'synthetic.jsonl', run(ROOT / 'synthetic'))
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
