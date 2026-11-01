"""판정·포함률 결과를 사람이 읽는 산출물(재판정 HTML·문서 청킹 보기·요약)로 내는 실행 단계."""

import json
from collections import Counter, defaultdict
from pathlib import Path

from eval.doc_fidelity.pictures import PLACEHOLDER, norm
from eval.doc_fidelity.review import load_jsonl, select_for_human, summarize, write_html
from eval.doc_fidelity.sources import ROOT, index_chunks, ready_documents


def chunk_view(r: dict, chunks: list[dict]) -> list[dict]:
    """그림 하나가 실제로 들어간 색인 청크들(문서 순서).

    캡션이 있으면: 그 캡션에만 있는 줄(정규화 4자 이상, 3개 넘는 청크에 나오는 줄은 뺀다)을 덮는 **최소 청크**를 탐욕적으로 고른다.
    줄 하나라도 든 청크를 전부 보이면 화면 캡처마다 청크가 너무 많이 붙는다 — 'Light Mode' 같은 공통 줄이
    모든 캡처에 있기 때문이다. 긴 캡션은 분할기가 두 청크로 나누므로 2개가 정상일 수 있다.
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
    picked: list[dict] = []
    covered: set[str] = set()
    while want - covered:
        best = max(near, key=lambda c: len({n for n in want - covered if n in norm(c['text'])}), default=None)
        gain = {n for n in want - covered if best is not None and n in norm(best['text'])}
        if best is None or not gain:
            break
        picked.append(best)
        covered |= gain
    order = {id(c): i for i, c in enumerate(chunks)}
    return [
        {'heading_path': c.get('heading_path') or [], 'page': c.get('page'), 'text': c['text'], 'note': ''}
        for c in sorted(picked, key=lambda c: order[id(c)])
    ]


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
        for d in await ready_documents(t):
            if (files and d.filename.startswith(tuple(files))) or (
                not files and d.filename in {r['filename'] for r in pics}
            ):
                docs.append((f'[{t}] {d.filename}', await index_chunks(d.id)))
                caps[f'[{t}] {d.filename}'] = caps.get(d.filename, [])
    target = (ROOT if len(tenants) > 1 else out) / 'docview.html'
    print(f'→ {write_docview_html(docs, caps, target)} (문서 {len(docs)}건)')


async def phase_review(out: Path) -> None:
    from eval.doc_fidelity.review import write_coverage_html

    picked = select_for_human(load_jsonl(out / 'judged.jsonl'))
    docs = {d: await index_chunks(d) for d in {r['document_id'] for r in picked}}
    for r in picked:  # 사용자가 실제로 받는 것 = 색인 청크(OpenSearch 조회값) — 캡션 옆에 둔다
        r['chunk_view'] = chunk_view(r, docs[r['document_id']])
    p = write_html(picked, out / 'review.html')
    print(f'→ {p} ({len(picked)}장 — {dict(Counter(r["review_reason"] for r in picked))})')
    if (out / 'coverage.jsonl').exists():
        lost = [u for u in load_jsonl(out / 'coverage.jsonl') if u.get('loss_label') in ('유실', '판정 실패')]
        chunks = {d: await index_chunks(d) for d in {u['document_id'] for u in lost}}
        print(f'→ {write_coverage_html(lost, chunks, out / "coverage_review.html")} ({len(lost)}건)')


def coverage_table(rows: list[dict]) -> list[str]:
    by: defaultdict[tuple[str, str], Counter[str]] = defaultdict(Counter)
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
