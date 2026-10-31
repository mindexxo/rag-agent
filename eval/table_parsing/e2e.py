"""#253 E2E — 합성 표 46 × 4경로를 **실제 업로드 → 색인 → 청크 확인 → 질문·답변**까지 돌리고 사람이 볼 리뷰 HTML을 쓴다.

docling 결과가 멀쩡한데 하네스 수치만 보고 오보고한 일(#253 — 중첩·나란히 '불가')을 막기 위한 것이다: 색인 청크와
실제 답변을 **눈으로** 본다. 경로마다 테넌트를 따로 둔다(`tbl_e2e_<path>`) — 같은 값의 표가 4경로에 있으면 어느 경로가
답했는지 알 수 없다. 워커 대신 outbox 행을 직접 처리한다(tests.conftest.ingest와 같은 길).

용법(로컬 앱 8000 + 개발계 DB·OpenSearch·TEI·vLLM 필요):
    python -m eval.table_parsing.e2e --phase all                 # 업로드·색인 → 질문 → 리뷰 → (테넌트 문서 삭제는 --cleanup)
    python -m eval.table_parsing.e2e --phase upload --paths img pdf --only S01_basic
    python -m eval.table_parsing.e2e --phase ask                 # manifest.json 기준으로 질문만
    python -m eval.table_parsing.e2e --phase report              # answers.jsonl 기준으로 리뷰만
    python -m eval.table_parsing.e2e --phase cleanup             # 테넌트 문서 삭제 + DROP 처리
산출물(<out>/): manifest.json(문서 id·청크), answers.jsonl(질문·답·인용·판정), review.html(정의 이미지·기대 행·실제 청크·
Q/A 나란히), summary.md(축 × 경로 정답률·인용 일치율).
"""

import argparse
import asyncio
import json
import time
from collections import defaultdict
from pathlib import Path

import httpx

from eval.table_parsing.build import build_all
from eval.table_parsing.expected import expected_rows, expected_tables, norm
from eval.table_parsing.llm_judge import final_ok, rejudge
from eval.table_parsing.questions import ALL_QUESTIONS, judge
from eval.table_parsing.render_image import cell_text
from eval.table_parsing.specs import BY_ID, PATH_LABEL, axis_name, axis_no

PATHS = ('img', 'pdf', 'docx', 'xlsx')
MIME = {
    'pdf': 'application/pdf',
    'docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    'xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
}
DOMAIN_HINT = '사내 규정·복리후생·출장·장비·급여 기준표를 안내하는 상담 지식'
QUERY_GAP = 1.6  # 테넌트 리미터(동시 1)가 슬롯을 늦게 돌려줘 연속 전송이 429를 받는다(#209) — 1.5초면 통과


def tenant_of(path: str) -> str:
    return f'tbl_e2e_{path}'


# ── 업로드·색인 ─────────────────────────────────────────────────────────────────


async def _ingest(document_id: int) -> dict:
    from database import AsyncSessionLocal
    from rag import outbox

    async with AsyncSessionLocal() as s:
        ids = await outbox.pending_row_ids(s, document_id=document_id)
    return await outbox.drain_once(row_ids=ids)


async def _chunks(document_id: int) -> list[dict]:
    from config import settings
    from rag.search import os_client

    c = os_client.client()
    resp = await c.search(
        index=settings.opensearch_index,
        body={
            'size': 1000,
            '_source': ['chunk_id', 'text', 'heading_path', 'meta'],
            'query': {'term': {'document_id': document_id}},
            'sort': [{'chunk_id': 'asc'}],
        },
    )
    return [h['_source'] for h in resp['hits']['hits']]


async def phase_upload(base: str, out: Path, only, paths, context: bool = False) -> list[dict]:
    made = build_all(out / 'docs', only, set(paths), context=context)
    manifest = []
    async with httpx.AsyncClient(base_url=base, timeout=120) as client:
        for spec_id, path, file in made:
            spec = BY_ID[spec_id]
            fname = f'{spec_id}.{path}{file.suffix}'
            files = {'file': (fname, file.read_bytes(), MIME[file.suffix.lstrip(".")])}
            if path == 'xlsx':
                files['document-data'] = (
                    'blob',
                    json.dumps({'description': f'{spec["axis"]} 기준표'}),
                    'application/json',
                )
            res = await client.post('/kms/documents', files=files, headers={'X-Tenant-Id': tenant_of(path)})
            if res.status_code != 200:
                manifest.append(
                    dict(id=spec_id, path=path, filename=fname, error=f'upload {res.status_code}: {res.text[:200]}')
                )
                print(f'{spec_id:16} {path:5} 업로드 실패 {res.status_code}', flush=True)
                continue
            doc_id = res.json()['document_id']
            t = time.time()
            drained = await _ingest(doc_id)
            from database import AsyncSessionLocal
            from rag.models import Document

            async with AsyncSessionLocal() as s:
                d = await s.get(Document, doc_id)
                status, reason = d.status, d.status_reason
            chunks = await _chunks(doc_id) if status == 'ready' else []
            manifest.append(
                dict(
                    id=spec_id,
                    path=path,
                    filename=fname,
                    document_id=doc_id,
                    status=status,
                    status_reason=reason,
                    chunks=chunks,
                    index_seconds=round(time.time() - t, 1),
                )
            )
            print(
                f'{spec_id:16} {path:5} doc={doc_id} {status} 청크 {len(chunks)} ({time.time() - t:.1f}s) {drained}',
                flush=True,
            )
    (out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    return manifest


# ── 질문 ────────────────────────────────────────────────────────────────────────


def _sse(text: str):
    events = []
    for block in text.strip().split('\n\n'):
        lines = block.splitlines()
        if len(lines) >= 2 and lines[0].startswith('event: '):
            try:
                events.append((lines[0][7:], json.loads(lines[1].removeprefix('data: '))))
            except json.JSONDecodeError:
                pass
    return events


async def phase_ask(base: str, out: Path, manifest: list[dict], only: set[str] | None = None) -> list[dict]:
    """only를 주면 그 정의만 다시 묻고, 기존 answers.jsonl에서 같은 정의의 행을 바꿔 끼운다(정답 키 수정 뒤 재판정용)."""
    answers = []
    if only and (out / 'answers.jsonl').exists():
        answers = [json.loads(l) for l in open(out / 'answers.jsonl') if json.loads(l)['id'] not in only]
        (out / 'answers.jsonl').write_text(''.join(json.dumps(a, ensure_ascii=False) + '\n' for a in answers))
    async with httpx.AsyncClient(base_url=base, timeout=300) as client:
        for m in manifest:
            if m.get('status') != 'ready' or (only and m['id'] not in only):
                continue
            for q in ALL_QUESTIONS[m['id']][m['path']]:
                t = time.time()
                res = await client.post(
                    '/kms/query',
                    json={'query': q.text, 'domain_hint': DOMAIN_HINT},
                    headers={'X-Tenant-Id': tenant_of(m['path'])},
                )
                events = _sse(res.text) if res.status_code == 200 else []
                answer = ''.join(d.get('text', '') for e, d in events if e == 'delta')
                done = next((d for e, d in events if e == 'done'), {})
                cited = [c.get('filename') for c in done.get('citations', [])]
                row = dict(
                    id=m['id'],
                    path=m['path'],
                    document_id=m['document_id'],
                    question=q.text,
                    expected=q.expected,
                    manual=q.manual,
                    note=q.note,
                    answer=answer,
                    finish=done.get('finish_reason'),
                    cited=cited,
                    cited_ok=m['filename'] in cited,
                    value_ok=judge(answer, q.expected),
                    http=res.status_code,
                    seconds=round(time.time() - t, 1),
                )
                answers.append(row)
                with open(out / 'answers.jsonl', 'a') as f:
                    f.write(json.dumps(row, ensure_ascii=False) + '\n')
                mark = '✅' if row['value_ok'] and row['cited_ok'] else ('🔎' if q.manual else '❌')
                print(
                    f"{m['id']:16} {m['path']:5} {mark} {q.text[:28]:28} → {answer[:50]!r} 인용={row['cited_ok']} ({row['seconds']}s)",
                    flush=True,
                )
                await asyncio.sleep(QUERY_GAP)
    return answers


# ── 리뷰 ────────────────────────────────────────────────────────────────────────


def _loose(t: str) -> str:
    return norm(t).replace(' ', '')


def chunk_loss(spec, path: str, chunks: list[dict]) -> list[str]:
    """정의의 칸 글자 중 실제 색인 청크에 없는 것(띄어쓰기 무시) — '청킹이 누락 없이 됐는가'의 근거."""
    text = _loose(' '.join(c['text'] for c in chunks))
    cells = [cell_text(c) for t in expected_tables(spec, path) for row in t for c in row if c is not None]
    return [c for c in dict.fromkeys(cells) if _loose(c) and _loose(c) not in text]


def diagnose(row: dict, chunks: list[dict]) -> str:
    """최종 실패한 답의 원인 추정 — 답변 모양·인용·청크에 정답이 있는지로 가른다(LLM 판정 이유와 나란히 보여 준다)."""
    if row['final_ok']:
        return ''
    in_chunk = _loose(row['expected']) in _loose(' '.join(c['text'] for c in chunks))
    refused = ('확인할 수 없' in row['answer']) or row.get('llm_label') == 'not_answered'
    if refused and not row['cited']:
        return (
            '검색 단계에서 후보 0건(근거 없음 게이트) — 정답이 청크에 있는데 질문 어휘가 표 어휘와 달라 못 찾음'
            if in_chunk
            else '검색 후보 0건 + 정답이 청크에도 없음 — 파싱 단계 손실이 원인'
        )
    if row.get('llm_label') == 'ungrounded':
        return '청크에 근거가 없는데 답함 — 파싱 손실(OCR 기호·병합) 위에 모델이 추정'
    if not in_chunk:
        return '청크 자체에 정답이 없거나 틀림 — 파싱 손실(OCR 오독·캡션 오독·렌더 누락)'
    if row['cited'] and not row['cited_ok']:
        return '다른 문서를 근거로 답함 — 검색이 엉뚱한 문서를 올림'
    return '청크엔 정답이 있는데 답이 다름 — 생성 모델 오류'


def _esc(s: str) -> str:
    return str(s).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def phase_report(out: Path, manifest: list[dict], answers: list[dict]) -> None:
    by_doc = defaultdict(list)
    for a in answers:
        a['value_ok'] = judge(a['answer'], a['expected'])  # 판정 규칙이 바뀌면 리뷰만 다시 만들어도 반영되게
        a['final_ok'] = final_ok(a)
        by_doc[(a['id'], a['path'])].append(a)
    agg = defaultdict(lambda: dict(q=0, v=0, f=0, c=0, man=0, idx=0, ready=0, ung=0))
    cards = []
    for m in sorted(
        manifest, key=lambda x: (int(axis_no(BY_ID[x['id']])), x['id'], PATHS.index(x['path']))
    ):  # 축 번호 순(1→23), 경로 순
        spec = BY_ID[m['id']]
        a = agg[(spec['axis'], m['path'])]
        a['idx'] += 1
        a['ready'] += m.get('status') == 'ready'
        qa = by_doc.get((m['id'], m['path']), [])
        for x in qa:
            a['q'] += 1
            a['v'] += x['value_ok']
            a['f'] += x['final_ok']
            a['c'] += x['cited_ok']
            a['man'] += x['manual']
            a['ung'] += x.get('llm_label') == 'ungrounded'
        lost = chunk_loss(spec, m['path'], m.get('chunks') or []) if m.get('status') == 'ready' else []
        loss_html = (
            '<div class="loss ok">청킹 누락 없음 — 정의의 모든 칸 글자가 청크에 있음(띄어쓰기 무시)</div>'
            if not lost
            else f'<div class="loss bad">청킹 누락 {len(lost)}칸: {_esc(", ".join(lost[:8]))}{" …" if len(lost) > 8 else ""}</div>'
        )
        exp = expected_rows(spec, expected_tables(spec, m['path'])[0], m['path'])
        exp_md = '\n'.join('| ' + ' | '.join(r) + ' |' for r in exp)
        chunks_html = ''.join(
            f'<div class="chunk"><div class="hp">#{i} {_esc(" › ".join(c.get("heading_path") or []))}</div>'
            f'<pre>{_esc(c["text"])}</pre></div>'
            for i, c in enumerate(m.get('chunks') or [])
        )

        def _llm_cell(x):
            if x.get('llm_label') is None:
                return '—' if x['value_ok'] else ('오류' if x.get('llm_error') else '—')
            name = {'correct': '맞음', 'wrong': '틀림', 'ungrounded': '근거 없음', 'not_answered': '미답'}[
                x['llm_label']
            ]
            return f"<b>{name}</b><br><span class=\"why\">{_esc(x.get('llm_reason') or '')}</span>"

        qa_html = ''.join(
            f"<tr class=\"{'ok' if x['final_ok'] and x['cited_ok'] else ('ung' if x.get('llm_label') == 'ungrounded' else 'bad')}\">"
            f"<td>{_esc(x['question'])}</td><td>{_esc(x['expected'])}</td><td>{_esc(x['answer'])}</td>"
            f"<td class=\"{'y' if x['value_ok'] else 'n'}\">{'성공' if x['value_ok'] else '실패'}</td>"
            f"<td>{_llm_cell(x)}</td>"
            f"<td class=\"{'y' if x['final_ok'] else 'n'}\">{'성공' if x['final_ok'] else '실패'}</td>"
            f"<td class=\"{'y' if x['cited_ok'] else 'n'}\">{'성공' if x['cited_ok'] else '실패'}</td>"
            f"<td class=\"why\">{_esc(diagnose(x, m.get('chunks') or []))}</td></tr>"
            for x in qa
        )
        png = out / 'docs' / 'png' / f"{m['id']}.png"
        img = f'<img src="docs/png/{png.name}">' if png.exists() else ''
        failed = (
            ''
            if m.get('status') == 'ready'
            else f" <span class=\"fail\">[실패: {_esc(m.get('status_reason') or m.get('error') or m.get('status'))}]</span>"
        )
        cards.append(
            f'<div class="c" id="{m["id"]}-{m["path"]}"><h3>[{axis_no(spec)}] [{_esc(axis_name(spec))}] [{PATH_LABEL[m["path"]]}]{failed}</h3>'
            f'<div class="row"><div class="col">{img}<h4>기대 행(설계 규칙)</h4><pre>{_esc(exp_md)}</pre></div>'
            f'<div class="col"><h4>실제 색인 청크 {len(m.get("chunks") or [])}개 (OpenSearch 조회)</h4>{loss_html}{chunks_html}</div></div>'
            f'<table><tr><th>질문</th><th>정답</th><th>답변(14B)</th><th>기계 판정</th><th>LLM 재판정</th><th>생성(최종)</th><th>인용</th><th>실패 원인(추정)</th></tr>{qa_html}</table></div>'
        )
    style = (
        '<style>body{font-family:-apple-system,sans-serif;margin:24px;background:#fafafa}.c{background:#fff;border:1px solid #ddd;'
        'border-radius:8px;padding:12px;margin:16px 0}.row{display:flex;gap:16px}.col{flex:1;min-width:0}img{max-width:100%;border:1px dashed #ccc}'
        'pre{white-space:pre-wrap;font-size:12px;background:#f6f6f6;padding:6px;border-radius:4px}.chunk{margin-bottom:6px}.hp{color:#888;font-size:11px}'
        'table{border-collapse:collapse;font-size:13px;width:100%}td,th{border:1px solid #ddd;padding:4px 8px;vertical-align:top}'
        'tr.ok td{background:#f0fff0}tr.bad td{background:#fff0f0}tr.ung td{background:#fffbe6}td.y{color:#1a7f37;font-weight:600}td.n{color:#c0392b;font-weight:600}.why{color:#666;font-size:11px}'
        '.fail{color:#c0392b}.loss{font-size:12px;padding:4px 8px;border-radius:4px;margin-bottom:6px}.loss.ok{background:#eaf7ea;color:#1a7f37}.loss.bad{background:#fdecec;color:#c0392b}'
        'h3{font-size:15px;margin:0 0 8px}h4{margin:8px 0 4px;font-size:13px}</style>'
    )
    (out / 'review.html').write_text(
        '<meta charset="utf-8">' + style + '<h1>#253 표 E2E 리뷰 — 정의 · 색인 청크 · 질문/답변</h1>'
        '<p>기계 판정 = 정답 값 문자열이 답변에 있나 · LLM 재판정 = 기계 실패분만 Claude가 질문·정답·답변·청크를 보고 맞음/틀림/근거 없음/미답 판정 · 생성(최종) = 둘 중 하나라도 맞음 · 인용 = 근거로 댄 문서가 이 문서인가. 초록 = 최종 성공+인용 성공, 노랑 = 근거 없는 정답(청크에 값이 없는데 맞게 답함), 빨강 = 실패. 청크는 OpenSearch 조회 그대로, 답변은 /kms/query 실제 응답.</p>'
        + ''.join(cards)
    )
    axes = sorted({k[0] for k in agg})
    lines = ['| 축 | ' + ' | '.join(PATH_LABEL[p] for p in PATHS) + ' |', '|---|' + '---|' * len(PATHS)]
    for ax in axes:
        cells = []
        for p in PATHS:
            a = agg.get((ax, p))
            if not a:
                cells.append('—')
                continue
            cells.append(
                f"색인 {a['ready']}/{a['idx']} · 생성 {a['f']}/{a['q']}(기계 {a['v']}) · 인용 {a['c']}/{a['q']}"
                + (f" · 근거없음 {a['ung']}" if a['ung'] else '')
            )
        lines.append(f'| {ax} | ' + ' | '.join(cells) + ' |')
    tot = {k: sum(a[k] for a in agg.values()) for k in ('q', 'v', 'f', 'c', 'ung')}
    judges = sorted({a['judge'] for a in answers if a.get('judge')})
    (out / 'summary.md').write_text(
        '# #253 E2E 요약 — 실제 업로드·색인·질문\n\n'
        + '\n'.join(lines)
        + f"\n\n합계: 질문 {tot['q']} · 생성 성공(최종) {tot['f']} (기계 판정만 {tot['v']}) · 인용 성공 {tot['c']}"
        f" · 근거 없는 정답 {tot['ung']}\n재판정자: {', '.join(judges) or '없음'} (기계 판정 실패분만)\n"
    )


# ── 정리 ────────────────────────────────────────────────────────────────────────


async def phase_cleanup(base: str, manifest: list[dict]) -> None:
    from database import AsyncSessionLocal
    from rag import outbox

    async with httpx.AsyncClient(base_url=base, timeout=120) as client:
        for path in PATHS:
            ids = [m['document_id'] for m in manifest if m['path'] == path and m.get('document_id')]
            for i in range(0, len(ids), 100):
                q = '&'.join(f'ids={x}' for x in ids[i : i + 100])
                res = await client.delete(f'/kms/documents?{q}', headers={'X-Tenant-Id': tenant_of(path)})
                print(path, 'delete', res.status_code, len(ids[i : i + 100]))
    async with AsyncSessionLocal() as s:
        from sqlalchemy import select

        from rag.models import SearchIndexOutbox

        rows = (
            (
                await s.execute(
                    select(SearchIndexOutbox.id)
                    .where(SearchIndexOutbox.status == 'pending')
                    .where(SearchIndexOutbox.tenant_id.in_([tenant_of(p) for p in PATHS]))
                )
            )
            .scalars()
            .all()
        )
    if rows:
        print('drain', await outbox.drain_once(row_ids=rows))


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', default='all', choices=['all', 'upload', 'ask', 'report', 'cleanup'])
    ap.add_argument('--out', default='eval/results/table_parsing/e2e1')
    ap.add_argument('--base-url', default='http://localhost:8000')
    ap.add_argument('--only', nargs='*')
    ap.add_argument('--paths', nargs='*', default=list(PATHS))
    ap.add_argument('--no-llm', action='store_true', help='기계 판정 실패분의 LLM 재판정을 건너뛴다')
    ap.add_argument('--context', action='store_true', help='실제 문서처럼 앞뒤 문맥을 둔 문서로(#247, context.py)')
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((out / 'manifest.json').read_text()) if (out / 'manifest.json').exists() else []
    if args.phase in ('all', 'upload'):
        made = await phase_upload(args.base_url, out, set(args.only) if args.only else None, args.paths, args.context)
        if args.phase == 'upload' and args.only:  # 일부만 다시 올릴 때 기존 manifest에 합친다(같은 id·path는 교체)
            keep = [m for m in manifest if (m['id'], m['path']) not in {(x['id'], x['path']) for x in made}]
            manifest = keep + made
            (out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
        else:
            manifest = made
    answers = (
        [json.loads(l) for l in open(out / 'answers.jsonl')]
        if (out / 'answers.jsonl').exists() and args.phase != 'all'
        else []
    )
    if args.phase in ('all', 'ask'):
        if args.phase == 'all' and (out / 'answers.jsonl').exists():
            (out / 'answers.jsonl').unlink()
        answers = await phase_ask(
            args.base_url, out, manifest, set(args.only) if (args.phase == 'ask' and args.only) else None
        )
    if args.phase in ('all', 'ask', 'report'):
        if not args.no_llm:
            for a in answers:
                a['value_ok'] = judge(a['answer'], a['expected'])
            chunks_by = {(m['id'], m['path']): m.get('chunks') or [] for m in manifest}
            targets = [a for a in answers if a.get('llm_label') is None]  # 이미 판정된 행은 다시 묻지 않는다
            done = await rejudge(targets, chunks_by)
            print(f'LLM 재판정 {len(done)}건', flush=True)
            with open(out / 'answers.jsonl', 'w') as f:
                for a in answers:
                    f.write(json.dumps(a, ensure_ascii=False) + '\n')
        phase_report(out, manifest, answers)
        print('리뷰:', out / 'review.html', '요약:', out / 'summary.md')
    if args.phase == 'cleanup':
        await phase_cleanup(args.base_url, manifest)


if __name__ == '__main__':
    asyncio.run(main())
