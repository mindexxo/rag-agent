"""표 파싱 E2E — 합성 표 46 × 4경로를 **실제 업로드 → 색인 → 청크 확인 → 질문·답변**까지 돌리고 사람이 볼 리뷰 HTML을 쓴다.

하네스 수치만 보면 docling 결과가 멀쩡한데도 '불가'로 오판할 수 있어서다: 색인 청크와
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
from pathlib import Path

import httpx

from eval.table_parsing.build import build_all
from eval.table_parsing.e2e_review import PATHS, chunk_loss, diagnose, phase_report
from eval.table_parsing.llm_judge import rejudge
from eval.table_parsing.questions import ALL_QUESTIONS, judge
from eval.table_parsing.specs import BY_ID

__all__ = ['chunk_loss', 'diagnose', 'phase_report']  # 테스트가 이 모듈에서 가져간다
MIME = {
    'pdf': 'application/pdf',
    'docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    'xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
}
DOMAIN_HINT = '사내 규정·복리후생·출장·장비·급여 기준표를 안내하는 상담 지식'
QUERY_GAP = 1.6  # 테넌트 리미터(동시 1)가 슬롯을 늦게 돌려줘 연속 전송이 429를 받는다 — 1.5초면 통과


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


async def phase_upload(base: str, out: Path, only: set[str] | None, paths: list, context: bool = False) -> list[dict]:
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
        answers = [json.loads(line) for line in open(out / 'answers.jsonl') if json.loads(line)['id'] not in only]
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


async def main() -> None:
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
        [json.loads(line) for line in open(out / 'answers.jsonl')]
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
