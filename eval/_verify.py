import asyncio, json, re
from sqlalchemy import select
from database import AsyncSessionLocal
from rag.models import Chunk, Document

def norm(s): return re.sub(r'\s+', '', s)

async def main():
    async with AsyncSessionLocal() as s:
        rows = (await s.execute(
            select(Chunk, Document.filename).join(Document, Chunk.document_id == Document.id)
            .where(Chunk.tenant_id == 'demo', Document.is_active.is_(True))
        )).all()
    by_file = {}
    for c, f in rows:
        by_file.setdefault(f, []).append(c)
    all_text = norm(' '.join(c.text for c, _ in rows))

    items = [json.loads(l) for l in open('eval/gold_set_v1.jsonl')]
    print(f'총 {len(items)}문항')

    label_issues = []
    for o in items:
        iid, typ = o['id'], o['type']
        for ch in o.get('expected_chunks', []):
            fn, hp, sn = ch['filename'], ch.get('heading_path', []), ch['snippet']
            cands = by_file.get(fn, [])
            hit = [c for c in cands if c.heading_path == hp and norm(sn) in norm(c.text)]
            if not hit:
                if any(norm(sn) in norm(c.text) for c in cands):
                    label_issues.append((iid, typ, 'heading 불일치', f'{fn} {hp}'))
                else:
                    label_issues.append((iid, typ, 'snippet 부재', f'{fn} "{sn}"'))
        cf = {c['filename'] for c in o.get('expected_chunks', [])}
        df = set(o.get('expected_docs', []))
        if cf - df:
            label_issues.append((iid, typ, 'docs 누락', str(cf - df)))

    print(f'\n[근거 라벨] 이슈 {len(label_issues)}건')
    for x in label_issues:
        print('  ', x)

    # trap/no_evidence: 정답이 정말 코퍼스에 없는지(핵심 키워드 부재 확인 보조)
    print('\n[trap/no_evidence] has_evidence=False 검토 대상 목록')
    for o in items:
        if o['type'] in ('trap', 'no_evidence'):
            print(f"  {o['id']} {o['type']}: {o['query']}")

    ids = [o['id'] for o in items]
    print(f'\nid 중복: {len(ids) - len(set(ids))}건')

asyncio.run(main())
