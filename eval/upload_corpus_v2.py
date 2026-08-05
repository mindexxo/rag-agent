"""corpus_v2 업로드·인덱싱 셋업 (C-2 E2E) — 실 TEI 임베딩 사용 (사내망 필요).

테넌트별 디렉터리의 문서를 실제 업로드 API(ASGI 직call)로 넣고, 인덱싱 잡을 직접 실행한다.
*.rev2.* 는 버저닝 테스트용이라 제외. 재실행 시 같은 filename이면 새 version이 쌓인다 (dedupe 제거, 2026-08-05).

사용:
  python3 -m eval.upload_corpus_v2            # 업로드 + 인덱싱 + 카나리아 검증
  python3 -m eval.upload_corpus_v2 --verify   # 카나리아 검증만
"""
import asyncio
import sys
from pathlib import Path

import httpx

TENANTS = ['summers', 'homeplus', 'adererror', 'aromanica', 'goodpeople', 'harim']
ROOT = Path(__file__).resolve().parent.parent / 'sample_docs' / 'corpus_v2'

# 격리 카나리아: (tenant, 질의, top-5 청크에 있어야 할 수치, 있으면 안 되는 타사 고유 수치)
# 금지어는 해당 테넌트 코퍼스에 존재하지 않음을 확인한 값만 사용
# ('30일' 같은 값은 여러 테넌트가 각자 규정으로 갖고 있어 금지어로 쓰면 오탐 — 07-17 확인)
CANARIES = [
    ('summers',   '단순변심 반품 기간이 며칠인가요?', '14일', '45,000'),
    ('homeplus',  '단순변심 반품 기간이 며칠인가요?', '30일', '45,000'),
    ('adererror', '반품은 며칠 안에 해야 하나요?',   '7일',  '45,000'),
    ('goodpeople', '기부금 환불 처리는 며칠 걸리나요?', '5영업일', '왕복 배송비'),
    ('harim',     '단순변심으로 반품 되나요?',        '반품', '45,000'),
]


async def upload_all() -> None:
    from main import app
    from rag.documents import index_pending_document

    total = 0
    for tenant in TENANTS:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url='http://local',
            headers={'X-Tenant-Id': tenant}, timeout=120,
        ) as client:
            for f in sorted((ROOT / tenant).iterdir()):
                if '.rev2' in f.name or f.name.startswith('.'):
                    continue
                data = {}
                if f.suffix == '.xlsx':
                    data['description'] = f.stem.split('_', 2)[-1]   # 표 설명 = 파일명 주제부
                res = await client.post('/kms/documents',
                                        files={'file': (f.name, f.read_bytes())}, data=data)
                if res.status_code != 200:
                    print(f'  ✗ {tenant}/{f.name}: HTTP {res.status_code} {res.text[:100]}')
                    continue
                body = res.json()
                if body['status'] == 'pending':
                    await index_pending_document(body['document_id'])
                    from database import AsyncSessionLocal
                    from rag.models import Document
                    async with AsyncSessionLocal() as s:
                        doc = await s.get(Document, body['document_id'])
                        mark = '✓' if doc.status == 'ready' else f'✗ {doc.status}: {doc.status_reason}'
                else:
                    mark = f"= {body['status']}"
                print(f'  {mark} {tenant}/{f.name}')
                total += 1
    print(f'\n업로드 시도 {total}건 완료')


async def verify() -> bool:
    from database import AsyncSessionLocal
    from rag.retriever import retrieve
    from sqlalchemy import func, select

    from rag.models import Chunk, Document

    ok = True
    async with AsyncSessionLocal() as session:
        for tenant in TENANTS:
            n_docs = (await session.execute(
                select(func.count()).select_from(Document)
                .where(Document.tenant_id == tenant).where(Document.status == 'ready')
            )).scalar()
            n_chunks = (await session.execute(
                select(func.count()).select_from(Chunk).where(Chunk.tenant_id == tenant)
            )).scalar()
            print(f'{tenant}: ready {n_docs}문서 / {n_chunks}청크')

        print('\n── 격리 카나리아 (실 임베딩 검색) ──')
        for tenant, query, expect, forbid in CANARIES:
            result = await retrieve(session, tenant, query)
            texts = ' '.join(c.text for c in result.chunks[:5])
            has, leaked = expect in texts, forbid in texts
            status = '✓' if (has and not leaked and not result.no_evidence) else '✗'
            if status == '✗':
                ok = False
            print(f'{status} {tenant} "{query}" → 기대 {expect}: {has} / '
                  f'금지 {forbid}: {leaked} / no_evidence={result.no_evidence} '
                  f'/ top1={result.chunks[0].filename if result.chunks else "-"}')
    return ok


async def main() -> None:
    if '--verify' not in sys.argv:
        await upload_all()
    ok = await verify()
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    asyncio.run(main())
