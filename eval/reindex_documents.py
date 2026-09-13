"""활성 문서 재색인 CLI — 청킹·임베딩 방식이 바뀌었을 때 검색 색인(OpenSearch)을 새 방식으로 다시 만든다.

배경(2026-08-03): 인덱스 입력에 '파일명 > 헤딩' 병합(rag/index_text)과 DOCX 섹션 청킹
(rag/chunking)이 들어갔다. 둘 다 **재색인해야** 반영된다 — 기존 dense 벡터는 옛 방식으로
만들어졌고, DOCX는 청크 경계 자체가 달라졌다.

사용법:
    python -m eval.reindex_documents --dry-run              # 대상만 출력, 변경 없음
    python -m eval.reindex_documents --tenant <tenant_id>   # 한 테넌트만 (권장: 먼저 하나로 확인)
    python -m eval.reindex_documents --all                  # 전체

안전장치:
  - --dry-run / --tenant / --all 중 하나를 반드시 지정 (실수로 전체가 도는 것 방지)
  - 문서 단위 트랜잭션. 한 건 실패해도 나머지는 계속되고, 실패 목록을 끝에 요약한다.
  - 실패한 문서는 status='failed'로 남는다 → 검색에서 빠지므로, 요약을 보고 재실행할 것.
  - blob 원본이 없으면 건드리지 않고 건너뛴다 (청크를 지운 뒤 재생성 못 하는 사태 방지).

롤백: 청크는 blob에서 언제든 다시 만들 수 있는 파생 데이터다(색인에만 있다 — #139).
      옛 방식으로 되돌리려면 코드를 revert한 뒤 이 스크립트를 다시 돌리면 된다.

주의: 임베딩 TEI 서버(사내망)가 필요하다. 로컬에서는 ConnectTimeout으로 실패한다.
"""
import argparse
import asyncio
from pathlib import Path

from sqlalchemy import select

from config import settings
from database import AsyncSessionLocal
from rag import cache, os_client, outbox
from rag.models import Document


async def _targets(tenant_id: str | None) -> list[tuple[int, str, str, int]]:
    """재색인 대상 = 활성 문서 중 ready 또는 failed. (id, tenant, filename, 색인 청크수).

    failed도 포함한다 — 재색인이 중간에 실패하면 그 문서는 failed로 남아 검색에서 조용히
    빠진다. 원인을 고친 뒤 같은 명령으로 복구할 수 있어야 한다.
    """
    async with AsyncSessionLocal() as session:
        stmt = (
            select(Document)
            .where(Document.is_active.is_(True))
            .where(Document.status.in_(('ready', 'failed')))
            .order_by(Document.tenant_id, Document.id)
        )
        if tenant_id:
            stmt = stmt.where(Document.tenant_id == tenant_id)
        docs = (await session.execute(stmt)).scalars().all()

        out = []
        for doc in docs:
            n = (await os_client.client().count(index=settings.opensearch_index, body={
                "query": {"term": {"document_id": doc.id}}}))["count"]
            out.append((doc.id, doc.tenant_id, doc.filename, n))
        return out


async def reindex_one(document_id: int) -> None:
    """문서 하나를 재색인. pending 되돌림 + INDEX_DOCUMENT 등재 → 워커 로직 재사용.

    옛 청크를 먼저 지울 필요가 없다 — 인제스션 핸들러가 색인 전에 그 문서의 청크를 지우고
    결정적 id로 다시 넣는다(rag/os_index.py의 index_parsed_document). 처리 중에도 옛 청크가
    검색에 남아 있다가 한 번에 교체된다(빈 창 없음).
    캐시도 함께 무효화한다 — 청크 경계가 바뀌면 그 문서를 근거로 만든 답이 낡은 것이 된다.
    """
    async with AsyncSessionLocal() as session:
        doc = await session.get(Document, document_id)
        if doc is None:
            raise ValueError(f'문서 없음: {document_id}')
        if not Path(doc.blob_path).exists():
            raise FileNotFoundError(f'원본 없음: {doc.blob_path}')

        await cache.invalidate_source(session, doc.tenant_id, doc.id)
        doc.status = 'pending'          # 워커 로직의 진입 조건
        row = outbox.enqueue(session, doc.tenant_id, outbox.INDEX_DOCUMENT, document_id=doc.id)
        await session.commit()

    # 이 문서의 행만 처리 — 공유 DB의 다른 pending 업로드를 건드리지 않게 row_ids로 한정 (#139)
    await outbox.drain_once(row_ids=[row.id])   # 청킹 + 임베딩 + 색인 + ready 승격

    async with AsyncSessionLocal() as session:  # 승격 확인 — 실패면 status가 failed로 남는다
        doc = await session.get(Document, document_id)
        if doc.status != 'ready':
            raise RuntimeError(f'재색인 실패 (status={doc.status}): {doc.status_reason}')


async def main_async(args) -> None:
    targets = await _targets(args.tenant)
    if not targets:
        print('대상 문서가 없습니다.')
        return

    total_chunks = sum(n for *_, n in targets)
    print(f'대상 {len(targets)}개 문서 / 현재 청크 {total_chunks}개')
    for doc_id, tenant, filename, n in targets:
        print(f'  [{doc_id:>4}] {tenant:<24} {filename[:44]:<46} 청크 {n}')

    if args.dry_run:
        print('\n--dry-run: 변경하지 않았습니다.')
        return

    print()
    ok, failed = 0, []
    for i, (doc_id, tenant, filename, _) in enumerate(targets, start=1):
        try:
            await reindex_one(doc_id)
            ok += 1
            print(f'  ({i}/{len(targets)}) OK   {filename[:50]}')
        except Exception as e:
            failed.append((doc_id, filename, str(e)))
            print(f'  ({i}/{len(targets)}) FAIL {filename[:50]} — {e}')

    after = sum(n for *_, n in await _targets(args.tenant))
    print(f'\n완료: 성공 {ok} / 실패 {len(failed)}')
    print(f'청크 {total_chunks} → {after}')
    if failed:
        print('\n실패 문서 (status=failed — 검색에서 빠진 상태):')
        for doc_id, filename, err in failed:
            print(f'  [{doc_id}] {filename}: {err}')


def main() -> None:
    parser = argparse.ArgumentParser(description='활성 문서 재색인 (청킹·임베딩 방식 변경 반영)')
    parser.add_argument('--tenant', help='이 테넌트만 재색인')
    parser.add_argument('--all', action='store_true', help='전체 테넌트 재색인')
    parser.add_argument('--dry-run', action='store_true', help='대상만 출력하고 변경하지 않음')
    args = parser.parse_args()

    if not (args.tenant or args.all or args.dry_run):
        parser.error('--tenant / --all / --dry-run 중 하나를 지정하세요.')
    asyncio.run(main_async(args))


if __name__ == '__main__':
    main()
