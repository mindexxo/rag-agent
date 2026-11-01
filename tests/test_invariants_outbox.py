"""불변식 대장(INVARIANTS.md)의 부분 칸 보강 — outbox·reconcile 축(등재 트랜잭션·payload 모양·결정적 실패·읽기 순서).

공유 대기열이라 drain은 conftest `ingest`·`sync_faq`처럼 문서·FAQ 단위로 한정하고, reconcile은 반드시 tenant_id를 넘긴다.
"""

import ast
from dataclasses import dataclass
from pathlib import Path

import pytest
from sqlalchemy import func, select

from database import AsyncSessionLocal
from rag import os_reconcile, outbox
from rag.chunking.errors import PictureLimitError, TableTooLargeError
from rag.chunking.xlsx import XlsxDescriptionRequired, XlsxTableTooLarge
from rag.models import SearchIndexOutbox
from tests.conftest import faq_doc, indexed_chunk_texts, ingest, sync_faq
from tests.helpers_documents import MD, get_doc, upload
from tests.helpers_index_sync import latest_index_row, outbox_rows

ROOT = Path(__file__).resolve().parent.parent

# op → payload 키(정의점: rag/outbox.py 연산 어휘 주석). 값은 id(int) 또는 id 목록뿐이다.
PAYLOAD_KEY = {
    outbox.INDEX_DOCUMENT: 'document_id',
    outbox.DROP_DOCUMENTS: 'document_ids',
    outbox.META_DOCUMENTS: 'document_ids',
    outbox.INDEX_FAQ: 'faq_id',
    outbox.DROP_FAQS: 'faq_ids',
    outbox.META_FAQS: 'faq_ids',
}
OP_CONST = {name: v for name, v in vars(outbox).items() if isinstance(v, str) and v in PAYLOAD_KEY}


# ── INV-10 enqueue는 commit하지 않는다 ───────────────────────────────────────────────────────────


class _NoCommitSession:
    """add만 받고 commit·flush는 터뜨리는 세션 대역."""

    def __init__(self) -> None:
        self.added: list = []

    def add(self, obj: object) -> None:
        self.added.append(obj)

    # 동기 함수로 둔다 — await 없이 부르기만 해도(코루틴 생성) 바로 터진다
    def commit(self) -> None:
        raise AssertionError('enqueue가 commit했다')

    def flush(self) -> None:
        raise AssertionError('enqueue가 flush했다')


def test_INV10_enqueue는_add만_하고_commit도_flush도_하지_않는다():
    """INV-10 outbox 등재는 호출부 트랜잭션에 얹는다 — enqueue는 session.add 한 줄이다."""
    s = _NoCommitSession()
    row = outbox.enqueue(s, 't-a', outbox.INDEX_FAQ, faq_id=7)
    assert s.added == [row]
    assert (row.tenant_id, row.op, row.payload, row.status) == ('t-a', outbox.INDEX_FAQ, {'faq_id': 7}, outbox.PENDING)


@pytest.mark.asyncio
async def test_INV10_호출부가_롤백하면_등재도_사라진다(tenant_id):
    async with AsyncSessionLocal() as s:
        outbox.enqueue(s, tenant_id, outbox.INDEX_FAQ, faq_id=10**9)
        await s.rollback()
    async with AsyncSessionLocal() as s:
        n = (
            await s.execute(
                select(func.count()).select_from(SearchIndexOutbox).where(SearchIndexOutbox.tenant_id == tenant_id)
            )
        ).scalar()
    assert n == 0


# ── INV-13 payload에는 id만 ──────────────────────────────────────────────────────────────────────


def _enqueue_calls() -> list[tuple[str, ast.Call]]:
    out = []
    for p in sorted((ROOT / 'rag').rglob('*.py')) + sorted((ROOT / 'routers').rglob('*.py')):
        for c in ast.walk(ast.parse(p.read_text(encoding='utf-8'))):
            f = getattr(c, 'func', None)
            name = f.attr if isinstance(f, ast.Attribute) else getattr(f, 'id', None)
            if isinstance(c, ast.Call) and name == 'enqueue':
                out.append((f'{p.relative_to(ROOT)}:{c.lineno}', c))
    return out


def test_INV13_모든_enqueue_호출은_op에_맞는_id_키_하나만_싣는다():
    """INV-13 outbox payload에는 id만 싣는다 — 등재 지점마다 키워드 인자가 op의 id 키 하나뿐이어야 한다."""
    calls = _enqueue_calls()
    assert len(calls) >= 10, f'enqueue 호출을 {len(calls)}개만 찾았다'
    bad = []
    for where, c in calls:
        op = c.args[2] if len(c.args) > 2 else None
        op_name = op.attr if isinstance(op, ast.Attribute) else getattr(op, 'id', None)
        keys = [k.arg for k in c.keywords]
        if op_name not in OP_CONST or keys != [PAYLOAD_KEY[OP_CONST[op_name]]]:
            bad.append(f'{where} op={op_name} keys={keys}')
    assert bad == [], bad


def _is_id_payload(op: str, payload: dict) -> bool:
    key = PAYLOAD_KEY[op]
    if list(payload) != [key]:
        return False
    value = payload[key]
    if key.endswith('_ids'):
        return isinstance(value, list) and all(type(v) is int for v in value)
    return type(value) is int


@pytest.mark.asyncio
async def test_INV13_실제로_쌓인_INDEX_META_DROP_행의_payload는_id뿐이다(client, tenant_id, blob_tmp, fake_embed):
    """문서·FAQ의 등록·토글·삭제가 남긴 행 전부 — 키는 op의 id 키 하나, 값은 int 또는 int 목록."""
    doc_id = (await upload(client, '환불정책.md', MD))['document_id']
    assert (await client.patch(f'/kms/documents/{doc_id}', json={'is_searchable': False})).status_code == 200
    assert (await client.delete(f'/kms/documents/{doc_id}')).status_code in (200, 204)
    faq_id = (await client.post('/kms/faqs', json={'question': '환불?', 'variants': [], 'answer': '7일'})).json()['id']
    assert (await client.patch(f'/kms/faqs/{faq_id}', json={'is_active': False})).status_code == 200
    assert (await client.delete(f'/kms/faqs/{faq_id}')).status_code in (200, 204)

    async with AsyncSessionLocal() as s:
        rows = (
            (await s.execute(select(SearchIndexOutbox).where(SearchIndexOutbox.tenant_id == tenant_id))).scalars().all()
        )
    ops = {r.op for r in rows}
    assert ops == set(PAYLOAD_KEY), f'여섯 op가 모두 쌓이지 않았다: {ops}'
    bad = [(r.op, r.payload) for r in rows if not _is_id_payload(r.op, r.payload)]
    assert bad == [], bad


# ── INV-14 결정적 실패는 재시도하지 않는다 ─────────────────────────────────────────────────────────

DETERMINISTIC = [
    XlsxTableTooLarge('시트1', 140, 300, 12),
    XlsxDescriptionRequired(),
    TableTooLargeError(3, ['항목', '값'], 130, 200, 2),
    PictureLimitError('그림 120장'),
    ValueError('추출된 텍스트가 없습니다'),
]


@pytest.mark.parametrize('exc', DETERMINISTIC, ids=lambda e: type(e).__name__)
def test_INV14_청킹_결정적_예외와_xlsx_거절은_일시로_분류되지_않는다(exc):
    assert outbox.is_transient(exc) is False


@pytest.mark.asyncio
@pytest.mark.parametrize('exc', DETERMINISTIC[:4], ids=lambda e: type(e).__name__)
async def test_INV14_결정적_색인_실패는_한번으로_확정하고_다시_돌지_않는다(
    client, tenant_id, blob_tmp, fake_embed, monkeypatch, exc
):
    """INV-14 결정적 실패는 재시도하지 않는다 — 1회로 failed, 다음 시도 시각 없음, 다음 회차에 변환을 다시 부르지 않는다."""
    import rag.ingestion as ing

    calls = []

    def chunk_fails(*a, **kw):
        calls.append(a)
        raise exc

    monkeypatch.setattr(ing, 'chunk_file', chunk_fails)
    doc_id = (await upload(client, '환불정책.md', MD))['document_id']
    assert await ingest(doc_id) == {'done': 0, 'failed': 1}
    row = await latest_index_row(doc_id)
    assert (row.status, row.attempts, row.next_attempt_at) == ('failed', 1, None)
    assert (await get_doc(doc_id)).status == 'failed'

    assert await ingest(doc_id) == {'done': 1, 'failed': 0}  # 남은 건 잔여 청크 DROP 행뿐
    assert len(calls) == 1, '결정적 실패 문서를 다시 변환했다'
    assert [r[:3] for r in await outbox_rows(doc_id)] == [
        (outbox.INDEX_DOCUMENT, 'failed', 1),
        (outbox.DROP_DOCUMENTS, 'done', 0),
    ]


# ── INV-16 reconcile 읽기 순서 — FAQ·dry-run ─────────────────────────────────────────────────────


@dataclass
class _Once:
    """첫 호출에서만 작업을 한 번 한다 — 몇 번째 스냅샷 직후에 불렸는지와 무관하게."""

    action: object
    fired: bool = False

    async def __call__(self) -> None:
        if not self.fired:
            self.fired = True
            await self.action()


class _PgHook:
    """세션 대역 — PG 조회가 끝난 직후, 그 조회가 다룬 쪽(문서·FAQ)의 훅을 부른다."""

    def __init__(self, session, hooks: dict[str, _Once]) -> None:
        self._s, self._hooks = session, hooks

    async def execute(self, stmt, *a, **kw):
        res = await self._s.execute(stmt, *a, **kw)
        table = 'faqs' if 'FROM faqs' in str(stmt) else 'documents'
        if table in self._hooks:
            await self._hooks[table]()
        return res

    def __getattr__(self, name):
        return getattr(self._s, name)


def _hook_engine_snapshot(monkeypatch, hooks: dict[str, _Once]) -> None:
    """엔진 스냅샷(document_id·faq_id 집계) 직후에도 같은 쪽 훅을 부른다 — 두 스냅샷 중 먼저 끝난 쪽 뒤에 걸린다."""
    real = os_reconcile._indexed_parent_ids
    field_table = {'document_id': 'documents', 'faq_id': 'faqs'}

    async def snapshot_then_fire(field, tenant_id):
        ids = await real(field, tenant_id)
        if field_table[field] in hooks:
            await hooks[field_table[field]]()
        return ids

    monkeypatch.setattr(os_reconcile, '_indexed_parent_ids', snapshot_then_fire)


def _faq_appears(client, holder: dict) -> _Once:
    """FAQ를 새로 등록하고 바로 색인한다 — PG와 엔진에 동시에 나타난다."""

    async def act():
        res = await client.post('/kms/faqs', json={'question': '배송비는?', 'variants': [], 'answer': '무료'})
        holder['faq_id'] = res.json()['id']
        assert await sync_faq(holder['faq_id']) == {'done': 1, 'failed': 0}

    return _Once(act)


def _doc_becomes_ready(doc_id: int) -> _Once:
    async def act():
        assert await ingest(doc_id) == {'done': 1, 'failed': 0}

    return _Once(act)


@pytest.mark.asyncio
async def test_INV16_reconcile_faqs는_두_스냅샷_사이에_생긴_FAQ의_청크를_지우지_않는다(
    client, tenant_id, fake_embed, monkeypatch
):
    """INV-16 reconcile_faqs도 엔진을 PG보다 먼저 읽는다 — 반대면 사이에 등록·색인된 FAQ를 잉여로 보고 지운다."""
    holder: dict = {}
    hooks = {'faqs': _faq_appears(client, holder)}
    _hook_engine_snapshot(monkeypatch, hooks)
    async with AsyncSessionLocal() as s:
        r = await os_reconcile.reconcile_faqs(_PgHook(s, hooks), tenant_id)
    assert hooks['faqs'].fired
    assert r['deleted'] == 0, f'두 스냅샷 사이에 생긴 FAQ 청크를 지웠다: {r}'
    assert await faq_doc(holder['faq_id']) is not None
    assert r['indexed'] == 1  # 무해한 방향 — 재등재


@pytest.mark.asyncio
async def test_INV16_dry_run_대조도_엔진을_먼저_읽어_새_문서와_FAQ를_잉여로_보지_않는다(
    client, tenant_id, blob_tmp, fake_embed, monkeypatch
):
    """_diff(대조만)도 같은 계약 — 문서·FAQ 쌍마다 엔진 스냅샷이 PG 스냅샷보다 먼저여야 한다."""
    doc_id = (await upload(client, '환불정책.md', MD))['document_id']
    holder: dict = {}
    hooks = {'documents': _doc_becomes_ready(doc_id), 'faqs': _faq_appears(client, holder)}
    _hook_engine_snapshot(monkeypatch, hooks)
    async with AsyncSessionLocal() as s:
        (doc_missing, doc_extra), (faq_missing, faq_extra) = await os_reconcile._diff(_PgHook(s, hooks), tenant_id)
    assert all(h.fired for h in hooks.values())
    assert doc_id not in doc_extra and holder['faq_id'] not in faq_extra, '스냅샷 사이에 생긴 것을 잉여로 봤다'
    assert doc_id in doc_missing and holder['faq_id'] in faq_missing
    assert await indexed_chunk_texts(doc_id)
