"""불변식 대장(INVARIANTS.md)의 빈 칸 채우기 — 문서·색인·인용 축 (#315).

각 테스트는 이름이 INV 번호로 시작하고 docstring 첫 줄에 그 규칙을 적는다. 규칙의 "왜"는 정의점 파일
(대장의 정의점 열)에 있다 — 여기는 그 규칙이 깨지면 실패하는 단언만 둔다.

공유 개발계 DB·OpenSearch를 쓰므로 drain은 conftest `ingest`처럼 문서 단위로 한정하고,
reconcile은 반드시 tenant_id를 넘긴다(전역 apply는 다른 세션의 문서를 지울 수 있다).
"""

import asyncio
import re
from dataclasses import dataclass
from pathlib import Path

import pytest
from sqlalchemy import event, func, select

from database import AsyncSessionLocal, engine
from rag import limiter, os_reconcile, otel, outbox
from rag.models import Document, Message, TurnStatus
from rag.search.retrieval_types import RetrievalResult, RetrievedChunk
from rag.turn.citation_labels import citation_tail, sources_from_chunks
from rag.turn.prepared import PreparedRag
from rag.turn.service import RagService
from rag.turn.streaming import _run_generation
from tests.conftest import indexed_chunk_texts, ingest, sse_done, sse_meta
from tests.helpers_documents import MD, get_doc, upload

# ── INV-16 reconcile 읽기 순서 ─────────────────────────────────────────────


class _SessionProxy:
    """reconcile에 넘기는 세션 대역 — 첫 execute(=PG 스냅샷)가 끝난 직후 훅을 부를 수 있게 한다.
    순서를 뒤집은 구현(PG 먼저)에서도 훅이 "두 스냅샷 사이"에 끼도록 하려는 장치다."""

    def __init__(self, session, after_read):
        self._s, self._after_read = session, after_read

    async def execute(self, *a, **kw):
        res = await self._s.execute(*a, **kw)
        await self._after_read()
        return res

    def __getattr__(self, name):
        return getattr(self._s, name)


@dataclass
class _IngestOnce:
    """첫 호출에서만 doc_id를 색인·ready로 만든다 — 몇 번째 스냅샷 직후에 불렸는지와 무관하게 한 번."""

    doc_id: int
    fired: bool = False

    async def __call__(self) -> None:
        if not self.fired:
            self.fired = True
            assert await ingest(self.doc_id) == {'done': 1, 'failed': 0}


def _between_snapshots_hook(monkeypatch, doc_id) -> _IngestOnce:
    """엔진 스냅샷·PG 스냅샷 중 **먼저 끝난 쪽** 직후에 doc_id를 색인·ready로 만든다(한 번만).
    올바른 순서(엔진 먼저)면 _indexed_parent_ids 반환 직후, 뒤집힌 순서면 PG 조회 직후에 걸린다."""
    fire = _IngestOnce(doc_id)
    real = os_reconcile._indexed_parent_ids

    async def snapshot_then_fire(field, tenant_id):
        ids = await real(field, tenant_id)
        await fire()
        return ids

    monkeypatch.setattr(os_reconcile, '_indexed_parent_ids', snapshot_then_fire)
    return fire


@pytest.mark.asyncio
async def test_INV16_reconcile은_두_스냅샷_사이에_ready가_된_문서의_청크를_지우지_않는다(
    client, tenant_id, blob_tmp, fake_embed, monkeypatch
):
    """INV-16 reconcile은 엔진 스냅샷을 PG보다 먼저 읽는다 — 반대면 방금 색인된 청크를 유령으로 지운다.

    스냅샷 사이에 인제스션이 끼면 그 문서는 PG에만 있어 '누락'(재등재 — 무해)이어야 하고,
    '잉여'(삭제)가 되면 안 된다."""
    doc_id = (await upload(client, '환불정책.md', MD))['document_id']  # pending — 아직 어느 스냅샷에도 없다
    fire = _between_snapshots_hook(monkeypatch, doc_id)

    async with AsyncSessionLocal() as s:
        r = await os_reconcile.reconcile(_SessionProxy(s, fire), tenant_id)  # 테넌트 한정
    assert fire.fired, '훅이 두 스냅샷 사이에서 불리지 않았다'
    assert r['deleted'] == 0, f'두 스냅샷 사이에 ready가 된 문서 청크를 지웠다: {r}'
    assert await indexed_chunk_texts(doc_id), '방금 색인된 청크가 사라졌다'
    assert r['indexed'] == 1  # 무해한 방향 — 재등재(pending + INDEX 행)
    assert await ingest(doc_id) == {'done': 1, 'failed': 0}  # 재등재분도 정상 처리된다
    assert (await get_doc(doc_id)).status == 'ready'


# ── INV-17 옛 blob은 커밋 뒤에 지운다 ──────────────────────────────────────


async def _failed_document(client, filename) -> Document:
    """③ 커밋 단계를 터뜨려 failed로 확정된 문서(tests/test_integration_index_sync._break_commit_stage 패턴).
    monkeypatch.context로 감싸 blob_tmp·fake_embed의 패치는 되돌리지 않는다."""
    doc_id = (await upload(client, filename, MD))['document_id']

    async def boom(session, row_id):
        raise RuntimeError('③ 커밋 직전 폭발')

    with pytest.MonkeyPatch.context() as m:
        m.setattr(outbox, 'mark_done', boom)
        assert await ingest(doc_id) == {'done': 0, 'failed': 1}
    doc = await get_doc(doc_id)
    assert doc.status == 'failed'
    return doc


def _break_upload_commit(monkeypatch):
    """업로드 라우터의 session.commit만 실패시킨다 — handle_upload가 끝난 바로 그 세션에 한정."""
    import routers.documents as rd

    real = rd.handle_upload

    async def handle_then_break(session, *a, **kw):
        out = await real(session, *a, **kw)

        async def commit_fails():
            raise RuntimeError('업로드 커밋 실패 (테스트 주입)')

        session.commit = commit_fails
        return out

    monkeypatch.setattr(rd, 'handle_upload', handle_then_break)


@pytest.mark.asyncio
async def test_INV17_재업로드_커밋이_실패하면_옛_blob은_디스크에_남는다(
    client, tenant_id, blob_tmp, fake_embed, monkeypatch
):
    """INV-17 옛 blob은 호출부가 커밋 뒤에 지운다 — 커밋 전에 지우면 롤백된 DB가 없는 파일을 가리킨다.

    옛 blob을 지우는 경로는 failed 행 재사용(#161)뿐이라 failed 문서를 같은 이름으로 다시 올린다."""
    old = await _failed_document(client, '환불정책.md')
    _break_upload_commit(monkeypatch)
    with pytest.raises(RuntimeError, match='업로드 커밋 실패'):
        await client.post('/kms/documents', files={'file': ('환불정책.md', MD, 'text/markdown')})

    after = await get_doc(old.id)
    assert after.status == 'failed' and after.blob_path == old.blob_path  # 롤백 — DB는 옛 경로 그대로
    assert Path(old.blob_path).is_file(), 'DB가 가리키는 옛 blob이 커밋 실패에도 지워졌다'


@pytest.mark.asyncio
async def test_INV17_재업로드_커밋이_성공하면_옛_blob을_지운다(client, tenant_id, blob_tmp, fake_embed):
    """INV-17 옛 blob은 호출부가 커밋 뒤에 지운다 — 위 테스트가 공허하지 않다는 대조(지우는 경로가 실제로 돈다)."""
    old = await _failed_document(client, '환불정책.md')
    again = await upload(client, '환불정책.md', MD)
    assert again['document_id'] == old.id  # failed 행 재사용(#161)
    new = await get_doc(old.id)
    assert new.blob_path != old.blob_path and Path(new.blob_path).is_file()
    assert not Path(old.blob_path).exists()


# ── INV-18 업로드 1건 = UUID 파일 1개 ─────────────────────────────────────


@pytest.mark.asyncio
async def test_INV18_같은_바이트를_여러_번_올려도_업로드마다_blob_파일이_따로_생긴다(
    client, tenant_id, blob_tmp, fake_embed
):
    """INV-18 업로드 1건 = UUID 파일 1개 — 내용이 같아도 blob을 공유하지 않는다.

    같은 파일명 재업로드(v1·v2)와 다른 파일명 셋 다 같은 바이트다."""
    ids = [
        (await upload(client, '환불정책.md', MD))['document_id'],
        (await upload(client, '환불정책.md', MD))['document_id'],  # 같은 이름 → 새 version
        (await upload(client, '반품정책.md', MD))['document_id'],
    ]
    docs = [await get_doc(i) for i in ids]
    assert [d.version for d in docs] == [1, 2, 1]
    paths = [d.blob_path for d in docs]
    assert len(set(paths)) == 3, f'같은 내용의 업로드가 blob을 공유한다: {paths}'
    assert all(Path(p).is_file() and Path(p).read_bytes() == MD for p in paths)
    assert all(Path(p).parent == blob_tmp / tenant_id for p in paths)
    assert len(list((blob_tmp / tenant_id).iterdir())) == 3


# ── INV-07 채팅 첨부 추출은 저장하지 않는다 ───────────────────────────────


async def _row_counts(tenant_id) -> tuple[int, int]:
    async with AsyncSessionLocal() as s:
        docs = await s.scalar(select(func.count()).select_from(Document).where(Document.tenant_id == tenant_id))
        msgs = await s.scalar(select(func.count()).select_from(Message).where(Message.tenant_id == tenant_id))
    return docs, msgs


def _blob_files(root) -> list:
    return sorted(p for p in root.rglob('*') if p.is_file())


def _dependency_calls(dependant) -> list:
    """라우트 의존성 트리의 호출 대상 전부(하위 의존성까지) — 라우터·앱 수준 의존성도 dependant에 합쳐져 있다."""
    out = []
    for dep in dependant.dependencies:
        out += [dep.call, *_dependency_calls(dep)]
    return out


@pytest.mark.asyncio
async def test_INV07_첨부_추출은_DB_행도_blob도_남기지_않는다(client, tenant_id, blob_tmp, fake_embed):
    """INV-07 채팅 첨부 추출은 저장하지 않는다(DB 행·blob 없음) — 저장은 질의 시점에 대화에 묶여 이뤄진다.

    테넌트 행 수만 보면 테넌트 없이 쓰는 저장을 못 잡으므로, 추출 동안 이 프로세스가 DB에 보낸
    쓰기 문장(INSERT·UPDATE·DELETE)도 엔진 이벤트로 센다 — 공유 DB라 전역 행 수는 쓸 수 없다.
    응답 뒤에 도는 쓰기(백그라운드 태스크·세션 의존성 정리)가 리스너 해제 뒤로 빠져나가지 않게 두 겹으로 막는다:
    ① 라우트 의존성 트리에 DB 세션(get_session)이 없다 ② 해제 전에 루프를 몇 번 양보해 남은 콜백을 돌린다."""
    from database import get_session
    from main import app

    (route,) = [r for r in app.routes if getattr(r, 'path', None) == '/kms/attachments/extract']
    assert get_session not in _dependency_calls(route.dependant), '첨부 추출 라우트가 DB 세션을 받는다'

    before = (await _row_counts(tenant_id), _blob_files(blob_tmp))
    writes: list[str] = []

    def on_execute(conn, cursor, statement, *a):
        if statement.lstrip().split(' ', 1)[0].upper() in ('INSERT', 'UPDATE', 'DELETE'):
            writes.append(statement)

    event.listen(engine.sync_engine, 'before_cursor_execute', on_execute)
    try:
        res = await client.post('/kms/attachments/extract', files={'file': ('고객자료.md', MD, 'text/markdown')})
        for _ in range(5):
            await asyncio.sleep(0)
    finally:
        event.remove(engine.sync_engine, 'before_cursor_execute', on_execute)

    assert res.status_code == 200, res.text
    assert '14일 이내' in res.json()['text']  # 추출은 실제로 됐다
    assert writes == []
    assert (await _row_counts(tenant_id), _blob_files(blob_tmp)) == before


# ── INV-51 API citations ≡ SSE done.citations ────────────────────────────


@pytest.mark.asyncio
async def test_INV51_일반_문서_인용은_SSE_done과_대화_API가_같은_키와_값이다(
    client, tenant_id, blob_tmp, fake_embed, fake_llm, pass_gate, relaxed_quota
):
    """INV-51 API `citations`와 SSE `done.citations`는 같은 이름·의미 — 일반 문서 인용(FAQ 아님)도.

    문서 둘을 색인하고 꼬리를 [2,1]로 내 인용 순서까지 본다(1번 고정이면 순서 어긋남을 못 잡는다)."""
    for name in ('환불정책.md', '반품정책.md'):
        await ingest((await upload(client, name, MD))['document_id'])
    fake_llm.answer = f'14일 이내입니다. {citation_tail([2, 1])}'
    headers = {'X-User-Id': 'agent-kim'}

    res = await client.post('/kms/query', json={'query': '반품 기간 알려줘'}, headers=headers)
    assert res.status_code == 200, res.text
    sse = sse_done(res)['citations']
    assert len(sse) == 2 and all(c['document_id'] is not None and c['faq_id'] is None for c in sse)
    assert {c['filename'] for c in sse} == {'환불정책.md', '반품정책.md'}

    conv_id = sse_meta(res)['conversation_id']
    msgs = (await client.get(f'/kms/conversations/{conv_id}/messages', headers=headers)).json()
    (assistant,) = [m for m in msgs if m['role'] == 'assistant']
    assert assistant['citations'] == sse  # 같은 키·같은 값·같은 순서


# ── INV-50 인용 번호: 컨텍스트 블록 ≡ 꼬리 제약 ≡ 파서 ─────────────────────


def _mixed_prepared() -> PreparedRag:
    """같은 문서 여러 조각·다른 문서·FAQ 둘이 섞인 검색 결과 + 첨부 둘. 본문은 출처마다 유일하다."""

    def ch(cid, text, document_id=None, filename='FAQ', faq_id=None, heading=None):
        return RetrievedChunk(
            chunk_id=cid,
            document_id=document_id,
            text=text,
            heading_path=heading or ['섹션'],
            page=None,
            filename=filename,
            version=1,
            faq_id=faq_id,
        )

    chunks = [
        ch(1, '본문-A1', 10, '가.pdf'),
        ch(2, '본문-B', 20, '나.docx'),
        ch(3, 'Q: 환불?\n본문-F7', faq_id=7, heading=['환불?']),
        ch(4, '본문-A2', 10, '가.pdf'),  # 같은 문서의 둘째 조각 — 새 번호를 받으면 안 된다
        ch(5, '본문-C', 30, '다.md'),
        ch(6, 'Q: 배송?\n본문-F8', faq_id=8, heading=['배송?']),
        ch(7, '본문-B2', 20, '나.docx'),
    ]
    attachments = [{'filename': '첨부1.txt', 'text': '본문-X1'}, {'filename': '첨부2.pdf', 'text': '본문-X2'}]
    return PreparedRag(
        conversation_id=1,
        assistant_message_id=1,
        original_query='질문',
        standalone_query='질문',
        prior_turns=[],
        retrieval=RetrievalResult(chunks=chunks, no_evidence=False, reason=None),
        sources=sources_from_chunks(chunks),  # rag/turn/service.py prepare()와 같은 파생
        source_doc_ids=[],
        attachments=attachments,
    )


# 출처 본문 표지 → 정체. 컨텍스트 블록 쪽과 파서 쪽을 같은 어휘로 비교하려는 것.
_BODY_IDENTITY = {
    '본문-A1': ('doc', 10),
    '본문-A2': ('doc', 10),
    '본문-B': ('doc', 20),
    '본문-B2': ('doc', 20),
    '본문-C': ('doc', 30),
    '본문-F7': ('faq', 7),
    '본문-F8': ('faq', 8),
    '본문-X1': ('att', '첨부1.txt'),
    '본문-X2': ('att', '첨부2.pdf'),
}


def _context_numbering(prompt_text: str) -> dict[int, set]:
    """프롬프트 → {[번호]: 그 블록 본문들의 정체 집합}. 블록 머리 '[n] …' 다음 줄들에서 본문 표지를 찾는다."""
    out: dict[int, set] = {}
    current = None
    for line in prompt_text.splitlines():
        m = re.match(r'^\[(\d+)\] ', line)
        if m:
            current = int(m.group(1))
            continue
        for marker, ident in _BODY_IDENTITY.items():
            if current is not None and line.endswith(marker):
                out.setdefault(current, set()).add(ident)
    return out


def _citation_identity(c) -> tuple:
    if c.faq_id is not None:
        return ('faq', c.faq_id)
    if c.document_id is not None:
        return ('doc', c.document_id)
    assert c.filename.startswith('첨부: '), c
    return ('att', c.filename.removeprefix('첨부: '))


@pytest.mark.asyncio
async def test_INV50_컨텍스트_블록_꼬리_제약_파서가_같은_입력에_같은_번호_객체_매핑을_낸다(
    tenant_id, fake_llm, monkeypatch
):
    """INV-50 인용 번호는 검색 출처 순서 1부터, 이어서 첨부 순서 — 컨텍스트 블록·꼬리 제약·파서가 같은 순서.

    운영 경로를 그대로 탄다: 생성 태스크(rag/turn/streaming._run_generation)가 RagService.generate를 불러 프롬프트와
    꼬리 제약을 만들고(LLM 호출에서 둘 다 캡처), 스트림 끝에서 파서가 인용을 확정한다(finalize에서 캡처).
    가짜 LLM은 받은 제약의 후보 수만큼 1..N을 꼬리에 단다 — 제약이 틀리면 꼬리도 그만큼 틀린다.
    DB 행은 없다(합성 PreparedRag) — finalize·maybe_cache는 캡처만 하고 쓰지 않는다."""
    prepared = _mixed_prepared()
    llm_calls: list[tuple[list[dict], object]] = []
    finalized: list[list] = []
    real_astream = fake_llm.astream

    async def capturing_astream(messages, *, citations=None, **kw):
        llm_calls.append((messages, citations))
        fake_llm.answer = f'답변입니다. {citation_tail(range(1, citations.count + 1))}'
        async for token in real_astream(messages, citations=citations, **kw):
            yield token

    async def capturing_finalize(self, prepared, answer, citations, status=TurnStatus.DONE, latency_ms=None):
        assert status == TurnStatus.DONE, f'생성이 정상 완료되지 않았다: {status}'
        finalized.append(citations)

    async def no_cache(self, *a, **kw):
        return None

    monkeypatch.setattr(fake_llm, 'astream', capturing_astream)
    monkeypatch.setattr(RagService, 'finalize', capturing_finalize)
    monkeypatch.setattr(RagService, 'maybe_cache', no_cache)
    lease = await limiter.try_acquire(tenant_id, 10, 'agent-x', 10)
    root_span, token = otel.start_turn()
    otel.detach_turn(token)
    await _run_generation(prepared, asyncio.Queue(), lease, 0.0, root_span)

    ((messages, constraint),) = llm_calls
    (parsed,) = finalized
    context = _context_numbering(messages[-1]['content'])
    expected = [
        ('doc', 10),
        ('doc', 20),
        ('faq', 7),
        ('doc', 30),
        ('faq', 8),
        ('att', '첨부1.txt'),
        ('att', '첨부2.pdf'),
    ]
    assert context == {n: {ident} for n, ident in enumerate(expected, start=1)}  # 블록: 첫 등장 순, 같은 출처 같은 번호
    assert constraint.count == len(context)  # 제약: 후보 수 = 컨텍스트 블록 수(출처 + 첨부)
    assert [_citation_identity(c) for c in parsed] == [
        ident for n in sorted(context) for ident in context[n]
    ]  # 파서: i → 블록 [i]
