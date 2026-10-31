"""drain 한 행의 처리 순서 — DB 없이 가짜 세션으로 SQL·commit·지표·훅의 순서를 고정한다.

통합 테스트는 결과 상태만 보므로, 커밋 경계(확정 처리가 한 커밋인가)와 지표 라벨은 여기서만 잡힌다.
"""

import asyncio

import pytest
from sqlalchemy.sql import Select, Update

from rag import outbox
from rag.metrics import INDEX_TOTAL, SEARCH_INDEX_SYNC_TOTAL

ROW = 7
DOC = 42


def _shape(stmt) -> str:
    """문장을 '동사 테이블 [SET 컬럼] [RETURNING] [WHERE 조건]' 꼴로 — SET 값은 빼고 구조와 조건만 비교한다."""
    if isinstance(stmt, Update):
        cols = ','.join(sorted(c.name for c in stmt._values))
        tail = ' RETURNING' if stmt._returning else ''
        where = str(stmt.whereclause.compile(compile_kwargs={'literal_binds': True}))
        return f'UPDATE {stmt.table.name} SET {cols}{tail} WHERE {where}'
    if isinstance(stmt, Select):
        return f'SELECT {stmt.get_final_froms()[0].name}'
    raise TypeError(type(stmt))


class _Result:
    def __init__(self, session: '_Session', shape: str) -> None:
        self._s, self._shape = session, shape

    def all(self):
        return self._s.rows

    def scalar(self):
        return self._s.attempts if 'RETURNING' in self._shape else self._s.filename


class _Session:
    """execute·commit·rollback·add를 순서대로 기록한다. UPDATE … RETURNING은 attempts, SELECT documents는 filename을 돌려준다."""

    def __init__(self, rows, *, attempts: int = 1, filename: str | None = 'a.pdf') -> None:
        self.rows, self.attempts, self.filename = rows, attempts, filename
        self.log: list[str] = []
        self.set_values: dict[str, object] = {}  # 마지막 UPDATE의 SET 값

    def _record_values(self, stmt) -> None:
        if isinstance(stmt, Update):
            self.set_values = {c.name: getattr(v, 'value', v) for c, v in stmt._values.items()}

    async def execute(self, stmt):
        shape = _shape(stmt)
        self.log.append(shape)
        self._record_values(stmt)
        return _Result(self, shape)

    async def commit(self) -> None:
        self.log.append('COMMIT')

    async def rollback(self) -> None:
        self.log.append('ROLLBACK')

    def add(self, row) -> None:
        self.log.append(f'ADD {row.op} {row.payload}')


def _counter(metric, **labels) -> float:
    return metric.labels(**labels)._value.get()


def _apply_raising(exc: BaseException):
    async def _apply(session, op, payload, row_id):
        raise exc

    return _apply


@pytest.mark.asyncio
async def test_성공한_행은_mark_done과_같은_커밋이고_ok_지표가_오른다(monkeypatch):
    async def _ok(session, op, payload, row_id):
        pass

    monkeypatch.setattr(outbox, '_apply', _ok)
    s = _Session([(ROW, 't', outbox.META_FAQS, None)])  # payload NULL도 빈 dict로 다룬다
    before = _counter(SEARCH_INDEX_SYNC_TOTAL, op=outbox.META_FAQS, result='ok')

    assert await outbox.drain(s) == {'done': 1, 'failed': 0}
    assert s.log == [
        'SELECT search_index_outbox',
        f'UPDATE search_index_outbox SET status WHERE search_index_outbox.id = {ROW}',
        'COMMIT',
    ]
    assert _counter(SEARCH_INDEX_SYNC_TOTAL, op=outbox.META_FAQS, result='ok') == before + 1


@pytest.mark.asyncio
async def test_INDEX_DOCUMENT_성공은_drain이_done을_찍지_않는다(monkeypatch):
    async def _ok(session, op, payload, row_id):
        pass

    monkeypatch.setattr(outbox, '_apply', _ok)
    s = _Session([(ROW, 't', outbox.INDEX_DOCUMENT, {'document_id': DOC})])
    assert await outbox.drain(s) == {'done': 1, 'failed': 0}
    assert s.log == ['SELECT search_index_outbox', 'COMMIT']  # done은 핸들러가 ready 승격과 같은 커밋에


@pytest.mark.asyncio
async def test_일시_실패는_백오프만_미루고_문서_지표는_건드리지_않는다(monkeypatch):
    monkeypatch.setattr(outbox, '_apply', _apply_raising(ConnectionError('down')))
    hook = []
    monkeypatch.setattr(outbox, '_on_failed', lambda *a: hook.append(a))
    s = _Session([(ROW, 't', outbox.META_DOCUMENTS, {'document_ids': [DOC]})], attempts=3)
    err_before = _counter(SEARCH_INDEX_SYNC_TOTAL, op=outbox.META_DOCUMENTS, result='error')
    idx_before = {r: _counter(INDEX_TOTAL, ext='pdf', result=r) for r in ('retry', 'failed')}

    assert await outbox.drain(s) == {'done': 0, 'failed': 1}
    assert s.log == [
        'SELECT search_index_outbox',
        'ROLLBACK',
        f'UPDATE search_index_outbox SET attempts,last_error RETURNING WHERE search_index_outbox.id = {ROW}',
        f'UPDATE search_index_outbox SET next_attempt_at WHERE search_index_outbox.id = {ROW}',
        'COMMIT',
    ]
    assert hook == []
    assert _counter(SEARCH_INDEX_SYNC_TOTAL, op=outbox.META_DOCUMENTS, result='error') == err_before + 1
    assert {r: _counter(INDEX_TOTAL, ext='pdf', result=r) for r in ('retry', 'failed')} == idx_before


@pytest.mark.asyncio
async def test_INDEX_DOCUMENT_일시_실패는_retry로_센다(monkeypatch):
    monkeypatch.setattr(outbox, '_apply', _apply_raising(TimeoutError('slow')))
    s = _Session([(ROW, 't', outbox.INDEX_DOCUMENT, {'document_id': DOC})])
    before = _counter(INDEX_TOTAL, ext='pdf', result='retry')

    assert await outbox.drain(s) == {'done': 0, 'failed': 1}
    assert s.log[-2:] == ['COMMIT', 'SELECT documents']  # 파일명 조회는 커밋 뒤
    assert _counter(INDEX_TOTAL, ext='pdf', result='retry') == before + 1


@pytest.mark.asyncio
async def test_INDEX_DOCUMENT_확정은_행_문서_DROP을_한_커밋에_넣고_훅_뒤에_failed로_센다(monkeypatch):
    monkeypatch.setattr(outbox, '_apply', _apply_raising(ValueError('bad blob')))
    hook = []
    monkeypatch.setattr(outbox, '_on_failed', lambda *a: hook.append(a))
    s = _Session([(ROW, 't', outbox.INDEX_DOCUMENT, {'document_id': DOC})], filename='x.docx')
    before = _counter(INDEX_TOTAL, ext='docx', result='failed')

    assert await outbox.drain(s) == {'done': 0, 'failed': 1}
    assert s.log == [
        'SELECT search_index_outbox',
        'ROLLBACK',
        f'UPDATE search_index_outbox SET attempts,last_error RETURNING WHERE search_index_outbox.id = {ROW}',
        f'UPDATE search_index_outbox SET status WHERE search_index_outbox.id = {ROW}',
        f"UPDATE documents SET status,status_reason WHERE documents.id = {DOC} AND documents.status = 'pending'",
        f"ADD {outbox.DROP_DOCUMENTS} {{'document_ids': [{DOC}]}}",
        'COMMIT',
        'SELECT documents',
    ]
    assert hook == [(outbox.INDEX_DOCUMENT, ROW, 'bad blob')]
    assert _counter(INDEX_TOTAL, ext='docx', result='failed') == before + 1


@pytest.mark.asyncio
async def test_MAX_ATTEMPTS에_닿은_일시_실패는_확정이다(monkeypatch):
    monkeypatch.setattr(outbox, '_apply', _apply_raising(ConnectionError('down')))
    monkeypatch.setattr(outbox, 'MAX_ATTEMPTS', 3)
    hook = []
    monkeypatch.setattr(outbox, '_on_failed', lambda *a: hook.append(a))
    s = _Session([(ROW, 't', outbox.DROP_FAQS, {'faq_ids': [1]})], attempts=3)

    assert await outbox.drain(s) == {'done': 0, 'failed': 1}
    assert s.log[3:] == [
        f'UPDATE search_index_outbox SET status WHERE search_index_outbox.id = {ROW}',
        'COMMIT',
    ]  # 문서 행이 아니라 DROP 등재 없음
    assert hook == [(outbox.DROP_FAQS, ROW, 'down')]


@pytest.mark.asyncio
async def test_파일명이_없는_문서는_빈_확장자로_센다(monkeypatch):
    monkeypatch.setattr(outbox, '_apply', _apply_raising(ValueError('x')))
    monkeypatch.setattr(outbox, '_on_failed', lambda *a: None)
    s = _Session([(ROW, 't', outbox.INDEX_DOCUMENT, {'document_id': DOC})], filename=None)
    before = _counter(INDEX_TOTAL, ext=outbox.ext_label(''), result='failed')
    await outbox.drain(s)
    assert _counter(INDEX_TOTAL, ext=outbox.ext_label(''), result='failed') == before + 1


@pytest.mark.asyncio
async def test_오류_문구는_500자에서_자른다(monkeypatch):
    monkeypatch.setattr(outbox, '_apply', _apply_raising(ValueError('x' * 900)))
    monkeypatch.setattr(outbox, '_on_failed', lambda *a: None)
    s = _Session([(ROW, 't', outbox.INDEX_DOCUMENT, {'document_id': DOC})], attempts=1)
    await outbox.drain(s)
    reason = s.set_values['status_reason']  # 마지막 UPDATE = documents
    assert reason.startswith('색인 1회 실패: xxx') and len(reason) == 500


@pytest.mark.asyncio
async def test_취소는_그대로_올라가고_행을_건드리지_않는다(monkeypatch):
    monkeypatch.setattr(outbox, '_apply', _apply_raising(asyncio.CancelledError()))
    s = _Session([(ROW, 't', outbox.META_FAQS, {'faq_ids': [1]})])
    with pytest.raises(asyncio.CancelledError):
        await outbox.drain(s)
    assert s.log == ['SELECT search_index_outbox']


@pytest.mark.asyncio
async def test_한_행의_실패가_다음_행을_막지_않는다(monkeypatch):
    calls = []

    async def _second_fails(session, op, payload, row_id):
        calls.append(row_id)
        if row_id == 1:
            raise ValueError('x')

    monkeypatch.setattr(outbox, '_apply', _second_fails)
    monkeypatch.setattr(outbox, '_on_failed', lambda *a: None)
    s = _Session([(1, 't', outbox.META_FAQS, {}), (2, 't', outbox.META_FAQS, {})])
    assert await outbox.drain(s) == {'done': 1, 'failed': 1}
    assert calls == [1, 2]
