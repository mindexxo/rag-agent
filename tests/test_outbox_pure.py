"""outbox 실패 분류·백오프 눈금 — DB 없이 (#185). 정의점은 rag/outbox.py의 is_transient·backoff_delay."""
from datetime import timedelta

import httpx
import pytest
from opensearchpy import ConnectionError as OsConnectionError, ConnectionTimeout, NotFoundError, RequestError, TransportError
from sqlalchemy.exc import InterfaceError, IntegrityError, OperationalError, ProgrammingError

from rag import outbox


def _http_status(code: int) -> httpx.HTTPStatusError:
    req = httpx.Request('POST', 'http://tei/embed')
    return httpx.HTTPStatusError('x', request=req, response=httpx.Response(code, request=req))


@pytest.mark.parametrize('exc', [
    ConnectionRefusedError('refused'),                    # builtin ConnectionError 하위
    TimeoutError('t'),
    httpx.ConnectError('c'),
    httpx.ReadTimeout('r'),
    httpx.RemoteProtocolError('p'),
    _http_status(429), _http_status(503),
    OsConnectionError('N/A', 'Cannot connect', None),
    ConnectionTimeout('TIMEOUT', 'timed out', None),
    TransportError(503, 'unavailable', None),
    TransportError(429, 'too many', None),
    TransportError('N/A', 'conn', None),
    OperationalError('SELECT 1', {}, Exception('connection was closed')),   # PG 커넥션 순단(SQLAlchemy 래핑)
    InterfaceError('SELECT 1', {}, Exception('connection is closed')),
])
def test_일시_실패로_분류(exc):
    assert outbox.is_transient(exc) is True


@pytest.mark.parametrize('exc', [
    ValueError('추출된 텍스트가 없습니다'),                # 빈 파일·파싱
    RuntimeError('docling 변환 불완전'),
    RuntimeError('bulk 색인 실패: mapper_parsing_exception'),
    FileNotFoundError('[Errno 2] No such file'),          # blob 없음 — OSError지만 ConnectionError 아님
    KeyError('document_id'),
    _http_status(400), _http_status(413),
    NotFoundError(404, 'index_not_found', None),
    RequestError(400, 'parsing_exception', None),
    IntegrityError('INSERT', {}, Exception('unique_violation')),          # DB지만 연결 문제가 아니다
    ProgrammingError('SELECT nope', {}, Exception('column does not exist')),
])
def test_결정적_실패로_분류(exc):
    assert outbox.is_transient(exc) is False


def test_백오프는_두_배씩_커지다_상한에_멈춘다():
    minutes = [outbox.backoff_delay(a) // timedelta(minutes=1) for a in range(1, outbox.MAX_ATTEMPTS + 1)]
    assert minutes[:7] == [1, 2, 4, 8, 16, 32, 60]
    assert all(m == outbox.BACKOFF_CAP_MINUTES for m in minutes[6:])
    # 총 창 — "반나절 죽었다 살아나도 사람 손 없이"의 눈금. 바꾸면 모듈 docstring의 시간도 같이.
    assert 18 <= sum(minutes) / 60 <= 20
