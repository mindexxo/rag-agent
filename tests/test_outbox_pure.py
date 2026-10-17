"""outbox 실패 분류·백오프 눈금 — DB 없이 (#185). 정의점은 rag/outbox.py의 is_transient·backoff_delay."""
from datetime import timedelta

import httpx
import pytest
from opensearchpy import ConnectionError as OsConnectionError, ConnectionTimeout, NotFoundError, RequestError, TransportError
from sqlalchemy.exc import InterfaceError, IntegrityError, OperationalError, ProgrammingError

from config import settings
from rag import outbox
from rag.chunking import CaptionFailedError, PictureLimitError


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
    CaptionFailedError('그림 캡션 실패'),                 # #223 결정 3 — VLM이 살아나면 저절로 ready
    TimeoutError('docling 변환이 900초를 넘겨 중단했다'),   # #223 결정 5 — 마감 초과도 일시(24회)
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
    PictureLimitError('그림 101장'),                       # #223 결정 4 — 다시 돌려도 그림 수는 같다
])
def test_결정적_실패로_분류(exc):
    assert outbox.is_transient(exc) is False


def test_백오프는_두_배씩_커지다_상한에_멈춘다():
    minutes = [outbox.backoff_delay(a) // timedelta(minutes=1) for a in range(1, outbox.MAX_ATTEMPTS + 1)]
    assert minutes[:7] == [1, 2, 4, 8, 16, 32, 60]
    assert all(m == outbox.BACKOFF_CAP_MINUTES for m in minutes[6:])
    # 총 창 — "반나절 죽었다 살아나도 사람 손 없이"의 눈금. 바꾸면 모듈 docstring의 시간도 같이.
    assert 18 <= sum(minutes) / 60 <= 20


class TestDrainRoundBudget:
    """회차 예산(#223 결정 2) — 행을 시작하기 전에 "최악으로 걸려도 arq job_timeout 안인가"를 본다."""

    def test_첫_행은_항상_시작한다(self):
        assert outbox._fits_round(0.0)

    def test_큰_문서_뒤에는_회차를_접는다(self):
        # 경계 = 여유 180초. 작은 문서(수 초)는 여러 건이 한 회차에 흐른다
        edge = outbox.DRAIN_ROUND_SECONDS - settings.docling_document_timeout_seconds - outbox.INDEX_TAIL_SECONDS
        assert edge == outbox.ROUND_SLACK_SECONDS == 180
        assert outbox._fits_round(edge) and not outbox._fits_round(edge + 0.1)

    def test_회차_예산은_마감에서_계산된다(self):
        # 고정값이면 마감을 올렸을 때 첫 행조차 예산에 못 들어가 대기열이 로그 없이 멈췄다(#223 리뷰)
        assert outbox.DRAIN_ROUND_SECONDS == (settings.docling_document_timeout_seconds
                                              + outbox.INDEX_TAIL_SECONDS + outbox.ROUND_SLACK_SECONDS)

    def test_워커_job_timeout이_회차_예산과_같다(self):
        # 둘이 어긋나면(예산 > job_timeout) arq가 회차를 취소해 처리 중이던 문서가 헛돈다(CancelledError, attempts 그대로)
        from rag.worker import WorkerSettings
        assert WorkerSettings.job_timeout == outbox.DRAIN_ROUND_SECONDS

    @pytest.mark.asyncio
    async def test_예산이_없으면_행을_건드리지_않고_끝낸다(self, monkeypatch):
        # 세션은 SELECT 한 번만 받는다 — 행 처리(_apply·commit)에 들어가면 가짜 세션이 터진다
        class Rows:
            def all(self):
                return [(1, 't', outbox.INDEX_DOCUMENT, {'document_id': 1})]

        class Session:
            async def execute(self, stmt):
                return Rows()
        monkeypatch.setattr(outbox, '_fits_round', lambda elapsed: False)
        assert await outbox.drain(Session()) == {'done': 0, 'failed': 0}
