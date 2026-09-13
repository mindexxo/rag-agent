"""워커 프로세스 지표 서버의 기동 조건 (#151).

arq 워커에는 HTTP 서버가 없어 웹의 /metrics(main.py)가 워커를 덮지 못한다 — 이 서버가
워커 메모리를 보는 유일한 창이고, 그래서 "언제 뜨고 언제 안 뜨는지"가 계약이다.
실제 포트를 바인딩해 확인한다 — start_http_server를 모킹하면 시그니처 변경을 못 잡는다.
"""
import socket
import urllib.request

import pytest

from config import settings
from rag import worker


@pytest.fixture(autouse=True)
def index_calls(monkeypatch):
    """startup의 나머지 절반(인덱스 보장)은 이 파일의 관심이 아니다 — 네트워크를 타지 않게 막되,
    호출 여부는 기록한다. '지표 서버가 실패해도 본업은 계속된다'를 이걸로 검사한다."""
    calls = []

    async def _noop():
        calls.append(1)

    monkeypatch.setattr(worker.opensearch, 'ensure_index_soft', _noop)
    return calls


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


@pytest.mark.asyncio
async def test_포트가_0이면_서버를_안_띄운다(monkeypatch, index_calls):
    started = []
    monkeypatch.setattr(settings, 'worker_metrics_port', 0)
    monkeypatch.setattr(worker, 'start_http_server', lambda *a, **k: started.append(a))

    await worker.startup({})

    assert started == []
    assert index_calls == [1]


@pytest.mark.asyncio
async def test_바인드_주소는_설정을_따른다(monkeypatch, index_calls):
    # 브리지 네트워크 컨테이너에서는 루프백에 리슨하면 호스트에서 못 긁는다(config.py 주석) —
    # 주소가 설정에서 오지 않으면 그 배포에서 지표가 조용히 사라진다.
    seen = {}
    monkeypatch.setattr(settings, 'worker_metrics_port', 9999)
    monkeypatch.setattr(settings, 'worker_metrics_host', '0.0.0.0')
    monkeypatch.setattr(worker, 'start_http_server', lambda port, addr: seen.update(port=port, addr=addr))

    await worker.startup({})

    assert seen == {'port': 9999, 'addr': '0.0.0.0'}


@pytest.mark.asyncio
async def test_포트를_주면_루프백에_지표를_연다(monkeypatch, index_calls):
    port = _free_port()
    monkeypatch.setattr(settings, 'worker_metrics_port', port)
    monkeypatch.setattr(settings, 'worker_metrics_host', '127.0.0.1')

    await worker.startup({})

    body = urllib.request.urlopen(f'http://127.0.0.1:{port}/metrics', timeout=5).read().decode()
    assert 'python_info' in body          # 기본 레지스트리가 실제로 실려 나온다
    assert index_calls == [1]
    # process_resident_memory_bytes는 Linux에서만 나온다(ProcessCollector가 /proc를 읽는다).
    # 그래서 여기서 단언하지 않는다 — macOS 개발 머신에서 조용히 빨개지는 테스트가 된다.


@pytest.mark.asyncio
async def test_포트가_점유돼도_색인은_계속한다(monkeypatch, index_calls):
    """관측 장치가 워커를 죽이면 안 된다 — 인덱싱이 본업이다 (_start_metrics_server 주석)."""
    with socket.socket() as blocker:
        blocker.bind(('127.0.0.1', 0))
        blocker.listen(1)
        monkeypatch.setattr(settings, 'worker_metrics_port', blocker.getsockname()[1])

        await worker.startup({})          # 예외가 올라오면 실패

    assert index_calls == [1]
