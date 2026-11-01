"""그림 캡션 VLM 대역 — 로컬 HTTP 서버가 OpenAI 호환 /v1/chat/completions를 흉내 낸다. conftest가 픽스처를 재노출한다."""

import json

import pytest

# docling 캡션은 OpenAI 호환 /v1/chat/completions를 HTTP로 부른다. 테스트는 로컬 HTTP 서버로 대신한다 —
# 외부 서비스 없이 결정적이고, 실패(5xx)·지연까지 주입할 수 있다. 캡션 실패는 문서 재시도라
# VLM을 죽인 채로는 그림 문서를 변환할 수 없다.
# 변환은 자식 프로세스(rag/chunking/convert_process.ConvertProcess)에서 일어나지만 HTTP라 그대로 닿는다 — 설정(URL)은
# 부모의 스냅숏이 자식에 넘어간다.
FAKE_CAPTION = '테스트 캡션: 로그인 화면'


class FakeVlm:
    def __init__(self):
        self.caption = FAKE_CAPTION
        self.status = 200  # 5xx로 바꾸면 docling은 빈 캡션을 준다 → CaptionFailedError
        self.delay = 0.0  # 초 — 마감 테스트용
        self.calls = 0
        self.finish_reason = 'stop'  # 'length'면 출력이 상한에서 잘린 응답(상용 OCR은 결정적 실패로 올린다)


@pytest.fixture(scope='session')
def _fake_vlm_server():
    """세션당 서버 하나 — URL이 테스트마다 바뀌면 설정 스냅숏이 달라져 변환 프로세스가 매번 재기동된다."""
    import threading
    import time
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    vlm = FakeVlm()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers.get('content-length') or 0))
            vlm.calls += 1
            if vlm.delay:
                time.sleep(vlm.delay)
            body = json.dumps(
                {
                    'id': 'fake',
                    'created': 0,
                    'model': 'fake',
                    'choices': [
                        {
                            'index': 0,
                            'finish_reason': vlm.finish_reason,
                            'message': {'role': 'assistant', 'content': vlm.caption},
                        }
                    ],
                }
            ).encode()
            self.send_response(vlm.status)
            self.send_header('content-type', 'application/json')
            self.send_header('content-length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield vlm, f'http://127.0.0.1:{server.server_port}/v1/chat/completions'
    server.shutdown()


@pytest.fixture
def fake_vlm(_fake_vlm_server, monkeypatch):
    from config import settings

    vlm, url = _fake_vlm_server
    vlm.__init__()  # 상태(캡션·실패·지연·호출 수)는 테스트마다 초기화
    monkeypatch.setattr(settings, 'vlm_caption_url', url)
    return vlm
