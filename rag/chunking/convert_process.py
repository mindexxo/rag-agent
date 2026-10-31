"""docling 변환을 맡는 상주 자식 프로세스 — 마감·취소·설정 스냅숏으로 자식을 관리하고, 자식 안에서는 convert_local이 변환한다."""

import os
import pickle
import socket
import subprocess
import sys
import threading
from pathlib import Path
from typing import TYPE_CHECKING

from config import settings
from rag.chunking import document_converter, ocr_llm

if TYPE_CHECKING:
    from docling_core.types.doc import DoclingDocument


def convert_local(file_path: str | Path, **kwargs) -> 'DoclingDocument':
    """이 프로세스에서 docling 변환 한 번 → DoclingDocument. 변환 프로세스(_convert_child_main)가 부른다.

    상용 OCR(OCR_PROVIDER=llm)의 PDF는 ocr_llm.convert_pdf으로, 나머지는 기본 변환기로 document_converter.convert_with를 거친다.
    kwargs는 converter.convert로 넘긴다(PDF의 max_num_pages — 쪽 개념이 없는 형식엔 주지 않는다).
    """
    if settings.ocr_provider == 'llm' and Path(file_path).suffix.lower() == '.pdf':
        return ocr_llm.convert_pdf(file_path, **kwargs)
    return document_converter.convert_with(document_converter.docling_converter(), file_path, **kwargs)


def _picklable(exc: BaseException) -> BaseException:
    """부모로 보낼 수 있는 예외 — 피클이 안 되면 같은 문구의 RuntimeError(결정적)로 바꾼다."""
    try:
        pickle.dumps(exc)
        return exc
    except Exception:  # noqa: BLE001 — 직렬화 불가 예외는 종류를 잃는다(문구는 남긴다)
        return RuntimeError(f'{type(exc).__name__}: {exc}')


def _convert_child_main(fd: int) -> None:
    """변환 프로세스의 본체 — 요청 (경로, kwargs)를 받아 ('ok', DoclingDocument) 또는 ('err', 예외)를 돌려준다.

    첫 메시지는 부모의 설정 스냅숏(CHILD_SETTING_PREFIXES)이다 — 자식은 config를 .env에서 새로 읽으므로, 부모가 바꾼 값
    (테스트의 monkeypatch 등)이 자식에 닿는 유일한 길이다. 모델 로딩은 'ready' 전에 끝내 첫 문서의 마감을 깎지 않는다.
    """
    from multiprocessing.connection import Connection

    conn = Connection(fd)
    for key, value in conn.recv().items():
        setattr(settings, key, value)
    try:
        document_converter.docling_converter()
    except Exception as e:  # noqa: BLE001 — 기동 실패도 부모에 사유로 돌려준다
        conn.send(('err', _picklable(e)))
        return
    conn.send(('ready', None))
    while True:
        try:
            file_path, kwargs = conn.recv()
        except EOFError:  # 부모가 끊었다(재기동·종료) — 부모가 죽어도 여기서 빠진다
            return
        try:
            conn.send(('ok', convert_local(file_path, **kwargs)))
        except Exception as e:  # noqa: BLE001 — 문서 단위 실패는 부모가 분류한다(outbox.is_transient)
            conn.send(('err', _picklable(e)))


_CHILD_START_TIMEOUT_SECONDS = 300  # 변환 프로세스 기동(모델 로딩) 상한 — 첫 실행의 모델 다운로드까지 품는 여유
_PROJECT_ROOT = Path(__file__).resolve().parents[2]  # rag/chunking/ 아래라 두 단계 위가 저장소 루트


# 변환 프로세스가 쓰는 설정의 접두사 — 부모는 이 설정만 자식에 넘기고, 이것이 바뀌었을 때만 자식을 다시 띄운다
# (변환과 무관한 설정을 바꿀 때마다 콜드 기동하지 않게). 이 패키지가 읽는 settings 필드가 모두 이 접두사 안인지는
# tests/test_chunking_docling.py가 검사한다 — 빠뜨리면 값을 바꿔도 자식이 옛 값으로 돈다.
CHILD_SETTING_PREFIXES = (
    'docling_',
    'ocr_',
    'vlm_caption_',
)


def _child_settings() -> dict:
    return {k: v for k, v in settings.model_dump().items() if k.startswith(CHILD_SETTING_PREFIXES)}


class ConvertProcess:
    """docling 변환을 맡는 상주 자식 프로세스 하나(이 모듈을 쓰는 프로세스마다 하나).

    프로세스인 이유: docling에는 취소 API가 없어 스레드로 돌리면 마감을 넘겨도 멈출 수 없다. 프로세스는 kill하고 다음
    요청 때 새로 띄운다. 자식은 변환만 한다 — 임베딩·쓰기·커밋은 부모에서 그 뒤라, 변환 중 kill해도 반쪽 색인은 없다.
    - 마감: settings.docling_document_timeout_seconds(전 형식, 캡션 포함). 넘기면 TimeoutError(일시 실패).
    - 동시성 1: 잠금 하나로 요청을 직렬화한다.
    - 띄우는 법: `python -c` + 소켓 한 쌍 위의 multiprocessing Connection. spawn은 부모의 __main__ 스크립트를 자식에서
      다시 import하고(보호 없는 스크립트는 본문이 한 번 더 돈다), fork는 부모의 스레드·잠금 상태를 복제해 교착 위험이 있다.
    - 설정 스냅숏이 바뀌면 재기동한다 — 자식은 기동 때의 설정으로 변환기를 만든다.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None
        self._conn = None
        self._snapshot: dict | None = None

    def convert(self, file_path: str | Path, kwargs: dict) -> 'DoclingDocument':
        with self._lock:
            snapshot = _child_settings()
            if self._proc is None or self._proc.poll() is not None or snapshot != self._snapshot:
                self._start(snapshot)
            deadline = settings.docling_document_timeout_seconds
            self._conn.send((str(Path(file_path).resolve()), kwargs))  # 자식의 cwd는 저장소 루트다
            if not self._conn.poll(deadline):
                self._stop()
                raise TimeoutError(f'docling 변환이 {deadline:.0f}초를 넘겨 중단했다(변환 프로세스 재기동)')
            try:
                kind, value = self._conn.recv()
            except EOFError:  # 자식이 죽었다(OOM kill 등) — poll은 EOF에도 True다
                code = self._exit_code()
                self._stop()
                raise RuntimeError(f'docling 변환 프로세스가 비정상 종료했다(exitcode={code})') from None
        if kind == 'ok':
            return value
        raise value

    def _start(self, snapshot: dict) -> None:
        from multiprocessing.connection import Connection

        self._stop()
        parent_sock, child_sock = socket.socketpair()
        env = {
            **os.environ,
            'PYTHONPATH': os.pathsep.join(p for p in (str(_PROJECT_ROOT), os.environ.get('PYTHONPATH')) if p),
        }
        try:
            self._proc = subprocess.Popen(
                [
                    sys.executable,
                    '-c',
                    'import sys; from rag.chunking.convert_process import _convert_child_main; _convert_child_main(int(sys.argv[1]))',
                    str(child_sock.fileno()),
                ],
                pass_fds=(child_sock.fileno(),),
                cwd=_PROJECT_ROOT,
                env=env,
            )
        except BaseException:
            parent_sock.close()
            raise
        finally:
            child_sock.close()
        self._conn = Connection(parent_sock.detach())
        self._conn.send(snapshot)
        if not self._conn.poll(_CHILD_START_TIMEOUT_SECONDS):
            self._stop()
            raise TimeoutError(f'docling 변환 프로세스 기동이 {_CHILD_START_TIMEOUT_SECONDS}초를 넘겼다')
        try:
            kind, value = self._conn.recv()
        except EOFError:
            code = self._exit_code()
            self._stop()
            raise RuntimeError(f'docling 변환 프로세스가 기동 중 종료했다(exitcode={code})') from None
        if kind != 'ready':
            self._stop()
            raise value
        self._snapshot = snapshot

    def abort(self) -> None:
        """진행 중인 변환을 끊는다 — **잠금 없이** 부른다(잠금은 기다리는 스레드가 쥐고 있다). 다른 스레드에서 불려도
        안전하다: kill만 하고, 기다리던 convert()가 EOF로 깨어나 정리(_stop)하고 잠금을 푼다."""
        proc = self._proc
        if proc is not None and proc.poll() is None:
            proc.kill()

    def _exit_code(self):
        try:
            return self._proc.wait(5)
        except subprocess.TimeoutExpired:
            return None

    def _stop(self) -> None:
        if self._conn is not None:
            self._conn.close()
        if self._proc is not None and self._proc.poll() is None:
            self._proc.kill()  # 자식은 지킬 상태가 없다 — C 코드에 묶여 있어도 끊기는 SIGKILL로
            self._proc.wait(10)
        self._proc = self._conn = self._snapshot = None


process = ConvertProcess()


def abort_conversion() -> None:
    """진행 중인 docling 변환을 끊는다(변환 프로세스 kill). 색인 잡이 취소될 때 rag/ingestion.py가 부른다."""
    process.abort()


def docling_convert(file_path: str | Path, **kwargs):
    """docling 변환 한 번 → DoclingDocument(변환 프로세스에 맡긴다). PDF·DOCX·MD·XLSX가 공유한다.

    실패 분류(outbox.is_transient): TimeoutError·CaptionFailedError·LlmReadFailedError는 일시,
    PictureLimitError·불완전 변환 RuntimeError는 결정적.
    """
    return process.convert(file_path, kwargs)
