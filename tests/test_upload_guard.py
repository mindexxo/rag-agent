"""업로드 크기 코스 가드 단위 테스트 — documents._reject_if_oversized.

Content-Length 기반 선제 차단 — 정확 경계는 read 후 len()이 담당하므로 여기선 여유분(8192) 포함 판정.
"""
import pytest
from fastapi import HTTPException
from starlette.requests import Request

from routers.documents import _reject_if_oversized

LIMIT = 10 * 1024 * 1024


def _request(content_length: str | None) -> Request:
    headers = []
    if content_length is not None:
        headers.append((b'content-length', content_length.encode()))
    return Request({'type': 'http', 'headers': headers})


def test_한계_초과는_413():
    with pytest.raises(HTTPException) as exc:
        _reject_if_oversized(_request(str(LIMIT + 8193)), LIMIT)
    assert exc.value.status_code == 413


def test_여유분_이내는_통과():
    _reject_if_oversized(_request(str(LIMIT + 8192)), LIMIT)  # 예외 없어야 함


def test_헤더_없으면_통과():
    _reject_if_oversized(_request(None), LIMIT)


def test_비숫자_헤더는_통과():
    _reject_if_oversized(_request('abc'), LIMIT)  # 정확 검사는 read 후 담당
