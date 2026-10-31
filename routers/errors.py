"""도메인 예외 → HTTP 변환 — rag/는 HTTP를 모르므로 라우터가 상태코드·문구로 바꾼다."""

from collections.abc import Iterator
from contextlib import contextmanager

from fastapi import HTTPException

from rag import faq as faq_svc


@contextmanager
def faq_http_errors() -> Iterator[None]:
    """rag/faq.py 도메인 예외 → HTTP. FAQ·카테고리 라우터가 공유한다."""
    try:
        yield
    except faq_svc.FaqNotFound as e:
        raise HTTPException(status_code=404, detail='FAQ not found') from e
    except faq_svc.FaqCategoryNotFound as e:
        raise HTTPException(status_code=404, detail='category not found') from e
    except faq_svc.FaqConflict as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
