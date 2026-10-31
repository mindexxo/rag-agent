"""도메인 예외 → HTTP 변환 — rag/는 HTTP를 모르므로 라우터가 상태코드·문구로 바꾼다."""

from collections.abc import Iterator
from contextlib import contextmanager

from fastapi import HTTPException

from rag import document_admin as admin
from rag import faq as faq_svc
from rag import folders as folder_svc


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


@contextmanager
def document_http_errors() -> Iterator[None]:
    """rag/document_admin.py·rag/folders.py 도메인 예외 → HTTP. 문서·폴더 라우터가 공유한다."""
    try:
        yield
    except admin.DocumentNotFound as e:
        raise HTTPException(status_code=404, detail='document not found') from e
    except (admin.DocumentsNotFound, admin.DocumentRejected) as e:
        raise HTTPException(status_code=404 if isinstance(e, admin.DocumentsNotFound) else 400, detail=str(e)) from e
    except folder_svc.FolderNotFound as e:
        raise HTTPException(status_code=404, detail='folder not found') from e
    except folder_svc.FolderConflict as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
