"""도메인 예외 → HTTP 변환 — rag/는 HTTP를 모르므로 라우터가 상태코드·문구로 바꾼다."""

from collections.abc import Iterator
from contextlib import contextmanager

from fastapi import HTTPException

from rag import document_admin as admin
from rag import faq as faq_svc
from rag import folders as folder_svc
from rag.turn.conversation import ConversationNotFound


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


@contextmanager
def conversation_http_errors() -> Iterator[None]:
    """소유 대화 조회 실패 → 404. 없거나 남의 것이거나 삭제된 대화를 구분하지 않는다(존재 여부 비노출)."""
    try:
        yield
    except ConversationNotFound as e:
        raise HTTPException(status_code=404, detail='대화를 찾을 수 없습니다.') from e
