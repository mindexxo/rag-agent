"""폴더 관리 — 1단 그룹과 폴더 단위 참조 on/off. 이름·설명·토글이 바뀌면 소속 문서의 캐시를 버리고 색인 메타를 fan-out한다.

조회 자체(get_folder)는 rag/documents.py가 정의점이다 — 업로드도 같은 함수로 남의 테넌트 폴더를 막는다.
"""

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from rag import outbox
from rag.documents import get_folder
from rag.models import Document, Folder
from rag.turn import cache
from schemas.kms import FolderCreateRequest, FolderUpdateRequest


class FolderNotFound(Exception):
    """내 테넌트에 없는 폴더."""


class FolderConflict(Exception):
    """같은 이름 폴더·문서가 남은 폴더 삭제. `str(e)`가 그대로 사용자에게 나간다."""


async def require_folder(session: AsyncSession, tenant_id: str, folder_id: int) -> Folder:
    folder = await get_folder(session, tenant_id, folder_id)
    if folder is None:
        raise FolderNotFound
    return folder


async def folder_is_on(session: AsyncSession, tenant_id: str, folder_id: int | None) -> bool:
    """폴더의 참조 여부. 미분류(None)는 항상 on — 검색기(retriever)의 판정과 같은 규칙."""
    if folder_id is None:
        return True
    return bool(
        (
            await session.execute(
                select(Folder.is_searchable).where(Folder.tenant_id == tenant_id).where(Folder.id == folder_id)
            )
        ).scalar()
    )


async def _folder_doc_ids(session: AsyncSession, tenant_id: str, folder_id: int) -> list[int]:
    return list(
        (
            await session.execute(
                select(Document.id).where(Document.tenant_id == tenant_id).where(Document.folder_id == folder_id)
            )
        )
        .scalars()
        .all()
    )


async def _invalidate_folder_docs(session: AsyncSession, tenant_id: str, folder_id: int) -> None:
    """폴더 소속 문서들을 근거로 만든 답변 캐시 무효화.

    폴더 설정(참조 on/off·설명)이 바뀌면 같은 질문의 검색 결과·순위가 달라지므로,
    옛 설정으로 만든 캐시 답변은 버려야 한다.
    """
    for doc_id in await _folder_doc_ids(session, tenant_id, folder_id):
        await cache.invalidate_source(session, tenant_id, doc_id)


async def list_folders(session: AsyncSession, tenant_id: str) -> list[Folder]:
    return list(
        (await session.execute(select(Folder).where(Folder.tenant_id == tenant_id).order_by(Folder.name)))
        .scalars()
        .all()
    )


async def create_folder(session: AsyncSession, tenant_id: str, request: FolderCreateRequest) -> Folder:
    name = request.name  # 공백 제거·빈 값 거절은 스키마가 했다
    folder = Folder(tenant_id=tenant_id, name=name, description=(request.description or '').strip() or None)
    session.add(folder)
    try:
        await session.commit()
    except IntegrityError as e:
        await session.rollback()
        raise FolderConflict(f"'{name}' 폴더가 이미 있습니다.") from e
    return folder


async def update_folder(session: AsyncSession, folder: Folder, request: FolderUpdateRequest) -> Folder:
    tenant_id = folder.tenant_id
    if request.name is not None:
        folder.name = request.name
    if request.description is not None:
        # 설명은 리랭커 입력에 들어가 검색 순위를 바꾼다 → 옛 순위로 만든 답변 캐시를 버린다.
        # (설명이 실제로 달라졌을 때만 — 같은 값 재전송으로 캐시가 날아가지 않게)
        new_desc = request.description.strip() or None
        if new_desc != folder.description:
            await _invalidate_folder_docs(session, tenant_id, folder.id)
        folder.description = new_desc
    if request.is_searchable is not None:
        # 폴더 off 전환 시 소속 문서들을 근거로 만든 캐시 무효화 (문서 단위 off와 같은 이유)
        if folder.is_searchable and not request.is_searchable:
            await _invalidate_folder_docs(session, tenant_id, folder.id)
        folder.is_searchable = request.is_searchable
    # 폴더 이름·설명·검색토글은 소속 문서의 **모든 청크 문서**에 복사돼 있다 —
    # 같은 트랜잭션에 fan-out 갱신을 적재한다. 폴더가 fan-out의 최대 단위이고, 부분 갱신이라
    # 청크 수와 무관하게 _update_by_query 1~2회로 끝난다.
    doc_ids = await _folder_doc_ids(session, tenant_id, folder.id)
    if doc_ids:
        outbox.enqueue(session, tenant_id, outbox.META_DOCUMENTS, document_ids=doc_ids)
    try:
        await session.commit()
    except IntegrityError as e:
        await session.rollback()
        raise FolderConflict('같은 이름의 폴더가 이미 있습니다.') from e
    return folder


async def delete_folder(session: AsyncSession, folder: Folder) -> None:
    """내부에 (삭제 안 된) 문서가 있으면 FolderConflict — 검색제외 폴더를 지웠을 때 문서가 미분류로 풀려
    조용히 검색에 복귀하는 사고 방지. 문서를 옮기거나 삭제한 뒤 폴더를 지운다."""
    doc_count = (
        await session.execute(
            select(func.count())
            .select_from(Document)
            .where(Document.tenant_id == folder.tenant_id)
            .where(Document.folder_id == folder.id)
            .where(Document.status != 'deleted')  # 소프트삭제된 문서는 카운트 제외
        )
    ).scalar()
    if doc_count:
        raise FolderConflict(
            f'폴더에 문서 {doc_count}개가 있어 삭제할 수 없습니다. 문서를 옮기거나 삭제한 뒤 다시 시도해 주세요.'
        )
    await session.delete(folder)
    await session.commit()
