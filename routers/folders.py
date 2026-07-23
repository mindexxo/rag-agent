"""폴더 라우터 (F2 — 1단 그룹, 검색 참조 제어).

폴더 CRUD + 폴더 단위 참조 on/off. 문서의 폴더 소속 변경은 documents 라우터의 PATCH가 담당.
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select, func
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_session
from rag.cache import AnswerCache
from rag.models import Document, Folder
from routers.kms import get_tenant_id
from schemas.kms import FolderInfo, FolderCreateRequest, FolderUpdateRequest

router = APIRouter(prefix='/kms')


async def _get_folder(session: AsyncSession, tenant_id: str, folder_id: int) -> Folder:
    folder = (await session.execute(
        select(Folder)
        .where(Folder.tenant_id == tenant_id)   # 격리 — WHERE 절 명시
        .where(Folder.id == folder_id)
    )).scalars().first()
    if folder is None:
        raise HTTPException(status_code=404, detail='folder not found')
    return folder


@router.get('/folders', response_model=list[FolderInfo])
async def list_folders(
        tenant_id: str = Depends(get_tenant_id),
        session: AsyncSession = Depends(get_session),
):
    folders = (await session.execute(
        select(Folder)
        .where(Folder.tenant_id == tenant_id)
        .order_by(Folder.name)
    )).scalars().all()
    return [FolderInfo(id=f.id, name=f.name, is_searchable=f.is_searchable) for f in folders]


@router.post('/folders', response_model=FolderInfo)
async def create_folder(
        request: FolderCreateRequest,
        tenant_id: str = Depends(get_tenant_id),
        session: AsyncSession = Depends(get_session),
):
    name = request.name.strip()
    if not name:
        raise HTTPException(status_code=422, detail='폴더 이름이 비어 있습니다.')
    folder = Folder(tenant_id=tenant_id, name=name)
    session.add(folder)
    try:
        await session.commit()
    except IntegrityError:
        await session.rollback()
        raise HTTPException(status_code=409, detail=f"'{name}' 폴더가 이미 있습니다.")
    return FolderInfo(id=folder.id, name=folder.name, is_searchable=folder.is_searchable)


@router.patch('/folders/{folder_id}', response_model=FolderInfo)
async def update_folder(
        folder_id: int,
        request: FolderUpdateRequest,
        tenant_id: str = Depends(get_tenant_id),
        session: AsyncSession = Depends(get_session),
):
    folder = await _get_folder(session, tenant_id, folder_id)
    if request.name is not None:
        name = request.name.strip()
        if not name:
            raise HTTPException(status_code=422, detail='폴더 이름이 비어 있습니다.')
        folder.name = name
    if request.is_searchable is not None:
        # 폴더 off 전환 시 소속 문서들을 근거로 만든 캐시 무효화 (문서 단위 off와 같은 이유)
        if folder.is_searchable and not request.is_searchable:
            doc_ids = (await session.execute(
                select(Document.id)
                .where(Document.tenant_id == tenant_id)
                .where(Document.folder_id == folder.id)
            )).scalars().all()
            cache = AnswerCache()
            for doc_id in doc_ids:
                await cache.invalidate_document(session, tenant_id, doc_id)
        folder.is_searchable = request.is_searchable
    try:
        await session.commit()
    except IntegrityError:
        await session.rollback()
        raise HTTPException(status_code=409, detail='같은 이름의 폴더가 이미 있습니다.')
    return FolderInfo(id=folder.id, name=folder.name, is_searchable=folder.is_searchable)


@router.delete('/folders/{folder_id}', status_code=204)
async def delete_folder(
        folder_id: int,
        tenant_id: str = Depends(get_tenant_id),
        session: AsyncSession = Depends(get_session),
):
    """폴더 삭제. 내부에 문서가 있으면 삭제 불가(409) — 검색제외 폴더를 지웠을 때 문서가
    미분류로 풀려 조용히 검색에 복귀하는 사고 방지(P1-17). 문서를 옮기거나 삭제한 뒤 폴더를 지운다."""
    folder = await _get_folder(session, tenant_id, folder_id)
    doc_count = (await session.execute(
        select(func.count()).select_from(Document)
        .where(Document.tenant_id == tenant_id)
        .where(Document.folder_id == folder_id)
        .where(Document.status != 'deleted')       # 소프트삭제된 문서는 카운트 제외
    )).scalar()
    if doc_count:
        raise HTTPException(
            status_code=409,
            detail=f'폴더에 문서 {doc_count}개가 있어 삭제할 수 없습니다. 문서를 옮기거나 삭제한 뒤 다시 시도해 주세요.',
        )
    await session.delete(folder)
    await session.commit()
