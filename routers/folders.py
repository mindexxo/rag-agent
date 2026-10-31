"""폴더 라우터 (1단 그룹, 검색 참조 제어).

폴더 CRUD + 폴더 단위 참조 on/off. 문서의 폴더 소속 변경은 documents 라우터의 PATCH가 담당.
"""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_session
from rag import folders as folder_svc
from rag.models import Folder
from routers.deps import get_tenant_id
from routers.errors import document_http_errors
from schemas.kms import FolderCreateRequest, FolderInfo, FolderUpdateRequest

router = APIRouter(prefix='/kms')


def _to_info(folder: Folder) -> FolderInfo:
    return FolderInfo(
        id=folder.id, name=folder.name, description=folder.description, is_searchable=folder.is_searchable
    )


@router.get('/folders', response_model=list[FolderInfo])
async def list_folders(
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_session),
) -> list[FolderInfo]:
    return [_to_info(f) for f in await folder_svc.list_folders(session, tenant_id)]


@router.post('/folders', response_model=FolderInfo)
async def create_folder(
    request: FolderCreateRequest,
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_session),
) -> FolderInfo:
    with document_http_errors():
        return _to_info(await folder_svc.create_folder(session, tenant_id, request))


@router.patch('/folders/{folder_id}', response_model=FolderInfo)
async def update_folder(
    folder_id: int,
    request: FolderUpdateRequest,
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_session),
) -> FolderInfo:
    with document_http_errors():
        folder = await folder_svc.require_folder(session, tenant_id, folder_id)
        return _to_info(await folder_svc.update_folder(session, folder, request))


@router.delete('/folders/{folder_id}', status_code=204)
async def delete_folder(
    folder_id: int,
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_session),
) -> None:
    """폴더 삭제. 내부에 문서가 있으면 삭제 불가(409) — 검색제외 폴더를 지웠을 때 문서가
    미분류로 풀려 조용히 검색에 복귀하는 사고 방지(P1-17). 문서를 옮기거나 삭제한 뒤 폴더를 지운다."""
    with document_http_errors():
        folder = await folder_svc.require_folder(session, tenant_id, folder_id)
        await folder_svc.delete_folder(session, folder)
