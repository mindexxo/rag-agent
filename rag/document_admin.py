"""문서 관리 — 목록·이름 확인·단건 조회와 폴더 소속·참조 on/off·표 설명 변경, 일괄 변경·삭제의 대상 확정.

업로드·supersede·삭제 자체는 rag/documents.py(handle_upload·soft_delete_documents)가 정의점이다.
"""

from pathlib import Path
from typing import NamedTuple

from sqlalchemy import ColumnElement, func, select
from sqlalchemy import text as sql_text
from sqlalchemy.ext.asyncio import AsyncSession

from rag import outbox
from rag.chunking.xlsx import description_missing
from rag.folders import folder_is_on, require_folder
from rag.models import ALIVE_DOCUMENT_STATUSES, Document, Folder
from rag.turn import cache
from schemas.kms import DocumentBulkUpdateRequest, DocumentUpdateRequest
from text_norm import LIKE_ESCAPE_CHAR, like_pattern, normalize_filename


class DocumentNotFound(Exception):
    """내 테넌트에 없는(또는 삭제된) 문서."""


class DocumentsNotFound(Exception):
    """일괄 요청의 id 중 일부가 없다. `str(e)`가 그대로 사용자에게 나간다."""


class DocumentRejected(Exception):
    """규칙 위반(xlsx 표 설명 비우기 등). `str(e)`가 그대로 사용자에게 나간다."""


class DocumentFilter(NamedTuple):
    """목록 조건. q는 파일명 부분 일치(빈 값은 호출부가 None으로), folder_id 0은 미분류만, status는 위생 처리된 값."""

    q: str | None
    folder_id: int | None
    status: list[str] | None
    is_searchable: bool | None


class DocumentPage(NamedTuple):
    docs: list[Document]
    total: int
    ref_counts: dict[str, int]


def _list_where(tenant_id: str, f: DocumentFilter) -> ColumnElement[bool]:
    """목록 조회 조건. **count와 페이지 쿼리가 같은 것을 써야** has_more가 어긋나지 않는다.

    supersede된 구버전(deleted)은 항상 빠진다 — 죽은 행에 폴더/참조 컨트롤이 노출되는 혼란 방지.
    """
    where = (Document.tenant_id == tenant_id) & (Document.status != 'deleted')
    if f.q is not None:
        # 파일명은 경계에서 NFC로 저장된다. 검색어도 맞춰야 한다 — text_norm의 정책은
        # "타이핑 입력은 IME가 NFC를 내므로 위험군이 아니다"지만, 파일명 검색은 사용자가
        # **이름을 복사해 붙여넣는** 경로가 흔하고 macOS에서 복사한 이름은 NFD일 수 있다.
        where &= Document.filename.ilike(like_pattern(normalize_filename(f.q)), escape=LIKE_ESCAPE_CHAR)
    if f.folder_id is not None:
        # 0은 '미분류' 약속 — 폴더 id는 1부터라 유효한 값과 겹치지 않는다. 쿼리 파라미터에는
        # null이 없어서(multipart와 같은 제약) 값 하나로 표현했다.
        where &= Document.folder_id.is_(None) if f.folder_id == 0 else Document.folder_id == f.folder_id
    if f.status:
        where &= Document.status.in_(f.status)
    if f.is_searchable is not None:
        where &= Document.is_searchable == f.is_searchable
    return where


async def ref_counts(session: AsyncSession, tenant_id: str, filenames: list[str]) -> dict[str, int]:
    """파일명별 인용 횟수 — 저장 시 확정된 실인용 목록(messages.cited_docs)의 집계.

    sources(검색 후보 노출 수)가 아닌 **실인용**이다. filename 키라 버전 교체 후에도 카운트가 이어진다.
    **이번 페이지의 파일명으로만** 좁혀 부른다.
    """
    if not filenames:
        return {}
    rows = (
        await session.execute(
            sql_text("""
        SELECT d AS filename, count(*) AS cnt
        FROM messages, jsonb_array_elements_text(messages.cited_docs) AS d
        WHERE messages.tenant_id = :tenant_id
          AND messages.role = 'assistant'
          AND jsonb_typeof(messages.cited_docs) = 'array'
          AND d = ANY(:filenames)
        GROUP BY 1
    """),
            {"tenant_id": tenant_id, "filenames": filenames},
        )
    ).all()
    return {r.filename: r.cnt for r in rows}


async def list_documents(
    session: AsyncSession, tenant_id: str, f: DocumentFilter, limit: int, offset: int
) -> DocumentPage:
    """최신 업로드가 위로. limit·offset 위생과 status 검증은 호출부가 했다."""
    where = _list_where(tenant_id, f)  # 한 번만 만들어 count와 페이지 쿼리가 함께 쓴다
    total = (await session.execute(select(func.count(Document.id)).where(where))).scalar_one()
    docs = list(
        (
            await session.execute(
                select(Document)
                .where(where)
                .order_by(Document.uploaded_at.desc(), Document.id.desc())
                .offset(offset)
                .limit(limit)
            )
        )
        .scalars()
        .all()
    )
    return DocumentPage(docs, total, await ref_counts(session, tenant_id, list({d.filename for d in docs})))


async def find_alive(session: AsyncSession, tenant_id: str, filename: str) -> Document | None:
    """이 이름의 살아 있는 최신 버전 — exists API의 판정. 기준은 업로드의 버전 확인·handle_upload의 재사용 판정과
    **정확히 같아야** 한다(tenant + filename 완전 일치, ALIVE_DOCUMENT_STATUSES). 업로드와 같은 경계 정규화를 한다 —
    여기만 빠지면 NFD 이름으로 물어본 클라이언트가 exists=false를 받고 중복 문서를 만든다."""
    filename = normalize_filename(filename)
    return (
        (
            await session.execute(
                select(Document)
                .where(Document.tenant_id == tenant_id)  # 격리 — WHERE 절 명시
                .where(Document.filename == filename)
                .where(Document.status.in_(ALIVE_DOCUMENT_STATUSES))
                .order_by(Document.version.desc())
                .limit(1)
            )
        )
        .scalars()
        .first()
    )


async def get_document(
    session: AsyncSession, tenant_id: str, document_id: int, *, include_deleted: bool = False
) -> Document:
    """단건 조회. 삭제된 문서는 없는 문서다 — 다운로드(인용된 옛 버전 원본)만 include_deleted로 연다."""
    stmt = select(Document).where(Document.tenant_id == tenant_id).where(Document.id == document_id)
    if not include_deleted:
        stmt = stmt.where(Document.status != 'deleted')
    doc = (await session.execute(stmt)).scalars().first()
    if doc is None:
        raise DocumentNotFound
    return doc


def effective_ref(doc_is_searchable: bool, folder_on: bool) -> bool:
    """실효 참조 = 문서 스위치 on AND 폴더 on(미분류는 on으로 친다). **문서 관리 판정의 정의점**.

    ⚠ rag/search/os_index.py의 `effective_searchable`과 **다른 판정**이다. 이름이 닮았지만 그쪽은
    색인용이라 is_faq·is_active·status=='ready'까지 본다. 여기는 사용자가 돌린 스위치 둘만 본다
    — 캐시 무효화 판정("근거가 검색에서 빠지는가")에 쓰이기 때문이다. 합치면 동작이 바뀐다.
    """
    return bool(doc_is_searchable) and bool(folder_on)


async def _effective_ref_now(session: AsyncSession, doc: Document) -> bool:
    """이 문서의 지금 실효 참조 — 폴더 상태를 문서마다 조회하는 단건 경로(문서 스위치가 꺼져 있으면 조회하지 않는다)."""
    if not doc.is_searchable:
        return False
    return effective_ref(doc.is_searchable, await folder_is_on(session, doc.tenant_id, doc.folder_id))


def require_all_found(requested: list[int], found: set[int]) -> None:
    """요청한 문서 id가 전부 조회됐는지. 하나라도 빠지면 **아무것도 바꾸지 않고** DocumentsNotFound.

    화면에서 체크한 목록과 결과가 어긋나면 무엇이 반영됐는지 되짚을 방법이 없어서, 부분 성공 대신 전부 거절한다.
    남의 테넌트 id인지 없는 id인지 **구분해 알려주지 않는다** — 요청자가 보낸 값을 되돌려줄 뿐이다.
    """
    if missing := [i for i in requested if i not in found]:
        raise DocumentsNotFound(f'문서를 찾을 수 없습니다: {missing[:10]}')


async def _bulk_targets(session: AsyncSession, tenant_id: str, ids: list[int]) -> list[Document]:
    # 테넌트 스코프 조회로 id를 먼저 확정한다 — 아래 outbox가 부르는 os_index.sync_meta_documents_now는
    # document_ids에 tenant 필터를 걸지 않고 호출부의 사전 스코프를 전제하기 때문이다.
    docs = list(
        (
            await session.execute(
                select(Document)
                .where(Document.tenant_id == tenant_id)
                .where(Document.status != 'deleted')
                .where(Document.id.in_(ids))
                .order_by(Document.id)
            )
        )
        .scalars()
        .all()
    )
    require_all_found(ids, {d.id for d in docs})
    return docs


async def _folder_states(session: AsyncSession, tenant_id: str, folder_ids: set[int]) -> dict[int, bool]:
    if not folder_ids:
        return {}
    rows = (
        await session.execute(
            select(Folder.id, Folder.is_searchable)
            .where(Folder.tenant_id == tenant_id)
            .where(Folder.id.in_(folder_ids))
        )
    ).all()
    return {r.id: r.is_searchable for r in rows}


async def bulk_update_documents(
    session: AsyncSession, tenant_id: str, request: DocumentBulkUpdateRequest
) -> list[Document]:
    """문서 여러 건의 폴더 소속·참조 on/off를 한 번에 바꾼다. 단건을 N번 부른 것과 결과는 같지만 커밋도, 색인
    갱신 대기열 행도 한 번이다. 대상이 하나라도 어긋나면(없는 id·다른 테넌트·삭제됨) 아무것도 바꾸지 않는다."""
    changing_folder = 'folder_id' in request.model_fields_set  # null 전송(미분류)과 미전송 구분
    changing_searchable = request.is_searchable is not None
    if not request.document_ids or not (changing_folder or changing_searchable):
        return []  # 바꿀 것이 없는 요청은 조회도 하지 않는다
    docs = await _bulk_targets(session, tenant_id, list(dict.fromkeys(request.document_ids)))  # 중복 제거(순서 유지)

    # 폴더 검증은 1회면 된다(대상 전부가 같은 폴더로 간다). 변경 **전** 실효 참조를 따지려면 각 문서가 지금 속한
    # 폴더의 상태도 필요하다 — 문서마다 폴더가 다르므로 한 번에 모아 온다.
    folder_on: dict[int, bool] = {}
    if changing_folder and request.folder_id is not None:
        folder = await require_folder(session, tenant_id, request.folder_id)  # 남의 테넌트 폴더 차단
        folder_on[folder.id] = folder.is_searchable
    folder_on |= await _folder_states(
        session, tenant_id, {d.folder_id for d in docs if d.folder_id is not None} - folder_on.keys()
    )

    def _effective(doc: Document) -> bool:
        return effective_ref(doc.is_searchable, doc.folder_id is None or bool(folder_on.get(doc.folder_id)))

    # 캐시 무효화는 **문서별로** 판정한다. 문서마다 현재 폴더·스위치가 달라 on→off 전이도
    # 제각각이다 — 일괄로 판정하면 off된 문서를 근거로 만든 답변이 캐시로 계속 나간다.
    for doc in docs:
        before = _effective(doc)
        if changing_folder:
            doc.folder_id = request.folder_id
        if (
            request.is_searchable is not None
        ):  # changing_searchable과 같은 조건 — bool 플래그로는 mypy가 None을 못 걸러 직접 쓴다
            doc.is_searchable = request.is_searchable
        if before and not _effective(doc):
            await cache.invalidate_source(session, tenant_id, doc.id)

    # 색인의 비정규화 메타 갱신은 **한 행**에 담는다. 워커는 최종값이 같은 문서끼리
    # 묶어 갱신하므로(os_index.sync_meta_documents_now) 200건이어도 보통 질의 1~2회다.
    outbox.enqueue(session, tenant_id, outbox.META_DOCUMENTS, document_ids=[d.id for d in docs])
    await session.commit()
    return docs


def _validated_description(doc: Document, request: DocumentUpdateRequest) -> tuple[str | None, bool]:
    """(새 표 설명, 재색인 필요?). 검증만 하고 doc은 건드리지 않는다.

    표 설명은 2-way: 미전송(None)이면 안 건드리고, 보낸 값은 strip 후 빈 문자열이면 지움으로 본다(폴더 설명과 같은 방식).
    """
    if request.description is None:
        return doc.description, False
    new_description = request.description.strip() or None
    is_xlsx = Path(doc.filename).suffix.lower() == '.xlsx'
    if is_xlsx and description_missing(new_description):
        # 업로드에서 필수로 받아놓고 여기서 비울 수 있으면 규칙이 샌다.
        raise DocumentRejected('xlsx 문서는 표 설명을 비울 수 없습니다.')
    # 재색인은 **ready인 xlsx의 값이 실제로 바뀐** 경우만. pending이면 대기 중인 그 행이 처리될 때
    # 최신값을 읽고, failed는 되살리지 않는다 — 되살리기는 재업로드 경로의 역할이다. 같은 값
    # 재전송에 재임베딩을 돌릴 이유도 없고, 비-xlsx는 설명이 청크에 안 들어가 재색인해도 같다.
    return new_description, is_xlsx and new_description != doc.description and doc.status == 'ready'


async def update_document(session: AsyncSession, doc: Document, request: DocumentUpdateRequest) -> Document:
    """문서 속성 변경: 폴더 소속·참조 on/off·표 설명. 보낸 필드만 반영.

    **경로가 둘이다.** folder_id·is_searchable은 색인의 비정규화 메타만 바꾸므로 `META_DOCUMENTS` 부분 갱신으로
    끝난다. 반면 `description`은 xlsx 청크 **본문**에 병합되는 값이라(rag/chunking/xlsx.chunk_xlsx) 재파싱·재임베딩이
    따르는 전체 재색인이 필요하다. 둘은 **배타적으로** 적재한다(재색인이 돌면 index_pending_document가 폴더·토글
    최신값을 스스로 다시 읽으므로 META 행은 중복이다).
    """
    tenant_id = doc.tenant_id
    # ── 검증 전부 먼저, mutation은 그 뒤 — 거절로 빠질 때 doc이 반쯤 바뀐 채 남지 않게 ──
    new_description, reindex = _validated_description(doc, request)
    if 'folder_id' in request.model_fields_set and request.folder_id is not None:
        await require_folder(session, tenant_id, request.folder_id)  # 남의 테넌트 폴더 차단

    effective_before = await _effective_ref_now(session, doc)
    # folder_id는 "null 전송 = 미분류 이동"과 "미전송 = 변경 없음"을 구분해야 함
    if 'folder_id' in request.model_fields_set:
        doc.folder_id = request.folder_id
    if request.is_searchable is not None:
        doc.is_searchable = request.is_searchable
    if request.description is not None:
        doc.description = new_description

    # 실효 참조가 on→off로 바뀌는 모든 경로(문서 off, off 폴더로 이동)에서 캐시 무효화 —
    # off된 문서를 근거로 만든 답변이 exact 캐시로 계속 나가는 것 방지. off→on은 무효화할 캐시가 없음.
    if effective_before and not await _effective_ref_now(session, doc):
        await cache.invalidate_source(session, tenant_id, doc.id)

    if reindex:
        # 설명이 청크 본문을 바꾸므로 전체 재색인 — pending으로 되돌리고 INDEX_DOCUMENT를
        # 다시 태운다(rag/os_reconcile.py의 reconcile과 같은 전례). 재파싱·재임베딩·ready
        # 승격은 index_pending_document 한 곳이 맡는다 — 두 벌을 두지 않는다.
        # 버전은 올리지 않는다: 번호가 올라가는 사건은 **파일 재업로드**뿐이다.
        # 수용한 대가 둘 — ① index_parsed_document가 같은 document_id의 청크를 지우고
        # 다시 넣으므로 1초 미만 검색 공백이 생긴다. ② 재색인이 실패하면 outbox가 문서를 failed로 굳힌다 —
        # 설명만 고치려던 사용자가 문서 전체를 잃는 모양이지만, 재업로드로 되살릴 수 있다.
        doc.status = 'pending'
        outbox.enqueue(session, tenant_id, outbox.INDEX_DOCUMENT, document_id=doc.id)
        # 내용이 바뀌었으므로 visibility 전이와 **무관하게** 무효화한다 — 위 effective 조건은
        # "검색에서 빠지는가"만 보기 때문에 설명 변경을 못 잡는다.
        await cache.invalidate_source(session, tenant_id, doc.id)
    else:
        # 외부 색인의 비정규화 메타 갱신을 같은 트랜잭션에 적재 — 검색가능·폴더 값이
        # 청크 문서마다 복사돼 있어 fan-out이 필요하다. **재색인이 아니라 부분 갱신**이다
        # (재색인은 벡터 없는 구성에서 재임베딩을 부른다 — rag/search/os_index.py의 _update_meta).
        outbox.enqueue(session, tenant_id, outbox.META_DOCUMENTS, document_ids=[doc.id])
    await session.commit()
    return doc


async def bulk_delete_targets(session: AsyncSession, tenant_id: str, ids: list[int]) -> list[int]:
    """일괄 삭제 대상 확정. **이미 삭제된 문서도 통과시킨다**(status 필터 없음 — 멱등): 삭제는 "그 문서가 없는 상태"를
    만드는 것이라 이미 그 상태면 실패로 볼 이유가 없다. 테넌트 스코프 조회로 id를 먼저 확정한다 —
    os_index.drop_documents_now는 document_ids에 tenant 필터를 걸지 않는다."""
    unique = list(dict.fromkeys(ids))
    found = set(
        (
            await session.execute(
                select(Document.id).where(Document.tenant_id == tenant_id).where(Document.id.in_(unique))
            )
        )
        .scalars()
        .all()
    )
    require_all_found(unique, found)
    return unique
