"""문서 업로드 서비스
업로드 파일을 dedupe/버전 정책(supersede)에 따라 처리한다.
"""
import asyncio
import mimetypes
import time
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from database import AsyncSessionLocal
from rag import cache, os_index, outbox
from rag.chunking import chunk_file, pdf_image_area_ratio
from rag.embeddings import embed_texts
from rag.index_text import build_index_text
from rag.metrics import INDEX_DURATION_SECONDS, INDEX_TOTAL, ext_label
from rag.models import Document, Folder

# 이미지 면적이 이 비율을 넘으면 인제스션에서 경고를 남긴다 (#137 결함 4).
# 실측(실문서 21건): 도표가 이미지인 3건이 21.1% / 4.4% / 0.7%, 나머지 18건 0%.
# 0.7%짜리도 수당 산정식 하나가 통째로 사라졌으므로 임계를 낮게 잡는다.
IMAGE_WARN_RATIO = 0.5

_MIME_OVERRIDES = {
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}


def _detect_mime(blob_path: Path) -> str:
    """확장자로 mime 추론 — 업로드 시점에 handle_upload가 부른다.

    `mimetypes`가 docx를 못 알아보는 환경이 있어 override를 먼저 본다.
    라우터의 SUPPORTED_SUFFIXES 5종이 전부 여기서 판별돼야 업로드가 500 없이 끝난다
    (tests/test_small_utils.py가 그 조건을 고정한다).
    """
    suffix = blob_path.suffix.lower()
    if suffix in _MIME_OVERRIDES:
        return _MIME_OVERRIDES[suffix]
    mime, _ = mimetypes.guess_type(str(blob_path))
    if not mime:
        raise ValueError(f"mime 추론 실패: {blob_path}")
    return mime


async def index_pending_document(document_id: int, *, outbox_row_id: int | None = None) -> None:
    """INDEX_DOCUMENT 핸들러 — 인제스션 전체 (#139 outbox). 워커 drain이 부른다.

    **실패는 예외로 올린다.** 재시도 횟수·failed 판정은 rag/outbox.drain이 한다(MAX_ATTEMPTS
    초과 시 문서도 failed). 이 함수 안에는 삼키는 곳이 없다.

    커밋은 **마지막에 한 번**이다. 그 커밋에 ready·is_active·supersede·캐시 무효화·outbox done이
    함께 들어간다 — 그래서 `ready ≡ 색인됨`이 커밋 단위로 성립한다. 어느 단계에서 죽어도
    PG는 pending 그대로, 대기열 행은 pending 그대로여서 다음 회차가 처음부터 다시 한다
    (엔진에 반쯤 들어간 청크는 같은 _id로 덮이고, 구버전 삭제는 이미 없으면 0건 — 멱등).

    세션 셋으로 나눈 이유는 그대로다: 무거운 파싱·임베딩 동안 DB 커넥션을 물지 않는다.
      ① 짧은 읽기   처리 대상 확인 + 파싱·색인에 필요한 값만 (커넥션 즉시 반납)
      ② DB 없이     파싱·청킹·임베딩 → 엔진 색인 + 구버전 엔진 삭제
      ③ 짧은 쓰기   유일한 커밋
    """
    # ── ① 짧은 읽기 ──
    async with AsyncSessionLocal() as session:
        row = (await session.execute(
            select(Document, Folder.name, Folder.description, Folder.is_searchable)
            .outerjoin(Folder, Document.folder_id == Folder.id)
            .where(Document.id == document_id)
        )).first()
        if row is None or row[0].status != 'pending':      # 이미 처리됨·삭제됨 — 할 일 없음
            if outbox_row_id is not None:
                await outbox.mark_done(session, outbox_row_id)
                await session.commit()
            return
        doc, folder_name, folder_desc, folder_searchable = row
        tenant_id, filename, version = doc.tenant_id, doc.filename, doc.version
        blob_path, description = doc.blob_path, doc.description or ''
        folder_id, doc_searchable = doc.folder_id, doc.is_searchable
        # 같은 filename의 active 구버전 — 엔진에서 지우고 PG에서 내릴 대상. 여기서 한 번 읽어
        # ②·③이 같은 집합을 쓴다. 이 집합이 ③까지 유효한 근거는 **단일 워커의 순차 처리**다
        # (drain이 행 하나를 끝까지 처리한 뒤 다음 행으로 — 같은 파일명의 다음 버전 INDEX 행은
        # 그 뒤에 처리된다). 라우터의 expect_version 검사는 FE가 보낼 때만 도는 선택적 방어라
        # 여기의 근거로 삼지 않는다.
        old_active_ids = list((await session.execute(
            select(Document.id)
            .where(Document.tenant_id == tenant_id)
            .where(Document.filename == filename)
            .where(Document.is_active.is_(True))
            .where(Document.id != document_id)
        )).scalars().all())

    # ── ② 무거운 계산 — DB 세션 없음 ──
    # 단계별 소요를 지표로 남긴다 (#151). 워커에는 이것 말고 "한 건이 얼마나 걸리나"를 볼 방법이
    # 없다 — 로그는 남지 않고, drain의 cron 소요는 배치 전체라 건당으로 나눌 수 없다.
    # 성공한 건만 기록된다: 실패는 kms_search_index_sync_total{result="error"}가 이미 센다.
    _ext = ext_label(filename)
    _t_start = time.monotonic()
    # 청킹은 동기 CPU 작업(pdfplumber·python-docx·openpyxl)이라 스레드로 보낸다 — 이벤트
    # 루프에서 돌리면 PDF 하나에 100~200ms(실문서는 초 단위) 동안 워커가 멈춘다.
    # to_thread(stdlib)를 쓴다 — 이 모듈은 라우터도 import하므로 starlette를 들이지 않는다.
    chunks = await asyncio.to_thread(chunk_file, blob_path, description=description)
    image_ratio = await asyncio.to_thread(pdf_image_area_ratio, blob_path)   # #137 결함 4 경고용
    if not chunks:      # 빈 파일·텍스트레이어 없는 PDF → ready 승격 대신 실패 (유령 ready 방지)
        raise ValueError('추출된 텍스트가 없습니다 (빈 파일이거나 파싱 결과가 비어 있음)')
    # 네 단계가 같은 분모(성공한 문서)를 갖도록 실패 분기 뒤에서 기록한다 — 단계별 p95를
    # 나란히 읽으려면 분모가 같아야 한다.
    INDEX_DURATION_SECONDS.labels(stage='parse', ext=_ext).observe(time.monotonic() - _t_start)
    _t_parse_done = time.monotonic()
    # 임베딩 입력에만 '파일명 > 헤딩' 컨텍스트를 얹는다 (저장되는 chunk.text는 원문 유지).
    # 리랭커·어휘 채널·엔진 색인이 같은 조립을 쓴다 — 세 소비자가 같은 형태를 본다.
    index_texts = [build_index_text(c.text, filename, c.heading_path) for c in chunks]
    embeddings = await embed_texts(index_texts)
    INDEX_DURATION_SECONDS.labels(stage='embed', ext=_ext).observe(time.monotonic() - _t_parse_done)
    _t_embed_done = time.monotonic()

    # searchable은 PG의 **현재** 상태(pending·inactive)가 아니라 **아래 ③이 만들 상태**
    # (ready·active)로 계산한다 — 켜진 채로 바로 넣어 "켜기" 단계를 없애기 위함이다.
    # 문서 검색토글·폴더 토글은 PG 값을 그대로 쓴다.
    indexed_searchable = os_index.effective_searchable(
        is_faq=False, doc_is_active=True, doc_status='ready',
        doc_is_searchable=doc_searchable, folder_is_searchable=folder_searchable)
    await os_index.index_parsed_document(
        document_id=document_id, tenant_id=tenant_id, filename=filename, version=version,
        folder_id=folder_id, folder_name=folder_name, folder_description=folder_desc,
        searchable=indexed_searchable, chunks=chunks, embeddings=embeddings)
    # 신버전이 이미 켜져 있으니 구버전을 지워도 빈 창이 없다.
    if old_active_ids:
        await os_index.drop_documents_now(old_active_ids)
    INDEX_DURATION_SECONDS.labels(stage='index', ext=_ext).observe(time.monotonic() - _t_embed_done)

    # ── ③ 유일한 커밋 ──
    async with AsyncSessionLocal() as session:
        doc = await session.get(Document, document_id)
        if doc is None or doc.status != 'pending':
            # ②가 도는 사이 삭제됐다. 엔진에 넣은 청크는 그 삭제가 만든 DROP 행이 지운다 —
            # **다음 회차**다: drain은 pending 행을 회차 시작 시 한 번의 SELECT로 고정하므로
            # ② 도중 생긴 행은 이번 배치에 없다. 노출 창 ≤ 1분(cron) — 제품 결정 범위 안.
            if outbox_row_id is not None:
                await outbox.mark_done(session, outbox_row_id)
                await session.commit()
            return

        # 청크 행은 PG에 쓰지 않는다 — 검색·본문·메타 전부 엔진이 든다(#139). 이 커밋은
        # 문서 상태(ready·active·supersede)와 대기열 행만 확정한다.
        # supersede — ①에서 읽은 같은 집합. 옛 active off를 먼저 반영해 유니크 위반을 피한다.
        if old_active_ids:
            await session.execute(
                update(Document).where(Document.id.in_(old_active_ids))
                .values(is_active=False, status='deleted'))
            await session.flush()

        doc.status = 'ready'
        doc.is_active = True
        doc.char_count = sum(len(c.text) for c in chunks)
        # 도표가 이미지로 렌더된 문서는 그 내용이 색인되지 않는데도 ready가 된다
        # (#137 결함 4). 근거가 비어 있다는 사실을 사용자가 알 수 있게 남긴다 —
        # 검색·인용은 정상 동작하므로 상태는 ready 그대로 둔다.
        if image_ratio >= IMAGE_WARN_RATIO:
            doc.status_reason = (
                f'이미지가 쪽 면적의 {image_ratio:.0f}%를 차지합니다. '
                f'이미지로 그려진 표·흐름도는 텍스트를 추출할 수 없어 검색되지 않습니다.'
            )[:500]
        doc.indexed_at = datetime.now(timezone.utc)
        for old_id in old_active_ids:
            await cache.invalidate_source(session, tenant_id, old_id)

        # ②가 도는 사이(초~분) 문서 검색토글·폴더 토글·폴더 이동이 있었을 수 있다. 그 META 행은
        # 이미 색인된 청크만 갱신하므로(이 문서는 그때 엔진에 없었다) 여기서 ① 시점 값으로 넣은
        # 것이 낡은 채 남는다 — reconcile은 id 집합만 보고 메타 드리프트는 못 잡는다(리뷰 지적).
        # 최신 PG 상태로 다시 계산해 어긋나면 META 행을 **이 커밋에** 얹어 다음 회차가 맞춘다.
        f_on = None
        if doc.folder_id is not None:
            f_on = (await session.execute(
                select(Folder.is_searchable).where(Folder.id == doc.folder_id))).scalar()
        now_searchable = os_index.effective_searchable(
            is_faq=False, doc_is_active=True, doc_status='ready',
            doc_is_searchable=doc.is_searchable, folder_is_searchable=f_on)
        if now_searchable != indexed_searchable or doc.folder_id != folder_id:
            outbox.enqueue(session, tenant_id, outbox.META_DOCUMENTS, document_ids=[document_id])

        if outbox_row_id is not None:
            await outbox.mark_done(session, outbox_row_id)   # ready와 같은 커밋 — 정의점 rag/outbox.py
        await session.commit()
    # total은 ③ 커밋까지 포함한다 — 사용자가 체감하는 "업로드 후 검색될 때까지"에 가장 가깝다
    # (여기에 cron 대기 최대 1분이 더 붙는다, rag/worker.py drain 주석).
    INDEX_DURATION_SECONDS.labels(stage='total', ext=_ext).observe(time.monotonic() - _t_start)
    INDEX_TOTAL.labels(ext=_ext, result='ok').inc()   # 실패 쪽은 outbox.drain이 센다(재시도/확정 구분)


async def handle_upload(
        session: AsyncSession,
        tenant_id: str,
        filename: str,
        blob_path: Path,
        description: str | None = None,
        uploaded_by: str | None = None,
        folder_id: int | None = None,
        folder_given: bool = False,
) -> Document:
    """업로드 시점 처리: pending row + 색인 대기열 행을 **같은 트랜잭션**에 등록한다 (#139).
    실제 청킹/임베딩/색인/supersede는 워커 drain이 index_pending_document로 수행한다.
    description은 표 설명(xlsx 검색 보강) — 워커가 청킹 시 병합한다.
    uploaded_by는 업로더 식별자(X-User-Id) — 없으면 NULL.
    folder_id/folder_given은 업로드 시 폴더 지정 (#165) — folder_given=False(미전송)면 계승,
    True면 folder_id를 그대로 쓴다(None이면 미분류로 떼는 것). 값만으로는 둘을 못 가른다.
    mime은 blob_path에서 직접 구한다 — 호출부가 계산해 넘길 이유가 없다.

    문서 식별은 **filename 완전 일치** 하나뿐 (2026-08-05 정책 확정).
    내용 해시(sha) dedupe는 제거 — 같은 이름이면 내용이 같아도 새 version이 된다.
    "같은 이름이면 물어보고, 확인하면 대체"라는 단일 규칙을 유지하기 위함
    (내용 동일 여부로 확인 창을 띄울지 말지 분기하면 규칙이 둘이 된다).
    """
    # 같은 filename의 모든 버전 조회 (다음 version 계산 + 설정 계승용)
    docs = (await session.execute(
        select(Document)
        .where(Document.tenant_id == tenant_id)
        .where(Document.filename == filename)
    )).scalars().all()

    # pending 버전 row만 insert. is_active=False(ready 전엔 검색 제외).
    # 폴더 소속·참조 on/off(F2)는 직전 버전에서 계승 — 개정판 업로드로 설정이 풀리지 않게.
    next_version = max((d.version for d in docs), default=0) + 1
    prev = max(docs, key=lambda d: d.version) if docs else None
    doc = Document(
        tenant_id=tenant_id,
        filename=filename,
        mime=_detect_mime(blob_path),
        blob_path=str(blob_path),
        version=next_version,
        is_active=False,
        status="pending",
        # 미전송이면 계승, 보냈으면 그 값(None이면 미분류) — #165. 호출부가 보낸 여부를
        # 따로 넘겨주는 이유는, folder_id=None이 "미분류로 보내라"와 "안 보냈다" 둘 다이기 때문이다.
        folder_id=(folder_id if folder_given else (prev.folder_id if prev else None)),
        is_searchable=prev.is_searchable if prev else True,
        description=description if description is not None else (prev.description if prev else None),
        # 등록자만은 **계승하지 않는다** (#164). 위 세 값은 "문서에 건 설정"이라 개정판에도
        # 이어져야 하지만, 등록자는 uploaded_at과 짝을 이루는 "이 버전을 올린 사람"이다 —
        # 계승하면 화면에서 최신 개정을 누가 했는지 알 수 없게 된다.
        uploaded_by=uploaded_by,
    )
    session.add(doc)
    await session.flush()  # doc.id 확보 (대기열 payload에 필요)
    # 트랜잭셔널 outbox — 호출부(라우터)가 커밋하면 문서와 대기열 행이 함께 확정된다.
    # arq 잡 등록은 없다: Redis 순단으로 잡이 유실돼 pending이 고착하던 실패 모드(P1-4)가
    # 이 한 줄로 사라진다. 사유·처리 규약은 rag/outbox.py.
    outbox.enqueue(session, tenant_id, outbox.INDEX_DOCUMENT, document_id=doc.id)
    return doc


