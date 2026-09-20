"""문서 업로드 서비스
업로드 파일을 dedupe/버전 정책(supersede)에 따라 처리한다.
"""
import asyncio
import mimetypes
import time
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from database import AsyncSessionLocal
from rag import cache, os_index, outbox
from rag.chunking import chunk_file, count_picture_placeholders, pdf_image_area_ratio
from rag.embeddings import embed_texts
from rag.index_text import build_index_text
from rag.metrics import (
    INDEX_DURATION_SECONDS,
    INDEX_PICTURE_PLACEHOLDER_TOTAL,
    INDEX_TOTAL,
    ext_label,
)
from rag.models import ALIVE_DOCUMENT_STATUSES, Document, Folder

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
    (엔진에 반쯤 들어간 청크는 같은 _id로 덮인다 — 멱등. 커밋 전엔 엔진에서 아무것도 **지우지** 않는다).

    **파괴적인 엔진 변경은 커밋 뒤 outbox로만** (#186). 구버전 청크 삭제는 ②에서 직접 하지 않고
    ③ 커밋에 DROP_DOCUMENTS 행으로 얹는다 — ③이 실패하면 행도 롤백돼 구버전은 PG(ready·active)·엔진
    모두 그대로다. ②에서 지웠던 때는 ③ 실패가 "목록엔 정상인데 검색엔 없는" 구버전을 영구로 남겼다.
    대가는 성공 경로에서 구·신 버전 청크가 다음 회차(≤1분)까지 공존하는 것 — 삭제 문서가 최대 1분
    인용될 수 있다는 제품 결정(rag/outbox.py)과 같은 범위. 신버전 색인(②)은 비파괴·멱등 upsert라
    커밋 전에 해도 되고, 실패로 굳으면 outbox.drain의 failed 확정 DROP(#184)이 치운다.

    세션 셋으로 나눈 이유는 그대로다: 무거운 파싱·임베딩 동안 DB 커넥션을 물지 않는다.
      ① 짧은 읽기   처리 대상 확인 + 파싱·색인에 필요한 값만 (커넥션 즉시 반납)
      ② DB 없이     파싱·청킹·임베딩 → 엔진 색인 (삭제는 없다 — 위 원칙)
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
        # 같은 filename의 **낮은 version** 중 살아 있는 것 — ③에서 PG를 내리고 DROP 행을 남길 대상(supersede).
        # "active만"이 아니라 pending도 포함하고 version 조건을 거는 이유(#185): outbox 행은 등재 순으로
        # 처리된다는 보장이 없다 — 백오프로 v1 행이 미뤄진 사이 v2 행이 먼저 끝날 수 있다. 그때 v2가
        # pending v1을 ③에서 내려두면 v1의 차례가 와도 위 "pending 아님 → done" 분기로 빠진다.
        # version 조건은 그 반대 방향의 안전판이다 — 어떤 경로로든 구버전이 나중에 돌아도 신버전을
        # supersede하지 못한다(그 경우 ③의 active 유니크 uq_docs_one_active_per_name에 걸려 결정적 실패 →
        # 구버전만 failed). 라우터의 expect_version 검사는 FE가 보낼 때만 도는 선택적 방어라 근거로 삼지 않는다.
        old_ids = list((await session.execute(
            select(Document.id)
            .where(Document.tenant_id == tenant_id)
            .where(Document.filename == filename)
            .where(Document.version < version)
            .where(Document.status.in_(ALIVE_DOCUMENT_STATUSES))
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
    # 캡션이 안 붙은 그림 수(#162). docling이 VLM 실패를 예외 없이 삼키므로 이 값이 유일한
    # 관측 창이다 — 캡션이 켜져 있는데 늘면 VLM이 죽었거나 느린 것이다.
    if (placeholders := count_picture_placeholders(chunks)):
        INDEX_PICTURE_PLACEHOLDER_TOTAL.labels(ext=_ext).inc(placeholders)
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
    # 구버전 청크는 여기서 지우지 않는다 — ③의 DROP 행이 다음 회차에 지운다(함수 docstring, #186).
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
        # 엔진 삭제는 같은 커밋의 DROP 행으로(#186) — deleted로 바뀌는 상태와 함께 확정되거나 함께 롤백된다.
        # 처리 시 outbox.drain의 DROP 가드는 이 문서들이 deleted라 통과시킨다. pending 구버전의 청크
        # (그쪽 ②가 썼다가 ③에서 실패해 남은 것)도 같은 집합이라 함께 지워진다.
        if old_ids:
            await session.execute(
                update(Document).where(Document.id.in_(old_ids))
                .values(is_active=False, status='deleted'))
            outbox.enqueue(session, tenant_id, outbox.DROP_DOCUMENTS, document_ids=old_ids)
            await session.flush()

        doc.status = 'ready'
        doc.is_active = True
        doc.char_count = sum(len(c.text) for c in chunks)
        # 도표가 이미지로 렌더된 문서를 사용자에게 알린다(#137 결함 4). 검색·인용은 정상
        # 동작하므로 상태는 ready 그대로 둔다. 문구는 두 번 바뀌었다 — #162 이후 그림은 VLM 캡션으로,
        # #168 이후 격자 표 이미지는 한국어 OCR로 읽히므로 "반영되지 않습니다"는 더 이상 사실이 아니다.
        # 남는 위험은 생성·인식 오류라 그것만 알린다.
        if image_ratio >= IMAGE_WARN_RATIO:
            doc.status_reason = (
                f'이미지가 쪽 면적의 {image_ratio:.0f}%를 차지합니다. '
                f'그림은 AI가 설명을 만들고 표 이미지는 글자를 인식해 검색에 반영합니다. '
                f'생성된 설명이나 인식된 값이 부정확할 수 있어 원문 확인이 필요합니다.'
            )[:500]
        doc.indexed_at = datetime.now(timezone.utc)
        for old_id in old_ids:
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


async def soft_delete_documents(
        session: AsyncSession, tenant_id: str, document_ids: list[int]
) -> list[int]:
    """문서 소프트 삭제 (F5) — 단건·일괄 삭제가 공유하는 하나의 경로 (#174).

    같은 filename의 **전 버전**을 함께 내린다. 한 버전만 내리면 그 이름의 옛 버전이 검색에
    되살아난 것처럼 보인다(supersede와 같은 메커니즘).

    - status='deleted' + is_active=False → 검색 즉시 제외 + 목록에서 사라짐
    - 캐시 무효화 → 이 문서를 근거로 만든 답변 재사용 방지
    - DROP_DOCUMENTS outbox → 색인에서 청크 제거. **이게 빠지면 지운 문서가 계속 인용된다**
    - documents row·blob은 보존 → 과거 대화 인용의 원본 다운로드 유지

    반환은 실제로 내려간 document_id들 — 전 버전을 함께 내리므로 요청한 id보다 많을 수 있다.
    **커밋하지 않는다**: 호출부 트랜잭션에 얹힌다(outbox.enqueue와 같은 규약). 문서 UPDATE·
    캐시 삭제·대기열 행이 한 커밋으로 묶여야 "문서는 지웠는데 색인엔 남는" 상태가 안 생긴다.

    status 필터를 두지 않는 것은 의도다 — 이미 deleted인 문서를 다시 지워도 성공으로 본다
    (삭제는 "그 문서가 없는 상태"를 만드는 것이라, 이미 그 상태면 실패로 볼 이유가 없다).
    단건 DELETE가 원래 이렇게 동작했고, 일괄도 같은 규칙을 쓴다.
    """
    if not document_ids:
        return []
    # id → filename을 **서브쿼리로** 푼다. 따로 SELECT해서 값을 받아오면 그 사이에 같은 이름의
    # 새 버전이 들어올 창이 생긴다 — 한 문장이면 그 창이 없다(단건이 쓰던 방식 그대로다).
    # 요청에 같은 문서의 다른 버전이 섞여 있어도 filename으로 모이므로 한 번만 처리된다.
    filenames = (
        select(Document.filename)
        .where(Document.tenant_id == tenant_id)      # 격리 — WHERE 절 명시
        .where(Document.id.in_(document_ids))
    )
    doc_ids = (await session.execute(
        update(Document)
        .where(Document.tenant_id == tenant_id)      # 격리 — UPDATE에도 유지(이중 방어).
                                                     # 이게 빠지면 같은 파일명을 쓰는 남의 테넌트
                                                     # 문서까지 지워진다 — 테스트로 고정해 둔다.
        .where(Document.filename.in_(filenames))
        .values(is_active=False, status='deleted', status_reason='user_deleted')
        .returning(Document.id)
    )).scalars().all()
    if not doc_ids:
        return []

    for did in doc_ids:
        await cache.invalidate_source(session, tenant_id, did)
    outbox.enqueue(session, tenant_id, outbox.DROP_DOCUMENTS, document_ids=list(doc_ids))
    return list(doc_ids)


class FailedReuseConflict(Exception):
    """failed 행을 되살리려는 순간 다른 요청이 먼저 되살렸다 (#161). 라우터가 409로 바꾼다."""


async def _reuse_failed_row(session: AsyncSession, target_id: int, **values) -> int | None:
    """failed 행 하나를 pending으로 되살린다 — **status='failed'인 동안에만** 먹는 조건부 UPDATE.

    같은 failed를 둘이 동시에 재업로드하면 한쪽의 UPDATE가 0행이 된다(그새 pending이 됐으니
    조건이 안 맞는다) — 그쪽이 None을 받아 FailedReuseConflict로 빠진다. 별도 함수로 뗀 이유는
    테스트가 이 경합을 monkeypatch로 재현하기 위해서다(test_동시_삽입은_유니크_인덱스가_막고_409와
    같은 방식). 반환: 되살린 행 id, 0행이면 None.
    """
    return (await session.execute(
        update(Document)
        .where(Document.id == target_id)
        .where(Document.status == 'failed')
        .values(**values)
        .returning(Document.id)
    )).scalar()


async def handle_upload(
        session: AsyncSession,
        tenant_id: str,
        filename: str,
        blob_path: Path,
        description: str | None = None,
        uploaded_by: str | None = None,
        folder_id: int | None = None,
        folder_given: bool = False,
) -> tuple[Document, str | None]:
    """업로드 시점 처리: pending row + 색인 대기열 행을 **같은 트랜잭션**에 등록한다 (#139).
    반환은 (문서, 지워야 할 옛 blob 경로 또는 None) — 옛 blob은 **호출부가 커밋 뒤에** 지운다.
    커밋 전에 지우면 롤백 시 DB는 옛 경로를 가리키는데 파일은 없는 상태가 된다.
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

    **예외 하나 — failed는 그 자리에서 다시 시도한다 (#161).** 같은 이름에 정상(pending·ready)
    행이 없고 failed 행만 있으면 새 version을 만들지 않고 그 행을 pending으로 되살린다.
    failed는 검색에 나오지 않고 사용자에겐 "등록이 안 된 상태"인데, 실패한 **시도**가 개정
    **번호**를 소비하면 "내가 언제 v1을 등록했지?"가 된다. 새 행을 안 만드니 failed v1·ready v2가
    목록에 두 줄로 남는 문제도 애초에 생기지 않는다.
    단, "failed = 청크 0개"는 **항상 참이 아니다** — 색인(②) 뒤 커밋(③)에서 실패가 반복된 문서는
    엔진에 청크가 남아 있다. 되살리는 행은 재색인이 먼저 지우고, 내리는 행은 DROP을 적재한다. exists·_current_version도 같은 기준으로
    failed를 "없는 문서"로 답한다(routers/documents.py) — 세 곳의 기준이 갈리면 확인창에서 본 것과
    다른 결과가 난다.
    """
    # 같은 filename의 모든 버전 조회 (다음 version 계산 + 설정 계승용)
    docs = (await session.execute(
        select(Document)
        .where(Document.tenant_id == tenant_id)
        .where(Document.filename == filename)
    )).scalars().all()

    # failed 재사용 (#161) — 정상 행이 없고 failed만 있을 때. deleted 이력이 섞여 있어도 된다
    # (예: v1 deleted + v2 failed → v2를 되살린다). failed가 여럿이면 최신을 되살리고 나머지는 내린다.
    alive = [d for d in docs if d.status in ALIVE_DOCUMENT_STATUSES]
    failed = [d for d in docs if d.status == 'failed']
    if not alive and failed:
        target = max(failed, key=lambda d: d.version)
        stale_blob = target.blob_path
        reused = await _reuse_failed_row(
            session, target.id,
            status='pending', is_active=False,
            # 워커 ③단계는 status_reason을 이미지 경고일 때만 덮어쓴다 — 여기서 안 지우면
            # 옛 실패 사유가 ready 문서에 그대로 남는다. page_count·char_count·indexed_at도 같이 비운다.
            status_reason=None, page_count=None, char_count=None, indexed_at=None,
            blob_path=str(blob_path), mime=_detect_mime(blob_path),
            # server_default는 INSERT에만 적용된다 — UPDATE에서 갱신하지 않으면 목록(uploaded_at
            # 내림차순, #164)에서 재업로드한 문서가 위로 올라오지 않는다.
            uploaded_at=func.now(), uploaded_by=uploaded_by,
            # 설정은 그 행의 값을 유지(= 계승과 같은 결과), 보냈으면 그 값 — #165 규칙 그대로.
            folder_id=(folder_id if folder_given else target.folder_id),
            description=(description if description is not None else target.description),
        )
        if reused is None:
            raise FailedReuseConflict(filename)
        # 되살리는 행 자신의 캐시도 지운다 — 같은 id를 **다른 내용**으로 되살리는 것이라, 잔여 청크로
        # 만들어진 답변 캐시가 있었다면 새 내용과 어긋난 답을 재사용한다. 워커 ③은 구버전(old_ids)만
        # 무효화하고 자기 자신은 건드리지 않는다(신버전은 새 id라 캐시가 없다는 전제) — 재사용은 그 전제 밖이다.
        await cache.invalidate_source(session, tenant_id, target.id)
        # 나머지 failed는 내린다 — soft_delete_documents와 **같은 세 동작**(상태·캐시·DROP). 그 함수에
        # 위임하지 않는 이유는 filename 기준으로 전 버전을 내려 되살리는 target까지 지우기 때문이다.
        # DROP이 필요한 이유: failed라도 엔진에 청크가 남을 수 있다 — index_pending_document는
        # ②에서 청크를 쓴 뒤 ③에서 커밋하므로, ③ 실패가 반복돼 failed로 굳은 문서는 searchable=True
        # 청크를 가진 채다(리뷰 지적). target은 재색인이 먼저 지우지만(index_parsed_document의
        # _delete_by_terms) others는 재색인이 없어 여기서 지워야 한다. #184 이후 failed 확정 자체가
        # DROP을 등재하므로(outbox.drain — "생길 때 지운다") 여기는 이중 안전판이다("내릴 때 지운다").
        others = [d.id for d in failed if d.id != target.id]
        if others:
            await session.execute(update(Document).where(Document.id.in_(others))
                                  .values(status='deleted', is_active=False,
                                          status_reason='failed_superseded'))
            for did in others:
                await cache.invalidate_source(session, tenant_id, did)
            outbox.enqueue(session, tenant_id, outbox.DROP_DOCUMENTS, document_ids=others)
        await session.refresh(target)      # UPDATE로 바뀐 값을 ORM 객체에 반영 (응답에 쓴다)
        outbox.enqueue(session, tenant_id, outbox.INDEX_DOCUMENT, document_id=target.id)
        return target, stale_blob

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
    return doc, None


