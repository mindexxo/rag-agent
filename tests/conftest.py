"""통합 테스트 공용 fixture.

원칙:
- 실제 앱(main.app)·Postgres·Redis를 그대로 쓴다. 가짜는 외부 의존(임베딩 TEI·LLM)만.
- 데이터 격리 = 랜덤 tenant + teardown 정리. 트랜잭션 롤백 방식이 아님 —
  commit 경계 자체(스트림 중 커밋, 백그라운드 태스크의 별도 세션 커밋)가 검증 대상이라서.
- 루프 위생: pytest-asyncio는 테스트마다 새 이벤트 루프를 만든다. 풀에 남은 커넥션이
  이전 루프에 묶여 터지므로(DB 엔진·Redis·http_async), 테스트 후 전부 정리한다.

가짜 임베딩 주의: 벡터는 결정적이지만 의미 없음 — 코사인 거리가 근거 게이트(0.6)를
통과한다는 보장이 없다. 검색 매칭까지 검증하는 테스트는 apply_gate를 함께 패치할 것.
"""

import hashlib
import json
import uuid

import pytest
import pytest_asyncio

from tests.fake_llm import FakeLlm
from tests.fake_vlm import FAKE_CAPTION, _fake_vlm_server, fake_vlm  # noqa: F401 — 픽스처·상수 재노출
from tests.tenant_client import client_for

# ── SSE 파싱 ──


def sse_events(res_or_text) -> list[tuple[str, object]]:
    """SSE 응답 → [(event, data), ...]. httpx Response나 본문 문자열 모두 받는다.
    이벤트 경계는 빈 줄(\\n\\n), 각 이벤트는 event/data 두 줄이라는 봉투 규격에 의존.
    """
    text = getattr(res_or_text, 'text', res_or_text)
    out = []
    for block in text.strip().split('\n\n'):
        lines = block.splitlines()
        out.append((lines[0].removeprefix('event: '), json.loads(lines[1].removeprefix('data: '))))
    return out


def sse_meta(res_or_text) -> dict:
    """meta 이벤트 payload — conversation_id·cached·cache_kind·reason·assistant_message_id."""
    return next(data for event, data in sse_events(res_or_text) if event == 'meta')


def sse_answer(res_or_text) -> str:
    """delta 이벤트를 이어붙인 최종 답변 텍스트."""
    return ''.join(d['text'] for e, d in sse_events(res_or_text) if e == 'delta')


def sse_done(res_or_text) -> dict:
    """done 이벤트 payload — finish_reason·latency_ms·citations."""
    return next(data for event, data in sse_events(res_or_text) if event == 'done')


async def register_faq(client) -> int:
    """검색 근거용 FAQ 1건 등록."""
    res = await client.post(
        '/kms/faqs',
        json={
            'question': '환불 기간은?',
            'variants': [],
            'answer': '7일 이내 처리됩니다.',
        },
    )
    faq_id = res.json()['id']
    await sync_faq(faq_id)  # 라우터는 대기열 행만 남긴다 — 검색에 보이려면 drain
    return faq_id


async def seed_turn(
    tenant_id: str,
    question: str,
    answer: str,
    status: str = 'done',
    standalone: str | None = None,
    intent: str | None = None,
    conversation_id: int | None = None,
    attachments: list[dict] | None = None,
) -> int:
    """user/assistant 한 턴 시딩 → conversation_id. conversation_id를 주면 기존 대화에 턴을
    이어 붙인다(연속 취소 체인 시딩용)."""
    from database import AsyncSessionLocal
    from rag.models import Conversation, Message

    async with AsyncSessionLocal() as s:
        if conversation_id is None:
            conv = Conversation(tenant_id=tenant_id, created_by='agent-a')
            s.add(conv)
            await s.flush()
            conversation_id = conv.id
        u = Message(
            tenant_id=tenant_id,
            conversation_id=conversation_id,
            role='user',
            content=question,
            standalone_query=standalone,
            attachments=attachments,
        )
        s.add(u)
        await s.flush()
        s.add(
            Message(
                tenant_id=tenant_id,
                conversation_id=conversation_id,
                role='assistant',
                content=answer,
                status=status,
                question_message_id=u.id,
                intent=intent,
            )
        )
        await s.commit()
    return conversation_id


@pytest.fixture
def fake_queue():
    """호환용 더미 — 업로드는 대기열 행만 남기고 Redis를 만지지 않아 패치할 것이 없다.
    시그니처에 남아 있는 테스트들이 그대로 돌게 빈 리스트만 돌려준다."""
    return []


async def ingest(document_id: int) -> dict:
    """문서 하나에 걸린 outbox 행만 워커처럼 처리한다 — 업로드 뒤엔 INDEX_DOCUMENT, 삭제·토글
    뒤엔 DROP/META_DOCUMENTS. 라우터는 색인을 직접 하지 않으므로 검색 상태를 보려면 이걸 부른다.

    공유 개발계 DB에서 전체 drain을 돌리면 다른 세션의 pending 업로드까지 처리해 버리므로
    row_ids로 한정한다. 실패 판정(attempts→failed)까지 운영과 같은 경로를 탄다.
    """
    from database import AsyncSessionLocal
    from rag import outbox

    async with AsyncSessionLocal() as s:
        ids = await outbox.pending_row_ids(s, document_id=document_id)
    return await outbox.drain_once(row_ids=ids)


async def sync_faq(faq_id: int) -> dict:
    """FAQ 하나에 걸린 대기열 행(INDEX_FAQ·META_FAQS·DROP_FAQS)만 drain — 라우터가 색인을 직접
    하지 않으므로 등록·수정·토글·삭제 뒤 검색 상태를 보려면 이걸 불러야 한다."""
    from database import AsyncSessionLocal
    from rag import outbox

    async with AsyncSessionLocal() as s:
        ids = await outbox.pending_row_ids(s, faq_id=faq_id)
    return await outbox.drain_once(row_ids=ids)


async def faq_doc(faq_id: int) -> dict | None:
    """FAQ 청크의 색인 문서(_source). 없으면 None — 청크는 OpenSearch에만 있다."""
    from config import settings
    from rag.search import os_client, os_index

    resp = await os_client.client().get(
        index=settings.opensearch_index, id=str(os_index.chunk_os_id(faq_id=faq_id)), ignore=404
    )
    return resp.get('_source') if resp.get('found') else None


async def indexed_chunk_texts(document_id: int) -> list[str]:
    """문서의 색인 청크 본문들(chunk_index 순)."""
    from config import settings
    from rag.search import os_client

    resp = await os_client.client().search(
        index=settings.opensearch_index,
        body={
            'size': 1000,
            '_source': ['chunk_id', 'text'],
            'query': {'term': {'document_id': document_id}},
            'sort': [{'chunk_id': 'asc'}],
        },
    )
    return [h['_source']['text'] for h in resp['hits']['hits']]


@pytest.fixture
def blob_tmp(monkeypatch, tmp_path):
    """blob 저장소를 테스트 임시 디렉터리로 — 실제 blob 디렉터리 오염 방지."""
    from config import settings

    monkeypatch.setattr(settings, 'blob_storage_dir', str(tmp_path))
    return tmp_path


# ── 가짜 임베딩 ──────────────────────────────────────────────


def fake_vector(text: str) -> list[float]:
    """결정적 1024차원 가짜 벡터 — 같은 텍스트는 같은 벡터 (배관 검증용, 의미 없음)."""
    out: list[float] = []
    block = 0
    while len(out) < 1024:
        seed = hashlib.sha256(f'{block}:{text}'.encode()).digest()
        out.extend((b / 255.0) - 0.5 for b in seed)
        block += 1
    return out[:1024]


@pytest.fixture
def fake_embed(monkeypatch):
    """임베딩을 결정적 가짜로 — TEI 서버 없이 인덱싱·캐시 경로가 돈다.

    호출부들이 `from rag.search.embeddings import ...`로 심볼을 바인딩해 가므로
    각 모듈의 이름을 개별 패치한다 (rag.search.embeddings만 패치하면 안 먹음).
    리랭커(외부 TEI)도 함께 끈다.

    반환값의 `.embed_calls`는 누적 TEI 왕복 횟수다 (턴당 1회 검증용).
    값이 아니라 **횟수**를 세는 이유: fake_vector는 텍스트 결정적이라 같은 질의는 항상
    같은 벡터고, "재사용했는지"와 "다시 계산했는지"를 값으로는 구별할 수 없다.
    세는 단위는 embed_texts/embed_query 호출 1건 = 왕복 1회다 (배치 안의 텍스트 개수가
    아니다 — 줄이려는 대상이 왕복이므로). FakeLlm.calls와 같은 관례.
    기대 벡터를 손으로 만들 때는 fake_vector를 직접 import해 쓴다 — 그건 왕복이 아니다.
    """
    from types import SimpleNamespace

    from rag.search.embeddings import Embedding

    counting = SimpleNamespace(embed_calls=0)

    async def _texts(texts: list[str], kind: str = "document") -> list[Embedding]:
        counting.embed_calls += 1
        return [Embedding(dense=fake_vector(t)) for t in texts]

    async def _query(text: str) -> Embedding:
        counting.embed_calls += 1
        return Embedding(dense=fake_vector(text))

    import rag.ingestion
    import rag.search.embeddings
    import rag.search.os_index
    import rag.search.retriever
    import rag.turn.cache
    from config import settings

    # 톱레벨로 이름을 당겨온 모듈은 개별 패치가 필요하다 — 원본 모듈 속성만 바꾸면 그 바인딩은
    # 옛 함수를 계속 가리킨다(조용히 실제 TEI를 때린다). os_index는 index_faq_chunks가 쓴다.
    monkeypatch.setattr(rag.search.embeddings, 'embed_texts', _texts)
    monkeypatch.setattr(rag.search.os_index, 'embed_texts', _texts)
    monkeypatch.setattr(rag.ingestion, 'embed_texts', _texts)
    monkeypatch.setattr(rag.search.retriever, 'embed_texts', _texts)  # 쿼리 확장의 배치 임베딩
    monkeypatch.setattr(rag.turn.cache, 'embed_query', _query)
    monkeypatch.setattr(settings, 'rerank_enabled', False)
    return counting


@pytest.fixture
def pass_gate(monkeypatch):
    """근거 게이트 무조건 통과 — 가짜 벡터로는 임계(0.6)를 못 넘어서."""
    import rag.search.retriever as rt

    monkeypatch.setattr(rt, 'apply_gate', lambda cands, max_dense_distance=0.6: rt.GateVerdict(False, None))


@pytest_asyncio.fixture
async def relaxed_quota(tenant_id):
    """동시 in-flight 상한을 넉넉히 준다 — 같은 사용자로 연달아 질의하는 테스트용.

    사용자별 기본 상한이 1건이라(config.user_concurrency_default), 같은 X-User-Id로 연속
    질의하면 **앞 요청의 슬롯이 아직 반납되기 전**이라 429를 맞을 수 있다. 생성 경로는 반납을
    백그라운드 태스크로 넘기므로(routers/kms.py의 concurrency_guard — 연결이 끊겨도 태스크가
    도는 동안 슬롯을 쥐고 있어야 한다) 응답이 끝나도 슬롯이 잠깐 남아 머신 속도에 따라 결과가 갈린다.

    상한만 올릴 뿐 limiter를 무력화하지 않는다 — 실제 경로(concurrency_guard → limiter)는
    그대로 탄다. 동시성 제한 자체를 검증하는 테스트에는 쓰지 마라.
    """
    from database import AsyncSessionLocal
    from rag.models import TenantQuota

    async with AsyncSessionLocal() as s:
        s.add(TenantQuota(tenant_id=tenant_id, concurrency_limit=50, user_concurrency=50))
        await s.commit()
    yield  # 정리는 purge_tenant가 한다 (TenantQuota 포함)


# ── 가짜 LLM ─────────────────────────────────────────────────


@pytest.fixture
def fake_llm(monkeypatch):
    """RagService가 쓰는 공용 LLM(shared_llm)과 보조 LLM(shared_aux_llm)을 같은 가짜로 치환."""
    fake = FakeLlm()
    import rag.turn.service

    monkeypatch.setattr(rag.turn.service, 'shared_llm', fake)
    monkeypatch.setattr(rag.turn.service, 'shared_aux_llm', fake)
    return fake


# ── 데이터 격리 + 루프 위생 ─────────────────────────────────


@pytest_asyncio.fixture
async def _loop_hygiene():
    """테스트마다 루프에 묶이는 연결을 정리한다 (엔진·Redis·http_async·OpenSearch).

    공용 Redis(clients.shared_redis — limiter·취소·재접속이 호출 시점에 읽는다)는 정리가 아니라 앞뒤 교체다.
    풀의 내부 Lock은 처음 경합한 루프에 묶여 disconnect() 뒤에도 남고, 다른 테스트 루프에서 경합하면
    "bound to a different event loop"로 죽는다 — 실패를 삼키는 리미터 반납에선 슬롯이 조용히 남는다.
    """
    import httpx
    import redis.asyncio as aioredis

    import rag.clients as clients
    from config import settings
    from database import engine

    clients.shared_redis = aioredis.from_url(settings.redis_url)
    yield

    await engine.dispose()
    await clients.shared_redis.aclose()
    clients.shared_redis = aioredis.from_url(settings.redis_url)  # lazy — 이 픽스처 밖 테스트용
    # http_async는 aclose 후 재사용 불가 → 새 인스턴스로 교체
    # (호출부가 함수 안에서 지연 import하므로 모듈 속성 교체가 먹는다)
    await clients.http_async.aclose()
    clients.http_async = httpx.AsyncClient()
    # OpenSearch 클라이언트(aiohttp)도 루프에 묶인다 — 닫고 비워 다음 테스트가 새로 만들게
    from rag.search import os_client

    await os_client.close_client()


async def purge_tenant(t: str) -> None:
    """해당 tenant의 전 테이블 데이터 + 검색 인덱스 문서 + Redis 키 정리 (fixture·다중 테넌트 테스트 공용).

    테넌트의 색인 문서는 운영 삭제 경로(_delete_by_terms)로 지운다(인덱스 보장은 픽스처 시작의
    _ensure_search_index — 직접 부르는 호출자를 위해 여기서도 한 번 더 보장).
    """
    from sqlalchemy import delete

    from database import AsyncSessionLocal
    from rag.search import os_client, os_index

    await os_client.ensure_index()
    await os_index._delete_by_terms('tenant_id', [t])
    from rag.models import (
        AnswerCache as AnswerCacheRow,
    )
    from rag.models import (
        Conversation,
        Document,
        Faq,
        FaqCategory,
        Folder,
        Message,
        SearchIndexOutbox,
        TenantQuota,
    )

    async with AsyncSessionLocal() as session:
        # FK 순서: 자식(청크·메시지) 먼저. outbox 행(done/failed 포함)도 테넌트 단위로 지운다 —
        # 운영은 행을 보존하지만(이력) 테스트의 랜덤 테넌트 행이 공유 DB에 쌓일 이유는 없다.
        for model in (
            Message,
            Conversation,
            AnswerCacheRow,
            Faq,
            FaqCategory,
            Document,
            Folder,
            TenantQuota,
            SearchIndexOutbox,
        ):
            await session.execute(delete(model).where(model.tenant_id == t))
        await session.commit()
    # Redis 잔재는 리미터의 in-flight 키뿐 (kms:inflight:*) — 취소 채널은 pub/sub이라 키가 안 남는다
    import rag.clients as clients

    keys = [k async for k in clients.shared_redis.scan_iter(match=f'kms:*{t}*')]
    if keys:
        await clients.shared_redis.delete(*keys)


async def _ensure_search_index() -> None:
    """검색 인덱스는 테스트 스위트의 전제다(청크는 OpenSearch에만 있다). 픽스처 **시작**에서
    보장한다 — 정리(purge_tenant)에서만 부르면 첫 테스트가 없는 인덱스에 색인하다 404로 죽는다.
    엔진이 안 떠 있으면 여기서 바로 실패한다 — 조용히 PG만으로 도는 척하지 않는다."""
    from rag.search import os_client

    await os_client.ensure_index()


@pytest_asyncio.fixture
async def tenant_id(_loop_hygiene):
    """랜덤 tenant + 테스트 후 정리. _loop_hygiene 의존으로 정리가 커넥션 정리보다 먼저."""
    await _ensure_search_index()
    t = str(uuid.uuid4())
    yield t
    await purge_tenant(t)


@pytest_asyncio.fixture
async def other_tenant_id(_loop_hygiene):
    """교차 테넌트 시나리오용 두 번째 tenant (정리 포함)."""
    await _ensure_search_index()
    t = str(uuid.uuid4())
    yield t
    await purge_tenant(t)


# ── ASGI 클라이언트 ──────────────────────────────────────────


@pytest_asyncio.fixture
async def client(tenant_id, fake_embed):
    """앱을 메모리에서 직접 호출하는 HTTP 클라이언트 (서버 기동 불필요).

    tenant 헤더가 미리 실려 있음. LLM까지 필요한 테스트는 fake_llm을 추가로 요청.
    """
    async with client_for(tenant_id) as c:
        yield c


@pytest.fixture(autouse=True, scope='session')
def _no_remote_ocr():
    """OCR 엔진은 원격 Triton이라 기본값(True) 그대로면 PDF를 태우는 모든 테스트가 그 서버에 묶인다
    — 도달 불가면 _docling_sections가 실패로 올려 문서 테스트가 전부 깨진다. 임베딩을 fake로 바꾸는 것과
    같은 이유로 세션 시작에 한 번 끈다. OCR 자체는 tests/test_chunking_docling.py::TestRemoteOcr가 자기
    monkeypatch로 켜고(끝나면 되돌아온다) 컨버터 싱글턴도 스스로 비운다.

    **함수 스코프 + monkeypatch로 만들면 안 된다** — autouse가 monkeypatch를 먼저 붙잡아 비동기 클라이언트·
    이벤트 루프 픽스처와의 해제 순서가 바뀌고, 루프가 닫힌 뒤 undo가 돌아 test_limiter_fail_open·
    test_retry_dispatch가 teardown에서 'Event loop is closed'로 깨진다."""
    from config import settings

    settings.docling_do_ocr = False
    settings.ocr_provider = 'triton'  # 상용 OCR도 같은 이유로 끈다 — .env가 llm이면 PDF 테스트가 상용 LLM을 부른다. 켜는 건 tests/test_ocr_llm.py
