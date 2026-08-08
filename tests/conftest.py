"""통합 테스트 공용 fixture (D-1).

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
import uuid

import pytest
import pytest_asyncio


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

    호출부들이 `from rag.embeddings import ...`로 심볼을 바인딩해 가므로
    각 모듈의 이름을 개별 패치한다 (rag.embeddings만 패치하면 안 먹음).
    리랭커(외부 TEI)도 함께 끈다.
    """
    from rag.embeddings import Embedding

    async def _texts(texts: list[str]) -> list[Embedding]:
        return [Embedding(dense=fake_vector(t)) for t in texts]

    async def _query(text: str) -> Embedding:
        return Embedding(dense=fake_vector(text))

    import rag.cache
    import rag.documents
    import rag.retriever
    import routers.faqs
    from config import settings

    monkeypatch.setattr(routers.faqs, 'embed_texts', _texts)
    monkeypatch.setattr(rag.documents, 'embed_texts', _texts)
    monkeypatch.setattr(rag.retriever, 'embed_texts', _texts)   # 쿼리 확장(#5)으로 배치 임베딩 전환
    monkeypatch.setattr(rag.cache, 'embed_query', _query)
    monkeypatch.setattr(settings, 'rerank_enabled', False)
    return fake_vector


@pytest.fixture
def pass_gate(monkeypatch):
    """근거 게이트 무조건 통과 — 가짜 벡터로는 임계(0.6)를 못 넘어서.
    (4개 테스트 파일에 복붙되던 것을 공용 승격 — #10 리뷰)"""
    import rag.retriever as rt
    monkeypatch.setattr(rt, 'apply_gate', lambda cands, max_dense_distance=0.6: (False, None))


# ── 가짜 LLM ─────────────────────────────────────────────────

class FakeLlm:
    """시스템 프롬프트로 용도를 판별해 고정 응답을 주는 가짜 LLM.

    - 인텐트 가드 → safe/KNOWLEDGE JSON (테스트에서 intent_json으로 변경 가능)
    - condense → 마지막 user 내용에서 질문만 반향
    - 그 외(답변 생성) → answer를 어절 단위로 스트리밍
    """

    def __init__(self, answer: str = '테스트 답변입니다. [테스트문서.pdf v1]'):
        self.answer = answer
        self.intent_json = '{"safe": true, "intent": "KNOWLEDGE"}'
        self.calls: list[str] = []          # 어떤 용도로 호출됐는지 기록 (검증용)
        self.system_prompts: list[tuple[str, str]] = []   # (용도, 시스템 프롬프트) — 주입 내용 검증용

    def _kind(self, messages: list[dict]) -> str:
        system = messages[0]['content'] if messages else ''
        if '분류기' in system or '입력 검사' in system:
            return 'intent'
        if '질문 여러 개' in system:      # CONDENSE_MULTI(#5) — '재작성'보다 먼저 (문자열 겹침)
            return 'condense_multi'
        if '재작성' in system:
            return 'condense'
        return 'generate'

    def _record(self, kind: str, messages: list[dict]) -> None:
        self.system_prompts.append((kind, messages[0]['content'] if messages else ''))

    async def acomplete(self, messages: list[dict]) -> str:
        kind = self._kind(messages)
        self.calls.append(kind)
        self._record(kind, messages)
        if kind == 'intent':
            return self.intent_json
        if kind == 'condense':
            return messages[-1]['content'].splitlines()[-1].strip()
        if kind == 'condense_multi':
            # 멀티쿼리(#5) 규격(3줄) 반향 — 한 줄만 주면 변형이 조용히 비어
            # 플래그 on 통합 테스트가 off와 동일 경로로 축소되는 걸 막는다.
            # 유저 메시지 마지막 줄은 라벨('검색용 독립 질문:')이라 '현재 질문:' 다음 줄을 집는다.
            lines = [l.strip() for l in messages[-1]['content'].splitlines() if l.strip()]
            q = lines[lines.index('현재 질문:') + 1] if '현재 질문:' in lines else lines[-1]
            return f'{q}\n{q} 변형A\n{q} 변형B'
        return self.answer

    async def astream(self, messages: list[dict]):
        kind = self._kind(messages)
        self.calls.append('stream:' + kind)
        self._record(kind, messages)
        for token in self.answer.split(' '):
            yield token + ' '


@pytest.fixture
def fake_llm(monkeypatch):
    """RagService가 쓰는 공용 LLM(shared_llm)을 가짜로 치환."""
    fake = FakeLlm()
    import rag.service
    monkeypatch.setattr(rag.service, 'shared_llm', fake)
    return fake


# ── 데이터 격리 + 루프 위생 ─────────────────────────────────

@pytest_asyncio.fixture
async def _loop_hygiene():
    """테스트 종료 시 루프에 묶인 커넥션 전부 정리 (엔진·Redis·http_async)."""
    yield
    import httpx

    import rag.clients as clients
    from database import engine

    await engine.dispose()
    await clients.cache_redis.connection_pool.disconnect()
    from rag.limiter import query_limiter
    await query_limiter._redis.connection_pool.disconnect()   # 동시성 제한기 싱글톤도 루프에 묶임
    # http_async는 aclose 후 재사용 불가 → 새 인스턴스로 교체
    # (호출부가 함수 안에서 지연 import하므로 모듈 속성 교체가 먹는다)
    await clients.http_async.aclose()
    clients.http_async = httpx.AsyncClient()


async def purge_tenant(t: str) -> None:
    """해당 tenant의 전 테이블 데이터 + Redis 키 정리 (fixture·다중 테넌트 테스트 공용)."""
    from sqlalchemy import delete

    from database import AsyncSessionLocal
    from rag.models import (
        AnswerCache as AnswerCacheRow,
        Chunk,
        Conversation,
        Document,
        Faq,
        Folder,
        Message,
        TenantQuota,
    )
    async with AsyncSessionLocal() as session:
        # FK 순서: 자식(청크·메시지) 먼저
        for model in (Chunk, Message, Conversation, AnswerCacheRow,
                      Faq, Document, Folder, TenantQuota):
            await session.execute(delete(model).where(model.tenant_id == t))
        await session.commit()
    import rag.clients as clients
    keys = [k async for k in clients.cache_redis.scan_iter(match=f'kms:*{t}*')]
    if keys:
        await clients.cache_redis.delete(*keys)


@pytest_asyncio.fixture
async def tenant_id(_loop_hygiene):
    """랜덤 tenant + 테스트 후 정리. _loop_hygiene 의존으로 정리가 커넥션 정리보다 먼저."""
    t = str(uuid.uuid4())
    yield t
    await purge_tenant(t)


@pytest_asyncio.fixture
async def other_tenant_id(_loop_hygiene):
    """교차 테넌트 시나리오용 두 번째 tenant (정리 포함)."""
    t = str(uuid.uuid4())
    yield t
    await purge_tenant(t)


# ── ASGI 클라이언트 ──────────────────────────────────────────

@pytest_asyncio.fixture
async def client(tenant_id, fake_embed):
    """앱을 메모리에서 직접 호출하는 HTTP 클라이언트 (서버 기동 불필요).

    tenant 헤더가 미리 실려 있음. LLM까지 필요한 테스트는 fake_llm을 추가로 요청.
    """
    from httpx import ASGITransport, AsyncClient

    from main import app

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url='http://testserver',
        headers={'X-Tenant-Id': tenant_id},
    ) as c:
        yield c
