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
import asyncio
import hashlib
import json
import uuid

import pytest
import pytest_asyncio


# ── SSE 파싱 (비스트리밍 JSON 경로 삭제 #26로 전 테스트가 SSE를 읽는다) ──

def sse_events(res_or_text) -> list[tuple[str, object]]:
    """SSE 응답 → [(event, data), ...]. httpx Response나 본문 문자열 모두 받는다.
    이벤트 경계는 빈 줄(\\n\\n), 각 이벤트는 event/data 두 줄이라는 봉투 규격에 의존.
    """
    text = getattr(res_or_text, 'text', res_or_text)
    out = []
    for block in text.strip().split('\n\n'):
        lines = block.splitlines()
        out.append((lines[0].removeprefix('event: '),
                    json.loads(lines[1].removeprefix('data: '))))
    return out


def sse_meta(res_or_text) -> dict:
    """meta 이벤트 payload — conversation_id·cached·cache_kind·reason·assistant_message_id."""
    return next(data for event, data in sse_events(res_or_text) if event == 'meta')


def sse_answer(res_or_text) -> str:
    """delta 이벤트를 이어붙인 최종 답변 텍스트."""
    return ''.join(d['text'] for e, d in sse_events(res_or_text) if e == 'delta')


def sse_done(res_or_text) -> dict:
    """done 이벤트 payload — finish_reason·latency_ms·citations (#56)."""
    return next(data for event, data in sse_events(res_or_text) if event == 'done')


async def register_faq(client) -> int:
    """검색 근거용 FAQ 1건 등록 — 4개 테스트 파일이 같은 것을 복붙하던 것을 공용 승격."""
    res = await client.post('/kms/faqs', json={
        'question': '환불 기간은?', 'variants': [], 'answer': '7일 이내 처리됩니다.',
    })
    return res.json()['id']


async def seed_turn(tenant_id: str, question: str, answer: str, status: str = 'done',
                    standalone: str | None = None, intent: str | None = None,
                    conversation_id: int | None = None,
                    attachments: list[dict] | None = None) -> int:
    """user/assistant 한 턴 시딩 → conversation_id (register_faq와 같은 공용 승격 — #59에서
    이력 격리·RETRY 테스트 2파일이 거의 같은 헬퍼를 복붙하던 것. #63 리뷰에서 첨부 시딩
    복붙이 또 생겨 attachments 인자로 흡수). conversation_id를 주면 기존 대화에 턴을
    이어 붙인다(연속 취소 체인 시딩용)."""
    from database import AsyncSessionLocal
    from rag.models import Conversation, Message
    async with AsyncSessionLocal() as s:
        if conversation_id is None:
            conv = Conversation(tenant_id=tenant_id, created_by='agent-a')
            s.add(conv)
            await s.flush()
            conversation_id = conv.id
        u = Message(tenant_id=tenant_id, conversation_id=conversation_id, role='user',
                    content=question, standalone_query=standalone, attachments=attachments)
        s.add(u)
        await s.flush()
        s.add(Message(tenant_id=tenant_id, conversation_id=conversation_id, role='assistant',
                      content=answer, status=status, question_message_id=u.id, intent=intent))
        await s.commit()
    return conversation_id


class _FakePool:
    def __init__(self, jobs: list):
        self._jobs = jobs

    async def enqueue_job(self, name: str, *args) -> None:
        self._jobs.append((name, *args))

    async def aclose(self) -> None:
        pass


@pytest.fixture
def fake_queue(monkeypatch):
    """arq enqueue를 기록만 하는 가짜로 — 실제 Redis 큐에 잡이 쌓이지 않게."""
    jobs: list = []

    async def _create_pool(*a, **kw):
        return _FakePool(jobs)

    import routers.documents as rd
    monkeypatch.setattr(rd, 'create_pool', _create_pool)
    return jobs


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

    호출부들이 `from rag.embeddings import ...`로 심볼을 바인딩해 가므로
    각 모듈의 이름을 개별 패치한다 (rag.embeddings만 패치하면 안 먹음).
    리랭커(외부 TEI)도 함께 끈다.

    반환값의 `.embed_calls`는 누적 TEI 왕복 횟수다 (#50 — 턴당 1회 검증용).
    값이 아니라 **횟수**를 세는 이유: fake_vector는 텍스트 결정적이라 같은 질의는 항상
    같은 벡터고, "재사용했는지"와 "다시 계산했는지"를 값으로는 구별할 수 없다.
    세는 단위는 embed_texts/embed_query 호출 1건 = 왕복 1회다 (배치 안의 텍스트 개수가
    아니다 — 줄이려는 대상이 왕복이므로). FakeLlm.calls와 같은 관례.
    기대 벡터를 손으로 만들 때는 fake_vector를 직접 import해 쓴다 — 그건 왕복이 아니다.
    """
    from types import SimpleNamespace

    from rag.embeddings import Embedding

    counting = SimpleNamespace(embed_calls=0)

    async def _texts(texts: list[str]) -> list[Embedding]:
        counting.embed_calls += 1
        return [Embedding(dense=fake_vector(t)) for t in texts]

    async def _query(text: str) -> Embedding:
        counting.embed_calls += 1
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
    return counting


@pytest.fixture
def pass_gate(monkeypatch):
    """근거 게이트 무조건 통과 — 가짜 벡터로는 임계(0.6)를 못 넘어서.
    (4개 테스트 파일에 복붙되던 것을 공용 승격 — #10 리뷰)"""
    import rag.retriever as rt
    monkeypatch.setattr(rt, 'apply_gate', lambda cands, max_dense_distance=0.6: (False, None))


# ── 가짜 LLM ─────────────────────────────────────────────────

def _current_question(user_content: str) -> str:
    """condense 유저 메시지에서 '현재 질문:' 다음 줄을 집는다 — FakeLlm 반향용."""
    lines = [l.strip() for l in user_content.splitlines() if l.strip()]
    return lines[lines.index('현재 질문:') + 1] if '현재 질문:' in lines else lines[-1]


class FakeSchemaRejected(Exception):
    """구버전 vLLM의 스키마 거부(400) 재현 — is_schema_rejected가 보는 status_code 덕 타이핑."""
    status_code = 400


class FakeLlm:
    """시스템 프롬프트로 용도를 판별해 고정 응답을 주는 가짜 LLM.

    - 인텐트 가드 → safe/KNOWLEDGE JSON (테스트에서 intent_json으로 변경 가능)
    - condense → 마지막 user 내용에서 질문만 반향
    - 그 외(답변 생성) → answer를 어절 단위로 스트리밍
    """

    def __init__(self, answer: str | None = None):
        # 기본 답변은 **1번 문서를 인용하는** 출처 꼬리를 단다 (#56 도입, #61에서 빈 꼬리→1번).
        # 빈 꼬리였던 것을 바꾼 이유: #61부터 인용 0건이 "근거없음"의 정의가 되어 캐시 제외·
        # 지표 분자에 걸린다. 빈 꼬리를 기본값으로 두면 대다수 테스트의 "정상 답변"이 근거없음으로
        # 분류돼(실제로 캐시 저장이 스킵돼 3개 테스트가 깨졌다) 픽스처가 비현실적인 전제를 깔게 된다.
        # 후보가 없는 테스트(차단·근거없음·OTHER)에서는 1번이 범위 밖이라 resolve_citations가
        # 걸러내 citations=[]가 그대로 유지된다 — 그 경로의 기대값은 안 바뀐다.
        # 인용 검증 테스트는 여전히 answer를 직접 교체해서 쓴다.
        # 어절 3개 유지 주의: pause_after_tokens=2(셋째 토큰 정지)를 쓰는 취소 테스트들의 전제.
        from rag.citation_labels import citation_tail
        self.answer = answer if answer is not None else f'테스트 답변입니다. {citation_tail([1])}'
        self.intent_json = '{"safe": true, "intent": "KNOWLEDGE"}'
        self.reuse_judge_json = '{"same_answer": true, "reason": "테스트 고정 승인"}'   # 캐시 판정 (#113)
        self.calls: list[str] = []          # 어떤 용도로 호출됐는지 기록 (검증용)
        self.system_prompts: list[tuple[str, str]] = []   # (용도, 시스템 프롬프트) — 주입 내용 검증용
        self.extra_bodies: list[dict | None] = []   # astream별 extra_body — 꼬리 제약 주입 검증용(#56·#65)
        self.acomplete_extra_bodies: list[dict | None] = []   # acomplete별 — 스키마 주입·재시도 증거(#43)
        self.reject_schema = False   # True면 structured_outputs 실린 acomplete를 400으로 거부 — 구버전 서버 재현(#43)
        # 취소 테스트용 정지 지점 (#30). None이면 기존 동작 그대로.
        # 왜 필요한가: astream에 진짜 await가 없으면 이벤트 루프에 제어가 넘어가지 않아
        # task.cancel()이 전달될 지점 자체가 없다 — 생성이 그대로 완주한다(실측).
        # 시간(sleep) 대신 Event로 멈추는 이유: 테스트가 "정확히 N토큰 뒤"를 잡을 수 있어
        # CI에서 타이밍 플레이크가 나지 않는다.
        self.pause_after_tokens: int | None = None
        self.finish_reason = 'stop'   # astream 완주 시 콜백에 전달 (#129) — 'length'로 바꿔 잘림 재현
        # 생성 실패 테스트용 — N번째 토큰에서 스트림이 끊긴 상황(LLM·네트워크 장애) 주입.
        # pause와 달리 재개가 없다: 실패 경로의 저장 규칙을 보려는 것이라 중단 자체가 목적.
        self.raise_after_tokens: int | None = None
        self.paused = asyncio.Event()        # 테스트: 정지 지점 도달을 기다린다
        self.resume = asyncio.Event()        # 테스트: set()하면 스트림이 계속된다

    def _kind(self, messages: list[dict]) -> str:
        system = messages[0]['content'] if messages else ''
        if '분류기' in system or '입력 검사' in system:
            return 'intent'
        if '질문 여러 개' in system:      # CONDENSE_MULTI(#5) — '재작성'보다 먼저 (문자열 겹침)
            return 'condense_multi'
        if '재작성' in system:
            return 'condense'
        if '재사용 판정기' in system:     # 캐시 재사용 판정 (#113)
            return 'reuse_judge'
        return 'generate'

    def _record(self, kind: str, messages: list[dict]) -> None:
        self.system_prompts.append((kind, messages[0]['content'] if messages else ''))

    async def acomplete(self, messages: list[dict], extra_body: dict | None = None) -> str:
        kind = self._kind(messages)
        self.calls.append(kind)
        self._record(kind, messages)
        self.acomplete_extra_bodies.append(extra_body)
        if self.reject_schema and extra_body and 'structured_outputs' in extra_body:
            raise FakeSchemaRejected('structured_outputs를 모르는 구버전 서버 재현')
        if kind == 'intent':
            return self.intent_json
        if kind == 'reuse_judge':
            # 캐시 재사용 판정(#113) — 판정 승인 없이는 히트가 없으므로 기본 승인.
            # 거절 시나리오는 reuse_judge_json을 교체해서 쓴다.
            return self.reuse_judge_json
        if kind == 'condense':
            # 실제 질문을 JSON으로 반향 (#43). 구현 주의: 옛 코드는 마지막 줄을 반향했는데
            # 그건 질문이 아니라 라벨이었다(잠복 버그 — 이 값을 검증하는 테스트가 없어 안 드러남).
            return json.dumps({'standalone': _current_question(messages[-1]['content'])},
                              ensure_ascii=False)
        if kind == 'condense_multi':
            # 멀티쿼리(#5) 규격 반향 — variants를 채워야 플래그 on 통합 테스트가
            # off와 동일 경로로 축소되지 않는다.
            q = _current_question(messages[-1]['content'])
            return json.dumps({'standalone': q, 'variants': [f'{q} 변형A', f'{q} 변형B']},
                              ensure_ascii=False)
        return self.answer

    async def astream(self, messages: list[dict], extra_body: dict | None = None,
                      on_finish_reason=None):
        kind = self._kind(messages)
        self.calls.append('stream:' + kind)
        self._record(kind, messages)
        self.extra_bodies.append(extra_body)
        for i, token in enumerate(self.answer.split(' ')):
            if self.raise_after_tokens is not None and i == self.raise_after_tokens:
                raise RuntimeError('LLM 스트림 중단 (테스트 주입)')
            if self.pause_after_tokens is not None and i == self.pause_after_tokens:
                self.paused.set()
                await self.resume.wait()      # 취소 테스트가 여기서 task.cancel()을 건다
            yield token + ' '
        # 실제 llm.py처럼 완주 시 종료 사유를 콜백으로 알린다 (#129).
        # finish_reason 속성을 'length'로 바꾸면 잘림 시나리오를 흉내 낼 수 있다.
        if on_finish_reason:
            on_finish_reason(self.finish_reason)


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
    # 공용 Redis 하나만 정리하면 된다 (#30: limiter·취소 pub/sub이 clients.shared_redis를 공유)
    await clients.shared_redis.connection_pool.disconnect()
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
    # Redis 잔재는 리미터의 in-flight 키뿐 (kms:inflight:*) — 취소 채널은 pub/sub이라 키가 안 남는다
    import rag.clients as clients

    keys = [k async for k in clients.shared_redis.scan_iter(match=f'kms:*{t}*')]
    if keys:
        await clients.shared_redis.delete(*keys)


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
