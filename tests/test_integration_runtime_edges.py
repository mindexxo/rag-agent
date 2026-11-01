"""런타임 내부 경계 — 캐시 판정기 장애·멀티턴 캐시·설명 변경 무효화·청소 경계, 턴 종료 규칙, 리미터 유령 슬롯·Redis 장애,
설정 기동 실패, 지표·스팬, CORS, 작성자 없는 대화.

HTTP 경로는 conftest 클라이언트로, 함수 단위 규칙은 직접 호출로 본다. 장애는 모듈 속성 패치로 주입한다.
"""

import asyncio
from datetime import timedelta

import pytest
from sqlalchemy import func, select, update

from config import Settings
from database import AsyncSessionLocal
from rag import clients, limiter, otel
from rag.models import AnswerCache as AnswerCacheRow
from rag.models import Conversation, Message
from rag.turn import cache
from rag.turn.turn_state import finalize_turn, sweep_stale_generating
from tests.conftest import ingest, register_faq, sse_events, sse_meta
from tests.helpers_documents import upload_xlsx
from tests.polling import wait_inflight

Q = '환불 기간 알려줘'


async def _ask(client, query: str = Q, **body: object) -> dict:
    res = await client.post('/kms/query', json={'query': query, **body})
    assert res.status_code == 200, res.text
    return sse_meta(res)


# ── 캐시 ──


@pytest.mark.asyncio
async def test_캐시_판정기가_죽으면_거절로_보고_새로_생성한다(
    client, tenant_id, fake_llm, pass_gate, relaxed_quota, monkeypatch
):
    """CA-06 — 잘못 승인(오답 재생)이 잘못 거절보다 비싸서 불확실하면 miss다."""
    await register_faq(client)
    assert (await _ask(client))['cached'] is False
    assert await wait_inflight(tenant_id, 0) == 0

    real = fake_llm.acomplete

    async def _judge_down(messages: list[dict], **kw: object) -> str:
        if fake_llm._kind(messages) == 'reuse_judge':
            raise ConnectionError('judge down')
        return await real(messages, **kw)

    monkeypatch.setattr(fake_llm, 'acomplete', _judge_down)
    meta = await _ask(client)
    assert meta['cached'] is False and meta['reason'] == 'ok'


@pytest.mark.asyncio
async def test_멀티턴_턴도_캐시에_저장되고_재사용된다(client, tenant_id, fake_llm, pass_gate, relaxed_quota):
    """CA-11 — 멀티턴 제외 분기가 없다. 재작성문 기준으로 저장되고, 새 대화의 같은 질문이 그 행을 재생한다."""
    await register_faq(client)
    cid = (await _ask(client, '배송은 며칠 걸려?'))['conversation_id']
    assert await wait_inflight(tenant_id, 0) == 0
    assert (await _ask(client, conversation_id=cid))['cached'] is False  # 멀티턴 둘째 턴 — 생성 후 저장
    assert await wait_inflight(tenant_id, 0) == 0
    async with AsyncSessionLocal() as s:
        texts = (
            (await s.execute(select(AnswerCacheRow.query_text).where(AnswerCacheRow.tenant_id == tenant_id)))
            .scalars()
            .all()
        )
    assert Q in texts
    assert (await _ask(client))['cached'] is True  # 새 대화에서 재생


@pytest.mark.asyncio
async def test_xlsx_설명을_바꾸면_그_문서_근거_캐시만_버린다(
    client, tenant_id, fake_queue, blob_tmp, fake_embed, monkeypatch
):
    """CA-13 — 설명은 청크 본문에 병합돼 재색인되므로 ready 문서면 그 문서 근거 캐시만 버린다."""
    target = (await upload_xlsx(client, '요금표.xlsx', '요금 표'))['document_id']
    assert (await ingest(target))['done'] == 1
    seen: list[int] = []
    real = cache.invalidate_source

    async def _spy(session, tenant: str, source_id: int) -> object:
        seen.append(source_id)
        return await real(session, tenant, source_id)

    monkeypatch.setattr(cache, 'invalidate_source', _spy)
    assert (await client.patch(f'/kms/documents/{target}', json={'description': '요금 안내 표'})).status_code == 200
    assert seen == [target]


@pytest.mark.asyncio
async def test_캐시_청소는_보존_기간을_넘긴_행만_지운다(tenant_id):
    """CA-21 — 정확히 90일 된 행은 남고, 그보다 1초 더 된 행만 지운다(< 비교)."""
    async with AsyncSessionLocal() as s:
        rows = []
        for label in ('경계', '초과'):
            row = AnswerCacheRow(
                tenant_id=tenant_id,
                query_text=label,
                answer='답',
                sources=[],
                source_doc_ids=[],
                query_embedding=[0.01] * 1024,
                model='test',
                cache_key=label,
            )
            s.add(row)
            rows.append(row)
        await s.flush()
        now = (await s.execute(select(func.now()))).scalar_one()
        await s.execute(
            update(AnswerCacheRow)
            .where(AnswerCacheRow.id == rows[0].id)
            .values(last_hit_at=now - timedelta(days=90) + timedelta(seconds=5))
        )
        await s.execute(
            update(AnswerCacheRow)
            .where(AnswerCacheRow.id == rows[1].id)
            .values(last_hit_at=now - timedelta(days=90, seconds=1))
        )
        await s.commit()
        assert await cache.sweep_stale(s) >= 1
        await s.commit()
        left = (
            (await s.execute(select(AnswerCacheRow.query_text).where(AnswerCacheRow.tenant_id == tenant_id)))
            .scalars()
            .all()
        )
    assert left == ['경계']


# ── 턴 상태 ──


@pytest.mark.asyncio
async def test_finalize는_어휘_밖_상태를_거부하고_남의_테넌트_메시지는_무시한다(tenant_id, other_tenant_id):
    """TS-05."""
    async with AsyncSessionLocal() as s:
        conv = Conversation(tenant_id=tenant_id, created_by='test-user')
        s.add(conv)
        await s.flush()
        msg = Message(tenant_id=tenant_id, conversation_id=conv.id, role='assistant', content='', status='generating')
        s.add(msg)
        await s.commit()
        with pytest.raises(ValueError):
            await finalize_turn(s, tenant_id, msg.id, '답', [], status='finished')
        await finalize_turn(s, other_tenant_id, msg.id, '남의 답', [], status='done')
        await s.commit()
        await s.refresh(msg)
    assert (msg.status, msg.content) == ('generating', '')


@pytest.mark.asyncio
async def test_500초를_넘긴_생성이_스윕된_뒤_완주하면_done으로_덮는다(client, tenant_id, fake_llm, pass_gate):
    """TS-13 — 완성된 답을 버리지 않는다(경고만 남긴다)."""
    await register_faq(client)
    fake_llm.pause_after_tokens = 2
    task = asyncio.create_task(client.post('/kms/query', json={'query': Q}))
    await asyncio.wait_for(fake_llm.paused.wait(), timeout=10)
    async with AsyncSessionLocal() as s:
        aid = (
            await s.execute(select(Message.id).where(Message.status == 'generating', Message.tenant_id == tenant_id))
        ).scalar_one()
        await s.execute(update(Message).where(Message.id == aid).values(created_at=func.now() - timedelta(seconds=600)))
        assert await sweep_stale_generating(s) >= 1
        await s.commit()
        assert (await s.get(Message, aid)).status == 'failed'
    fake_llm.pause_after_tokens = None
    fake_llm.resume.set()
    assert [e for e, _ in sse_events(await task)][-1] == 'done'
    assert await wait_inflight(tenant_id, 0) == 0
    async with AsyncSessionLocal() as s:
        assert (await s.get(Message, aid)).status == 'done'


# ── 리미터 ──


@pytest.mark.asyncio
async def test_반납되지_않은_슬롯은_보존_시간_뒤_회수된다(tenant_id, monkeypatch):
    """LM-07 — inflight_max_seconds를 넘긴 토큰은 다음 획득 때 prune된다."""
    from config import settings

    first = await limiter.try_acquire(tenant_id, 1)
    assert first is not None and not first.degraded
    assert await limiter.try_acquire(tenant_id, 1) is None
    later = limiter.time.time() + settings.inflight_max_seconds + 1
    monkeypatch.setattr(limiter.time, 'time', lambda: later)
    revived = await limiter.try_acquire(tenant_id, 1)
    assert revived is not None and not revived.degraded
    await limiter.release(revived)


@pytest.mark.asyncio
async def test_Redis가_죽어도_질의는_상한_없이_통과한다(client, tenant_id, fake_llm, pass_gate, monkeypatch):
    """ST-07 — fail-open. 원격 취소·재접속의 503은 기존·대화 테스트가 고정."""
    await register_faq(client)

    class _Down:
        def __getattr__(self, name: str) -> object:
            async def _fail(*a: object, **kw: object) -> None:
                raise ConnectionError('redis down')

            return _fail

    monkeypatch.setattr(clients, 'shared_redis', _Down())
    res = await client.post('/kms/query', json={'query': Q})
    assert res.status_code == 200 and [e for e, _ in sse_events(res)][-1] == 'done'


# ── 설정 기동 ──


def test_잘못된_추론_수준은_기동_실패():
    """PV-01 — 공급자 이름·상용 모델명 누락은 기존 테스트가 고정."""
    with pytest.raises(ValueError):
        Settings(llm_reasoning_effort='maximum')


@pytest.mark.parametrize('name', ['VLLM_BASE_URL', 'VLLM_MODEL', 'LLM_ENABLE_THINKING'])
def test_옛_환경변수가_env_파일에_남아_있으면_기동_실패(name, tmp_path, monkeypatch):
    """PV-03 — 환경변수 경로는 기존 테스트가, 여기서는 .env 파일 경로와 VLLM_BASE_URL."""
    env = tmp_path / '.env'
    env.write_text(f'{name}=x\n', encoding='utf-8')
    monkeypatch.delenv(name, raising=False)
    monkeypatch.setitem(Settings.model_config, 'env_file', (str(env),))
    with pytest.raises(ValueError, match='이름이 바뀐 설정이 남아 있습니다'):
        Settings()


# ── 관측성·공통 ──


@pytest.mark.asyncio
async def test_metrics는_테넌트_헤더_없이_kms_지표를_낸다(_loop_hygiene):
    """OBS-01·COM-03."""
    from httpx import ASGITransport, AsyncClient

    from main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url='http://testserver') as bare:
        res = await bare.get('/metrics')
    assert res.status_code == 200 and res.headers['content-type'].startswith('text/plain')
    assert 'kms_' in res.text


@pytest.mark.asyncio
async def test_루트_스팬에_턴_속성이_실린다(client, tenant_id, fake_llm, pass_gate, monkeypatch):
    """OBS-06 — ERROR 마킹 규칙은 test_integration_query_contract가 고정."""
    await register_faq(client)
    seen: list[dict] = []
    real = otel.set_attrs

    def _spy(sp: object, attrs: dict) -> None:
        seen.append(dict(attrs))
        real(sp, attrs)

    monkeypatch.setattr(otel, 'set_attrs', _spy)
    meta = await _ask(client)
    root = next(a for a in seen if 'kms.route' in a)
    assert root['kms.tenant_id'] == tenant_id and root['kms.conversation_id'] == meta['conversation_id']
    assert root['kms.route'] == 'knowledge' and root['kms.no_evidence'] is False and 'kms.cache_kind' in root


@pytest.mark.asyncio
async def test_CORS는_허용_origin에만_헤더를_준다(_loop_hygiene):
    """COM-04 — 허용 목록은 설정값, 그 밖의 origin은 preflight 400."""
    from httpx import ASGITransport, AsyncClient

    from config import settings
    from main import app

    allowed = settings.cors_allow_origins.split(',')[0].strip()
    preflight = {'Access-Control-Request-Method': 'POST', 'Access-Control-Request-Headers': 'x-tenant-id'}
    async with AsyncClient(transport=ASGITransport(app=app), base_url='http://testserver') as bare:
        ok = await bare.options('/kms/query', headers={'Origin': allowed, **preflight})
        bad = await bare.options('/kms/query', headers={'Origin': 'https://evil.example', **preflight})
    assert ok.status_code == 200 and ok.headers['access-control-allow-origin'] == allowed
    assert bad.status_code == 400 and 'access-control-allow-origin' not in bad.headers


@pytest.mark.asyncio
async def test_작성자_없는_대화는_누구에게도_보이지_않는다(client, tenant_id):
    """ISO-18 — created_by NULL은 어느 사용자와도 맞지 않는다(헤더 없음=test-user 포함)."""
    async with AsyncSessionLocal() as s:
        conv = Conversation(tenant_id=tenant_id, created_by=None)
        s.add(conv)
        await s.commit()
        cid = conv.id
    for headers in ({}, {'X-User-Id': 'agent-a'}):
        assert (await client.get('/kms/conversations', headers=headers)).json()['total'] == 0
        assert (await client.get(f'/kms/conversations/{cid}/messages', headers=headers)).status_code == 404
