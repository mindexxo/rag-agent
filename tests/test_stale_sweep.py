"""고착 generating 스윕 통합 테스트 — rag.conversation.sweep_stale_generating (#46).

lazy 스윕(GET 조회 시 정리)을 cron 단독으로 옮긴 변경의 회귀 그물이다.
옮기기 전엔 스윕 자체를 고정하는 테스트가 0건이었다.

실제 DB 사용 (conftest 패턴). LLM·TEI 불필요 — 스윕은 순수 UPDATE다.
"""
from datetime import timedelta

import pytest
from sqlalchemy import func, select, update

from database import AsyncSessionLocal
from rag.turn_state import GENERATION_STALE_SECONDS, sweep_stale_generating
from rag.models import Conversation, Message

USER_A = {'X-User-Id': 'agent-a'}


async def _seed_turn(tenant_id: str, *, status: str, age_seconds: int,
                     created_by: str = 'agent-a') -> int:
    """대화 1개 + assistant 메시지 1개를 심는다. 반환: 메시지 id.

    created_at은 **DB 서버 시계**(func.now())로 되돌린다 — 스윕의 비교도 func.now()라
    같은 시계여야 한다. 파이썬 datetime.now()로 심으면 로컬(KST)과 서버(UTC) 시차만큼
    미래로 심겨 스윕에 안 걸린다(실제로 그렇게 한 번 실패했다).
    """
    async with AsyncSessionLocal() as s:
        conv = Conversation(tenant_id=tenant_id, created_by=created_by)
        s.add(conv)
        await s.flush()
        msg = Message(tenant_id=tenant_id, conversation_id=conv.id, role='assistant',
                      content='', status=status)
        s.add(msg)
        await s.flush()
        mid = msg.id
        await s.execute(
            update(Message).where(Message.id == mid)
            .values(created_at=func.now() - timedelta(seconds=age_seconds))
        )
        await s.commit()
    return mid


async def _status_of(message_id: int) -> str:
    async with AsyncSessionLocal() as s:
        return (await s.execute(
            select(Message.status).where(Message.id == message_id)
        )).scalar_one()


@pytest.mark.asyncio
async def test_임계_지난_generating은_failed가_된다(tenant_id):
    stale = await _seed_turn(tenant_id, status='generating',
                             age_seconds=GENERATION_STALE_SECONDS + 60)
    async with AsyncSessionLocal() as s:
        swept = await sweep_stale_generating(s)
        await s.commit()
    assert swept >= 1                       # 다른 테스트 잔재가 함께 걸릴 수 있어 >=
    assert await _status_of(stale) == 'failed'


@pytest.mark.asyncio
async def test_임계_이내_generating은_건드리지_않는다(tenant_id):
    fresh = await _seed_turn(tenant_id, status='generating', age_seconds=10)
    async with AsyncSessionLocal() as s:
        await sweep_stale_generating(s)
        await s.commit()
    assert await _status_of(fresh) == 'generating'


@pytest.mark.asyncio
async def test_generating_외_상태는_오래돼도_무변화(tenant_id):
    # done·blocked·cancelled·failed 어느 것도 스윕 대상이 아니다 — WHERE가 status만 본다는 계약
    ids = {status: await _seed_turn(tenant_id, status=status,
                                    age_seconds=GENERATION_STALE_SECONDS + 60)
           for status in ('done', 'blocked', 'cancelled', 'failed')}
    async with AsyncSessionLocal() as s:
        await sweep_stale_generating(s)
        await s.commit()
    for status, mid in ids.items():
        assert await _status_of(mid) == status


@pytest.mark.asyncio
async def test_전역이다_다른_테넌트도_함께_치유된다(tenant_id, other_tenant_id):
    # lazy(대화 단위) → cron(전역)의 핵심 차이. 위생 스윕이라 테넌트 필터가 없는 게 맞다 —
    # 상태만 바꾸고 아무것도 반환·노출하지 않으므로 격리 위반이 아니다(sweep_stale_cache와 동일).
    a = await _seed_turn(tenant_id, status='generating',
                         age_seconds=GENERATION_STALE_SECONDS + 60)
    b = await _seed_turn(other_tenant_id, status='generating',
                         age_seconds=GENERATION_STALE_SECONDS + 60)
    async with AsyncSessionLocal() as s:
        swept = await sweep_stale_generating(s)
        await s.commit()
    assert swept >= 2
    assert await _status_of(a) == 'failed'
    assert await _status_of(b) == 'failed'


@pytest.mark.asyncio
async def test_GET은_더이상_상태를_바꾸지_않는다(client, tenant_id):
    """lazy 스윕 제거의 증명 — 이 케이스가 없으면 '지웠다고 착각'을 못 잡는다(#42 유형).

    오래된 generating을 심고 메시지 조회 API를 때려도 응답·DB 모두 generating 그대로여야
    한다. 치유는 이제 워커 cron(rag/worker.py)만의 소관이다.
    """
    stale = await _seed_turn(tenant_id, status='generating',
                             age_seconds=GENERATION_STALE_SECONDS + 60)
    async with AsyncSessionLocal() as s:
        conv_id = (await s.execute(
            select(Message.conversation_id).where(Message.id == stale)
        )).scalar_one()

    res = await client.get(f'/kms/conversations/{conv_id}/messages', headers=USER_A)
    assert res.status_code == 200
    assert res.json()[0]['status'] == 'generating'      # 응답도
    assert await _status_of(stale) == 'generating'      # DB도 그대로


def test_cron에_등록돼_있다():
    """워커 설정 회귀 — 함수를 만들고 등록을 빠뜨리면 스윕이 조용히 죽는다."""
    from rag.worker import WorkerSettings
    names = [c.name for c in WorkerSettings.cron_jobs]
    assert 'cron:sweep_stale_generating' in names   # arq가 'cron:' 접두사를 붙인다
    job = next(c for c in WorkerSettings.cron_jobs if c.name == 'cron:sweep_stale_generating')
    # 1분 주기 (#72에서 5분→1분). 임계(GENERATION_STALE_SECONDS=500)는 살아 있는 요청을
    # 보호하는 값이라 LLM 타임아웃 300초 위에 있어야 해서 못 내린다 — 대신 순수 지연인
    # 주기만 줄여 회복 최악을 ~800초에서 ~560초로 당겼다.
    assert job.minute == set(range(0, 60, 1))
    assert job.unique                                    # 워커 다중화 시 1회 실행 보장
