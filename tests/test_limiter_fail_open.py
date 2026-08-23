"""리미터의 Redis 장애 방침 — fail-open (2026-08-23 결정).

이전엔 예외가 그대로 전파돼 `/kms/query`가 500을 뱉었다. 상담 중단을 더 큰 손실로 보고
"상한 없이 통과"로 뒤집었다. 근거와 대가는 rag/limiter.py의 모듈 docstring에 있다.

여기서 지키려는 것은 두 가지다.
  ① Redis가 죽어도 질문이 받아들여진다 (본 결정)
  ② 그 뒤 반납도 요청을 깨뜨리지 않는다 — 획득만 fail-open이면 반쪽이다
"""
import pytest

from rag import limiter
from rag.limiter import Lease


class _BrokenRedis:
    """어떤 호출이든 터지는 Redis. 커넥션 거부·순단을 대표한다."""

    def __init__(self) -> None:
        self.calls = 0

    def __getattr__(self, _name):
        async def _boom(*_a, **_kw):
            self.calls += 1
            raise ConnectionError('Error 61 connecting to redis. Connection refused.')
        return _boom


@pytest.fixture
def broken_redis(monkeypatch):
    from rag import clients
    broken = _BrokenRedis()
    monkeypatch.setattr(clients, 'shared_redis', broken)
    return broken


@pytest.mark.asyncio
async def test_Redis가_죽으면_상한_없이_통과한다(broken_redis):
    lease = await limiter.try_acquire('t1', tenant_limit=1, user_id='u1', user_limit=1)

    assert lease is not None, '거절(None)이 아니라 통과해야 한다 — fail-open'
    assert lease.degraded is True
    assert lease.tenant_id == 't1' and lease.user_id == 'u1'


@pytest.mark.asyncio
async def test_상한이_1이어도_degraded면_계속_통과한다(broken_redis):
    """"무제한 통과"가 말 그대로임을 못박는다 — 이 대가를 알고 고른 방침이다."""
    leases = [await limiter.try_acquire('t1', tenant_limit=1) for _ in range(5)]

    assert all(x is not None and x.degraded for x in leases)


@pytest.mark.asyncio
async def test_degraded_lease_반납은_Redis를_건드리지_않는다(broken_redis):
    """zset에 등록된 적이 없으니 지울 것도 없다. 헛된 왕복·헛된 에러 로그를 만들지 않는다."""
    lease = await limiter.try_acquire('t1', tenant_limit=1)
    before = broken_redis.calls

    await limiter.release(lease)

    assert broken_redis.calls == before


@pytest.mark.asyncio
async def test_획득_후_Redis가_죽어도_반납이_요청을_깨뜨리지_않는다(broken_redis):
    """정상 획득한 lease의 반납 실패는 삼킨다.

    여기서 예외가 나가면 concurrency_guard의 finally(정상 응답이 500으로 뒤집힘)와
    _run_generation의 finally(root_span.end() 유실)에서 터진다 — 답변을 다 만들어 놓고
    뒷정리 실패로 요청을 깨는 셈이다. 유출된 토큰은 ZSET score 기반 prune이 회수한다.
    """
    healthy_lease = Lease(tenant_id='t1', token='tok', user_id='u1')   # degraded 아님

    await limiter.release(healthy_lease)      # 예외가 새어나오면 실패

    assert broken_redis.calls > 0, '실제로 Redis를 시도하긴 해야 한다 (조용히 건너뛰면 안 됨)'


@pytest.mark.asyncio
async def test_정상_Redis에서는_상한이_그대로_동작한다(tenant_id):
    """fail-open이 평시 동작을 무르게 만들지 않았는지 — 이게 무너지면 방어가 사라진 것이다."""
    first = await limiter.try_acquire(tenant_id, tenant_limit=1)
    second = await limiter.try_acquire(tenant_id, tenant_limit=1)

    assert first is not None and first.degraded is False
    assert second is None, '상한 초과는 여전히 거절돼야 한다'

    await limiter.release(first)
    third = await limiter.try_acquire(tenant_id, tenant_limit=1)
    assert third is not None, '반납 후에는 다시 통과'
    await limiter.release(third)
