"""동시 in-flight 제한 — Redis ZSET으로 테넌트(공유 GPU 보호) + 사용자(테넌트 내 공정성) 이중 상한.

LLM 스트리밍은 수십 초씩 걸려 분당 빈도보다 동시 점유 수가 실제 부하를 반영한다.
숫자 카운터 대신 ZSET(멤버=요청 토큰, score=시작 시각)을 쓰는 이유: release가 크래시로 누락돼도
오래된 항목만 prune해 유출분을 회수하고 진행 중은 건드리지 않는다.
user_id가 없으면 테넌트 상한만 적용한다.

Redis 장애 시 fail-open — 상한 없이 통과시킨다(상담 중단을 더 큰 손실로 본다). 그 구간엔 GPU 상한이
사라지고 동시성도 기록되지 않는다.
"""

import logging
import time
import uuid
from dataclasses import dataclass

from config import settings

logger = logging.getLogger(__name__)

_T_PREFIX = 'kms:inflight:t:'  # 테넌트별
_U_PREFIX = 'kms:inflight:u:'  # 사용자별 (tenant:user)

# 원자적 acquire. KEYS[1]=테넌트 zset, KEYS[2]=사용자 zset(선택).
# ARGV: 1=now, 2=tenant_limit, 3=user_limit, 4=max_age(sec), 5=token
# 반환: 1=획득, 0=상한 초과.
_ACQUIRE_LUA = """
local now = tonumber(ARGV[1])
local cutoff = now - tonumber(ARGV[4])
redis.call('ZREMRANGEBYSCORE', KEYS[1], '-inf', cutoff)
if redis.call('ZCARD', KEYS[1]) >= tonumber(ARGV[2]) then return 0 end
if #KEYS >= 2 then
  redis.call('ZREMRANGEBYSCORE', KEYS[2], '-inf', cutoff)
  if redis.call('ZCARD', KEYS[2]) >= tonumber(ARGV[3]) then return 0 end
end
redis.call('ZADD', KEYS[1], now, ARGV[5])
redis.call('EXPIRE', KEYS[1], tonumber(ARGV[4]))
if #KEYS >= 2 then
  redis.call('ZADD', KEYS[2], now, ARGV[5])
  redis.call('EXPIRE', KEYS[2], tonumber(ARGV[4]))
end
return 1
"""


@dataclass
class Lease:
    """획득한 슬롯 1개.

    handed_off=True면 반납 책임이 요청 핸들러에서 백그라운드 태스크로 넘어간 것이다 — 요청 쪽
    (concurrency_guard의 finally)은 반납을 건너뛴다. 이 플래그로 "정확히 한 번 반납"을 지킨다.
    """

    tenant_id: str
    token: str
    user_id: str | None = None
    handed_off: bool = False
    # Redis 장애로 상한 검사 없이 통과한 슬롯 (fail-open). zset에 없으므로 반납할 것도 없다.
    degraded: bool = False


def _redis():
    """공용 Redis 클라이언트를 **호출 시점에** 읽는다 — 테스트가 클라이언트를 교체하므로 붙잡아 두면 안 된다.
    지연 import는 동기 스크립트가 이 모듈만 쓸 때 LLM 클라이언트까지 만들어지지 않게 하려는 것."""
    from rag import clients

    return clients.shared_redis


async def try_acquire(
    tenant_id: str,
    tenant_limit: int,
    user_id: str | None = None,
    user_limit: int | None = None,
) -> Lease | None:
    """상한이 모두 미만이면 슬롯을 등록해 Lease, 하나라도 꽉 차면 None.

    user_limit이 0/None이면 config 기본값으로 폴백한다("0=무제한"이 아니다 — 끄려면 user_id를 넘기지 않는다).
    Redis에 못 붙으면 Lease(degraded=True)를 준다(fail-open).
    """
    now = time.time()
    token = uuid.uuid4().hex
    keys = [_T_PREFIX + tenant_id]
    if user_id:
        keys.append(_U_PREFIX + f'{tenant_id}:{user_id}')
        user_limit = user_limit or settings.user_concurrency_default
    argv = [now, tenant_limit, user_limit or 0, settings.inflight_max_seconds, token]
    try:
        ok = await _redis().eval(_ACQUIRE_LUA, len(keys), *keys, *argv)
    except Exception:
        logger.warning('리미터 Redis 실패 — 상한 없이 통과 (tenant=%s, user=%s)', tenant_id, user_id)
        return Lease(tenant_id=tenant_id, token=token, user_id=user_id, degraded=True)
    return Lease(tenant_id=tenant_id, token=token, user_id=user_id) if ok == 1 else None


async def release(lease: Lease | None) -> None:
    """토큰을 두 zset에서 제거한다(멱등).

    Redis 실패를 삼킨다 — 뒷정리 실패로 정상 응답이 500이 되거나 루트 스팬 종료가 유실되지 않게.
    삼켜도 남은 토큰은 inflight_max_seconds 뒤 prune이 회수한다.
    """
    if lease is None or lease.degraded:
        return
    try:
        await _redis().zrem(_T_PREFIX + lease.tenant_id, lease.token)
        if lease.user_id:
            await _redis().zrem(_U_PREFIX + f'{lease.tenant_id}:{lease.user_id}', lease.token)
    except Exception:
        logger.warning(
            '리미터 반납 실패 — prune(%ds)에 맡긴다 (tenant=%s)', settings.inflight_max_seconds, lease.tenant_id
        )
