"""동시 in-flight 제한 (F6) — Redis Sorted Set, 테넌트 + 선택적 사용자 이중 제한

테넌트별/사용자별 "진행 중 요청 수"를 세서 GPU 병목을 앱 레벨에서 방어한다.
LLM 스트리밍은 응답이 수십 초씩 걸려 분당 빈도(RPM)보다 동시 점유 수가 실제 부하를 반영.

카운터(단일 숫자) 대신 ZSET(멤버=요청 토큰, score=시작 시각)을 쓰는 이유(P1-11):
크래시로 release가 누락돼 새어나간 항목을, 숫자 카운터는 "유출분 vs 진행 중"을 구분 못 해
상한을 영구 잠식했다. ZSET은 시작 시각으로 오래된(=죽은) 항목만 개별 prune할 수 있어
진행 중은 안 건드리고 유출만 회수한다. Redis라 다중 워커/인스턴스에서 상한 공유.

이중 제한: 테넌트 zset(공유 GPU 보호) + 사용자 zset(테넌트 내 공정성). 같은 토큰을 두
zset에 넣어 release 한 번에 정리. user_id=None(헤더 없음)이면 테넌트만 적용 — 인증 도입
전까지는 X-User-Id 미전송 시 사실상 테넌트 단일 제한으로 동작한다(의도된 현행).

사용자 상한이 0/None으로 들어오면 config 기본값으로 폴백한다(#24). 이전엔 falsy 값이
사용자 zset 자체를 건너뛰게 해서 "0 = 무제한"으로 동작했다 — 상한을 조이려는 설정이
반대로 제한을 통째로 없애는 방향이라 폴백으로 바꿨다.

acquire는 Lua 스크립트로 prune+count+조건부 add를 서버측에서 원자적으로 1왕복 처리한다
(왕복 수 절감 + 체크↔추가 race 제거).
"""
import time
import uuid
from dataclasses import dataclass

import redis.asyncio as aioredis

from config import settings

_T_PREFIX = 'kms:inflight:t:'   # 테넌트별
_U_PREFIX = 'kms:inflight:u:'   # 사용자별 (tenant:user)

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
    """획득한 슬롯 1개. release에 tenant_id·token·user_id 세 값이 항상 함께 필요해 묶었다.

    handed_off는 "반납 책임이 요청 핸들러에서 백그라운드 태스크로 넘어갔는가"다. 생성 경로는
    응답을 반환한 뒤에도 태스크가 GPU를 물고 있으므로, 요청 종료 시점에 반납하면 안 된다.
    True가 되면 요청 쪽(concurrency_guard의 finally)은 반납을 건너뛰고 태스크가 담당한다 —
    이 플래그 하나로 "정확히 한 번 반납"을 지킨다.
    """
    tenant_id: str
    token: str
    user_id: str | None = None
    handed_off: bool = False


class ConcurrencyLimiter:
    def __init__(self, redis_url: str):
        # from_url은 lazy — 첫 명령 때 연결된다
        self._redis = aioredis.from_url(redis_url)

    async def try_acquire(
            self, tenant_id: str, tenant_limit: int,
            user_id: str | None = None, user_limit: int | None = None,
    ) -> Lease | None:
        """테넌트(+사용자) 상한 모두 미만이면 슬롯을 등록해 Lease 반환, 하나라도 꽉 차면 None.
        Lua로 원자 실행 — prune+판정+등록이 1왕복, race 없음.

        user_limit이 0/None이면 config 기본값을 쓴다 — 사용자 제한을 끄는 경로는 없다
        (끄고 싶으면 user_id를 넘기지 않는다).
        """
        now = time.time()
        token = uuid.uuid4().hex
        keys = [_T_PREFIX + tenant_id]
        if user_id:
            keys.append(_U_PREFIX + f'{tenant_id}:{user_id}')
            user_limit = user_limit or settings.user_concurrency_default
        argv = [now, tenant_limit, user_limit or 0, settings.inflight_max_seconds, token]
        ok = await self._redis.eval(_ACQUIRE_LUA, len(keys), *keys, *argv)
        return Lease(tenant_id=tenant_id, token=token, user_id=user_id) if ok == 1 else None

    async def release(self, lease: Lease | None) -> None:
        """요청/태스크 종료(완료·중단·에러) 시 토큰을 두 zset에서 제거. lease 없으면 no-op.
        zrem은 멱등이라 중복 호출도 안전하다."""
        if lease is None:
            return
        await self._redis.zrem(_T_PREFIX + lease.tenant_id, lease.token)
        if lease.user_id:
            await self._redis.zrem(_U_PREFIX + f'{lease.tenant_id}:{lease.user_id}', lease.token)


# /kms/query 전용 공유 리미터
query_limiter = ConcurrencyLimiter(settings.redis_url)
