"""공용 클라이언트 싱글톤(LLM·TEI HTTP·Redis) — 요청마다 만들면 커넥션 풀이 누적된다.

소비자는 이 모듈의 속성을 **호출 시점에** 읽어야 한다(`clients.shared_redis.publish(...)`) —
import 시점에 붙잡으면 테스트의 클라이언트 교체가 먹지 않는다.
"""

import httpx
import redis.asyncio as aioredis

from config import settings
from rag.llm import aux_overridden, create_aux_llm, create_llm

# 공급자는 settings.llm_provider가 정한다. stateless라 동시 재사용 가능
shared_llm = create_llm()
# 보조 호출(의도 분류·질의 재작성·캐시 재사용 판정)용. llm_aux_*가 비면 생성 클라이언트 그 자체
shared_aux_llm = create_aux_llm() if aux_overridden() else shared_llm

# TEI(임베딩·리랭커)용. 타임아웃은 호출마다 달라 요청 시점에 지정한다.
# 커넥션이 이벤트 루프에 묶이므로 프로세스당 루프 1개 전제 — 루프를 갈아끼우는 테스트에서 이걸 타면
# 루프에 묶인다 — 테스트는 tests/conftest.py의 _loop_hygiene가 매번 닫고 새로 만든다.
http_async = httpx.AsyncClient()

# 동시성 제한(ZSET)과 취소 신호(pub/sub)가 함께 쓴다. from_url은 첫 명령 때 연결한다.
# pub/sub 구독은 풀에서 커넥션을 따로 물어 일반 명령을 막지 않는다.
shared_redis = aioredis.from_url(settings.redis_url)
