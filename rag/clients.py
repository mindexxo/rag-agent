"""공용 클라이언트 싱글톤 (P1-12).

LLM·TEI 클라이언트를 요청마다 새로 만들면 각자 HTTP 커넥션 풀을 열고 닫지 않아
fd·커넥션이 누적된다(limiter는 자체 Redis 클라를 든 모듈 싱글톤). 여기서 공용
인스턴스를 만들어 요청·백그라운드 태스크가 재사용한다.

Redis도 여기 있다(#30). 답변 캐시의 exact 계층이 제거되며(#16) 사용처가 limiter 하나만
남았을 땐 그쪽이 자기 클라이언트를 들었지만(#24), 취소 신호 pub/sub이 두 번째 사용처가
되면서 그 전제가 깨졌다 — 사용처마다 클라이언트를 만들면 타임아웃·재시도·TLS 같은 설정과
테스트 정리 지점이 사용처 수만큼 늘어난다.

**소비자는 이 모듈의 속성을 호출 시점에 읽어야 한다**(`clients.shared_redis.publish(...)`).
생성자나 import 시점에 붙잡으면 아래 http_async와 같은 이유로 교체가 먹지 않는다.
"""
import httpx
import redis.asyncio as aioredis

from config import settings
from rag.llm import LlmClient

# 공용 LLM 클라이언트 (stateless — AsyncOpenAI는 동시 재사용 설계)
shared_llm = LlmClient()

# TEI(임베딩·리랭커) 비동기 HTTP 공용 클라이언트 — 커넥션 풀 재사용.
# 타임아웃은 호출별로 다르므로(embed vs rerank) 요청 시점에 지정한다.
# 주의: 커넥션이 이벤트 루프에 묶이므로 프로세스당 루프 1개 전제(앱·워커 각자 import로 충족).
#       ⚠️ 테스트처럼 루프를 갈아끼우는 환경 대책은 아직 없음 — http_async를 타는 통합 테스트를
#       만들 때 conftest에서 테스트별 aclose/재생성 필수 (DB 풀 dispose 패턴과 동일. 미구현 상태).
http_async = httpx.AsyncClient()

# 공용 Redis — 동시성 제한(ZSET)과 취소 신호(pub/sub)가 함께 쓴다.
# from_url은 lazy라 첫 명령 때 연결된다. pub/sub 구독은 풀에서 커넥션 하나를 따로 물지만
# 일반 명령은 다른 커넥션으로 병행되므로 공유해도 서로를 막지 않는다.
shared_redis = aioredis.from_url(settings.redis_url)
