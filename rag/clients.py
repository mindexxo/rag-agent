"""공용 클라이언트 싱글톤 (P1-12).

AnswerCache·LlmClient를 요청마다 새로 만들면 각자 Redis/HTTP 커넥션 풀을 열고 닫지
않아 fd·커넥션이 누적된다(limiter는 이미 모듈 싱글톤). 여기서 공용 인스턴스를 만들어
요청·백그라운드 태스크가 재사용한다. (limiter 자체 클라와의 완전 통합 = QUALITY-4는 별도.)
"""
import httpx
import redis.asyncio as aioredis

from config import settings
from rag.llm import LlmClient

# AnswerCache 공용 Redis (decode_responses=True — 캐시는 str 값을 다룸)
cache_redis = aioredis.from_url(settings.redis_url, decode_responses=True)

# 공용 LLM 클라이언트 (stateless — AsyncOpenAI는 동시 재사용 설계)
shared_llm = LlmClient()

# TEI(임베딩·리랭커) 비동기 HTTP 공용 클라이언트 — 커넥션 풀 재사용.
# 타임아웃은 호출별로 다르므로(embed vs rerank) 요청 시점에 지정한다.
# 주의: 커넥션이 이벤트 루프에 묶이므로 프로세스당 루프 1개 전제(앱·워커 각자 import로 충족).
#       ⚠️ 테스트처럼 루프를 갈아끼우는 환경 대책은 아직 없음 — http_async를 타는 통합 테스트를
#       만들 때 conftest에서 테스트별 aclose/재생성 필수 (DB 풀 dispose 패턴과 동일. 미구현 상태).
http_async = httpx.AsyncClient()
