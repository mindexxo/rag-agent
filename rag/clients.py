"""공용 클라이언트 싱글톤 (P1-12).

LLM·TEI 클라이언트를 요청마다 새로 만들면 각자 HTTP 커넥션 풀을 열고 닫지 않아
fd·커넥션이 누적된다(limiter는 자체 Redis 클라를 든 모듈 싱글톤). 여기서 공용
인스턴스를 만들어 요청·백그라운드 태스크가 재사용한다.

Redis 클라이언트는 여기 없다 — 답변 캐시의 Redis exact 계층이 제거되면서(#16) 유일한
용도가 사라졌고, 남은 Redis 사용처는 limiter뿐이라 그쪽 싱글톤이 소유한다(#24).
"""
import httpx

from rag.llm import LlmClient

# 공용 LLM 클라이언트 (stateless — AsyncOpenAI는 동시 재사용 설계)
shared_llm = LlmClient()

# TEI(임베딩·리랭커) 비동기 HTTP 공용 클라이언트 — 커넥션 풀 재사용.
# 타임아웃은 호출별로 다르므로(embed vs rerank) 요청 시점에 지정한다.
# 주의: 커넥션이 이벤트 루프에 묶이므로 프로세스당 루프 1개 전제(앱·워커 각자 import로 충족).
#       ⚠️ 테스트처럼 루프를 갈아끼우는 환경 대책은 아직 없음 — http_async를 타는 통합 테스트를
#       만들 때 conftest에서 테스트별 aclose/재생성 필수 (DB 풀 dispose 패턴과 동일. 미구현 상태).
http_async = httpx.AsyncClient()
