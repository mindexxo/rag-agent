"""DB 커넥션 풀 설정 계약 (#201) — DB 없이 도는 순수 테스트.

풀 크기는 조용히 틀리는 종류의 설정이다. 작으면 부하가 올랐을 때만 `QueuePool limit`으로
터지고, 평상시엔 아무 증상이 없다. 실제로 #114 부하 실측에서 풀 30으로 동시 128을 돌려
**371건**이 났고, 그때까지 아무도 몰랐다.

값의 근거(실측): 앱 동시 192에서 DB 커넥션 피크 126 = 턴당 0.66개. 한 턴이 세션 2개를
열지만(요청 + SSE 배경 태스크) 요청 세션이 먼저 반납돼 동시 점유는 2보다 작다.
→ 동시 40에 약 26개, 동시 60에 약 40개 필요. 총 100은 거기에 2.5배 여유다.
"""
import pytest


class TestPoolSizing:
    def test_풀이_설정에서_온다(self):
        """하드코딩이면 환경별 조정이 안 된다 — 측정할 때마다 코드를 고쳐야 했다."""
        import inspect

        import database
        src = inspect.getsource(database)
        assert 'settings.db_pool_size' in src, '풀 크기가 설정에서 오지 않는다'
        assert 'settings.db_max_overflow' in src, '오버플로가 설정에서 오지 않는다'

    def test_총_상한이_실측_요구를_덮는다(self):
        """서비스 기준 동시 40에 약 26개가 필요하다(턴당 0.66 실측).
        여유를 보더라도 총 상한이 그보다 한참 커야 한다."""
        from config import settings
        total = settings.db_pool_size + settings.db_max_overflow
        assert total >= 60, (
            f'총 {total} — 동시 40(약 26개)에 여유가 없다. '
            f'부하가 오르면 QueuePool limit으로 터진다')

    def test_엔진에_실제로_반영된다(self):
        """설정만 바꾸고 엔진에 안 꽂히는 경우를 막는다 — 소스 문자열이 아니라 객체로 본다."""
        from config import settings
        from database import engine
        assert engine.pool.size() == settings.db_pool_size
        assert engine.pool._max_overflow == settings.db_max_overflow

    def test_서버_상한을_넘지_않는_범위인가(self):
        """공용 개발계라 앱·워커·사람 세션이 PostgreSQL max_connections를 나눠 쓴다.
        앱 하나가 전부 가져가면 안 된다. 서버는 1000이고 여기선 그 1/4을 넘지 않게 둔다."""
        from config import settings
        total = settings.db_pool_size + settings.db_max_overflow
        assert total <= 250, f'총 {total} — 공용 DB에서 앱 한 프로세스가 너무 많이 쥔다'
