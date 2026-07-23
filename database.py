"""DB 연결 + 세션 팩토리 모듈.

- engine             : PostgreSQL async connection pool (앱 전체에서 공유하는 1개)
- AsyncSessionLocal  : 세션 객체를 찍어내는 팩토리. AsyncSessionLocal() -> AsyncSession
- get_session()      : FastAPI 의존성. 요청 1건당 세션 1개 제공, 종료 시 풀로 반환

쿼리는 세션을 통해 실행. 트랜잭션 경계는 호출자가 명시 (`async with session.begin():`).
"""
from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from config import settings

# connection pool 객체
engine = create_async_engine(
    settings.database_url,
    # 커넥션 startup 시 search_path 지정 — 모델/raw SQL 모두 스키마 미명시라 여기서 일괄 해소.
    # DB role 설정(ALTER ROLE ... SET search_path)에 의존하지 않게 코드에서 못박음.
    connect_args={"server_settings": {"search_path": settings.db_search_path}},
    pool_pre_ping=True,   # 대여 전 죽은 커넥션 감지·교체 — idle 후 "connection is closed" 500 해소 (P1-9)
    pool_recycle=1800,    # 30분 넘은 커넥션은 선제 재생성 (DB/네트워크 idle 종료 대비)
    pool_size=10,         # SSE가 생성 완료까지 세션 점유(생성당 요청+태스크 2커넥션) → 기본 5보다 여유
    max_overflow=20,      # 순간 초과 허용 (총 상한 30)
)

# 세션 객체를 생성하는 팩토리, AsyncSessionLocal() -> AsyncSession 반환
# expire_on_commit=False -> commit 후에도 객체 속성을 유지하여 commit 후 불필요한 I/O 를 방지
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)

# FastAPI의 Depends(get_session)로 라우트에 주입
# 왜 yield?: 라우트 진입전 해당 메서드 호출, 라우트 끝난 후  정리
async def get_session() -> AsyncIterator[AsyncSession]:
    async with AsyncSessionLocal() as session:
        yield session