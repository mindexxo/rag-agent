from arq.connections import RedisSettings
from config import settings
from rag.documents import index_pending_document


async def ping(ctx):
    return "pong"

async def index_document(ctx, document_id: int):
    # 세션·트랜잭션 관리는 index_pending_document 내부가 담당
    # (무거운 청킹·임베딩을 트랜잭션 밖에서 하기 위해 세션을 3분할로 연다)
    await index_pending_document(document_id)

class WorkerSettings:
    functions = [ping, index_document]  # 워커가 처리할 함수 등록
    redis_settings = RedisSettings.from_dsn(settings.redis_url)
    max_jobs = 10       # 한 워커가 동시 진행하는 잡 수 (asyncio 코루틴 동시성, 스레드 아님). arq 기본값과 동일 — 명시.
    job_timeout = 600   # 대형 문서 임베딩 여유 (기본 300 초과 시 CancelledError로 pending 고착하던 것 완화)
    max_tries = 1       # 실패는 즉시 failed 기록 → 무한/낭비 재시도 방지 (타임아웃 잡은 lazy 스윕이 정리)
