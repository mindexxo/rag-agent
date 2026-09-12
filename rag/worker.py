import logging

from arq import cron
from arq.connections import RedisSettings
from config import settings
from database import AsyncSessionLocal
from rag import cache, opensearch, outbox
# 도메인 스윕과 아래 cron 래퍼가 같은 이름이라 별칭 — cron_jobs엔 래퍼가 등록돼야 한다
from rag.turn_state import sweep_stale_generating as _sweep_generating

logger = logging.getLogger(__name__)


async def ping(ctx):
    return "pong"


async def startup(ctx):
    """검색 인덱스 보장(#139) — 워커가 웹보다 먼저 뜨는 배포에서 첫 drain이 없는 인덱스에
    색인하면 동적 매핑(knn_vector 아님)으로 굳는다. 엔진에 못 붙으면 기동 실패가 맞다."""
    await opensearch.ensure_index()


async def shutdown(ctx):
    await opensearch.close_client()

async def sweep_stale_cache(ctx):
    """미히트 캐시 청소(#16) — cache_retention_days(90일) 지난 row 삭제. 일 1회면 충분."""
    async with AsyncSessionLocal() as session:
        deleted = await cache.sweep_stale(session)
        await session.commit()
    if deleted:
        logger.info('미히트 캐시 %d행 삭제 (보존 %d일)', deleted, settings.cache_retention_days)


async def sweep_stale_generating(ctx):
    """고착 generating 턴 회수(#46) — 도메인 함수 호출 + commit. 1분 주기(#72).

    웹 프로세스가 생성 도중 죽으면 asyncio 태스크가 증발해 자리표시만 남는다 — 그 회수를
    생성과 무관하게 살아있는 이 프로세스(워커)가 맡는다. 워커가 죽으면 이 치유도 멈추는데,
    그건 index_document의 pending 고착과 같은 기존 실패 등급이다(새 위험 범주 아님).
    """
    async with AsyncSessionLocal() as session:
        swept = await _sweep_generating(session)
        await session.commit()
    if swept:
        logger.info('고착 generating %d행을 failed로 정리', swept)

async def drain_search_index_outbox(ctx):
    """색인 작업 대기열 드레인 (#139 outbox) — 1분 주기. **인제스션을 포함한 모든 색인 작업의
    유일한 처리 경로다.** 업로드·삭제·토글·FAQ 변경은 라우터가 같은 트랜잭션에 행을 남기고,
    여기서 id 순으로 처리한다. arq 잡 등록(index_document)은 이것으로 대체됐다.

    단일 워커 전제 — 이전 회차가 아직 돌고 있으면(대량 인제스션) drain_once가 즉시 빠진다.
    검색 반영 지연은 최대 1분 + 재시도 — "문서 변경은 1~5분 뒤 반영될 수 있다"가 제품 가이드다.
    """
    r = await outbox.drain_once()
    if r.get('done') or r.get('failed'):
        logger.info('색인 대기열: 반영 %d · 실패 %d', r['done'], r['failed'])


class WorkerSettings:
    functions = [ping]                  # 인제스션은 cron drain이 outbox에서 꺼내 처리한다 (#139)
    on_startup = startup
    on_shutdown = shutdown
    cron_jobs = [
        cron(sweep_stale_cache, hour=19, minute=30),   # arq는 UTC — 19:30 UTC = KST 새벽 4:30
        # 1분 주기 — status='generating' 부분 인덱스(schema.sql #46)가 스캔을 받친다.
        # arq cron은 unique=True(기본) + 시각 기반 job_id라 워커가 여러 대여도 1회만 돈다.
        #
        # 주기만 줄인다(5분→1분, #72). 회복 지연은 '임계 + 주기'인데 둘은 성격이 다르다:
        # 임계(GENERATION_STALE_SECONDS=500)는 **살아 있는 요청을 보호**하는 값이라
        # LLM 호출 타임아웃 300초(rag/llm.py) 위에 있어야 한다 — 내리면 vLLM 큐에 밀린
        # 정상 요청이 먼저 failed로 선고되고 완주 시 done으로 덮여 좀비 플립이 난다.
        # 반면 주기는 순수 지연이라 줄여도 그 위험이 없다. 최악 ~800초 → ~560초.
        cron(sweep_stale_generating, minute=set(range(0, 60, 1))),
        # 1분 주기 — arq cron의 최소 단위. 색인 작업의 유일한 처리 경로다(인라인 없음).
        cron(drain_search_index_outbox, minute=set(range(0, 60, 1))),
    ]
    redis_settings = RedisSettings.from_dsn(settings.redis_url)
    max_jobs = 10       # 한 워커가 동시 진행하는 잡 수 (asyncio 코루틴 동시성, 스레드 아님). arq 기본값과 동일 — 명시.
    job_timeout = 600   # 대형 문서 임베딩 여유 (기본 300 초과 시 CancelledError로 pending 고착하던 것 완화)
    max_tries = 1       # cron 잡 자체의 재시도 없음 — 행 단위 재시도는 outbox.attempts가 담당
