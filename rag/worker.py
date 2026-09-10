import logging

from arq import cron
from arq.connections import RedisSettings
from config import settings
from database import AsyncSessionLocal
from rag import cache, outbox
# 도메인 스윕과 아래 cron 래퍼가 같은 이름이라 별칭 — cron_jobs엔 래퍼가 등록돼야 한다
from rag.turn_state import sweep_stale_generating as _sweep_generating
from rag.documents import index_pending_document

logger = logging.getLogger(__name__)


async def ping(ctx):
    return "pong"

async def index_document(ctx, document_id: int):
    # 세션·트랜잭션 관리는 index_pending_document 내부가 담당
    # (무거운 청킹·임베딩을 트랜잭션 밖에서 하기 위해 세션을 3분할로 연다)
    await index_pending_document(document_id)

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
    """외부 색인 반영 대기열 드레인 (#139) — 1분 주기.

    라우터·인제스션이 커밋 직후 인라인으로도 드레인하지만(빠른 반영), 그건 최선노력이다.
    프로세스가 죽거나 엔진이 죽어 있으면 행이 남는데, **반영을 보장하는 쪽이 여기다** —
    색인이 서빙의 정본인 구성에서 반영 누락은 곧 낡은 답변이므로 보장이 필요하다.

    search_backend='pg'(기본)면 대기열에 아무것도 안 쌓이므로 이 잡은 매번 0행으로 끝난다.
    """
    async with AsyncSessionLocal() as session:
        r = await outbox.drain(session)
    if r['done'] or r['failed']:
        logger.info('색인 대기열 드레인: 반영 %d · 실패 %d', r['done'], r['failed'])


class WorkerSettings:
    functions = [ping, index_document]  # 워커가 처리할 함수 등록
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
        # 1분 주기 — arq cron의 최소 단위다. 인라인 드레인이 보통 즉시 처리하므로 이건
        # 그게 실패했을 때의 보장선이고, 최악 반영 지연이 1분 + 백오프다.
        cron(drain_search_index_outbox, minute=set(range(0, 60, 1))),
    ]
    redis_settings = RedisSettings.from_dsn(settings.redis_url)
    max_jobs = 10       # 한 워커가 동시 진행하는 잡 수 (asyncio 코루틴 동시성, 스레드 아님). arq 기본값과 동일 — 명시.
    job_timeout = 600   # 대형 문서 임베딩 여유 (기본 300 초과 시 CancelledError로 pending 고착하던 것 완화)
    max_tries = 1       # 실패는 즉시 failed 기록 → 무한/낭비 재시도 방지 (타임아웃 잡은 lazy 스윕이 정리)
