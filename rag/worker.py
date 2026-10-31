"""arq 워커 — 색인 대기열 드레인·고착 턴 회수·캐시 청소 cron과 워커 지표 노출."""

import logging

from arq import cron
from arq.connections import RedisSettings
from prometheus_client import start_http_server

from config import settings
from database import AsyncSessionLocal
from rag import outbox
from rag.search import os_client
from rag.turn import cache

# 도메인 스윕과 아래 cron 래퍼가 같은 이름이라 별칭 — cron_jobs엔 래퍼가 등록돼야 한다
from rag.turn.turn_state import sweep_stale_generating as _sweep_generating

logger = logging.getLogger(__name__)


async def ping(ctx):
    return "pong"


def _start_metrics_server() -> None:
    """워커 프로세스 지표 HTTP 서버를 띄운다(arq엔 HTTP 서버가 없다). worker_metrics_port가 0이면 안 띄운다.

    프로세스 지표는 /proc를 읽으므로 Linux에서만 값이 나온다. 기동 실패는 로그만 남긴다 — 관측 장치가 색인을 멈추지 않는다.
    """
    port = settings.worker_metrics_port
    if not port:
        return
    host = settings.worker_metrics_host
    try:
        # 데몬 스레드라 이벤트 루프의 잡 처리를 막지 않는다.
        start_http_server(port, addr=host)
    except OSError as exc:
        logger.error('워커 지표 서버 기동 실패 (%s:%d) — 색인은 계속한다: %s', host, port, exc)
    else:
        logger.info('워커 지표 서버 기동 — %s:%d/metrics', host, port)


async def startup(ctx):
    """워커 지표 노출 + 검색 인덱스 보장.

    워커가 웹보다 먼저 떠 없는 인덱스에 색인하면 동적 매핑으로 굳으므로 먼저 보장한다. 엔진에 못 붙어도 기동은 한다.
    """
    _start_metrics_server()
    await os_client.ensure_index_soft()


async def shutdown(ctx):
    await os_client.close_client()


async def sweep_stale_cache(ctx):
    """미히트 캐시 청소 — cache_retention_days 지난 행 삭제. 일 1회."""
    async with AsyncSessionLocal() as session:
        deleted = await cache.sweep_stale(session)
        await session.commit()
    if deleted:
        logger.info('미히트 캐시 %d행 삭제 (보존 %d일)', deleted, settings.cache_retention_days)


async def sweep_stale_generating(ctx):
    """고착 generating 턴 회수 — 웹 프로세스가 생성 도중 죽어 남은 자리표시를 생성과 무관한 워커가 정리한다."""
    async with AsyncSessionLocal() as session:
        swept = await _sweep_generating(session)
        await session.commit()
    if swept:
        logger.info('고착 generating %d행을 failed로 정리', swept)


async def drain_search_index_outbox(ctx):
    """색인 대기열 드레인 — 인제스션을 포함한 **모든 색인 작업의 유일한 처리 경로**(규약은 rag/outbox.py)."""
    r = await outbox.drain_once()
    if r.get('done') or r.get('failed'):
        logger.info('색인 대기열: 반영 %d · 실패 %d', r['done'], r['failed'])


class WorkerSettings:
    functions = [ping]  # 인제스션은 cron drain이 outbox에서 꺼내 처리한다
    on_startup = startup
    on_shutdown = shutdown
    cron_jobs = [
        cron(sweep_stale_cache, hour=19, minute=30),  # arq는 UTC — 19:30 UTC = KST 새벽 4:30
        # arq cron은 unique=True(기본) + 시각 기반 job_id라 워커가 여러 대여도 1회만 돈다.
        # 회복 지연은 줄이려면 주기를 줄여라 — 임계(GENERATION_STALE_SECONDS)는 LLM 타임아웃보다 커야
        # 살아 있는 요청을 failed로 선고하지 않는다.
        # 스캔은 status='generating' 부분 인덱스(schema.sql)가 받친다.
        cron(sweep_stale_generating, minute=set(range(0, 60, 1))),
        # 1분 주기 — arq cron의 최소 단위.
        cron(drain_search_index_outbox, minute=set(range(0, 60, 1))),
    ]
    redis_settings = RedisSettings.from_dsn(settings.redis_url)
    max_jobs = 10  # 동시 진행 잡 수(코루틴 동시성). arq 기본값과 같다.
    job_timeout = outbox.DRAIN_ROUND_SECONDS  # drain 한 회차 상한 — outbox._fits_round가 이 예산 안에서만 행을 시작한다
    max_tries = 1  # cron 잡 자체의 재시도 없음 — 행 단위 재시도는 outbox.attempts가 담당
