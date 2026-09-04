"""KMS 상담 지식 어시스턴트 — FastAPI 앱 부트스트랩.

라우터 조립 + CORS + lifespan 배선만 담당한다. 도메인 로직은 rag/, 라우팅은 routers/.
(lifespan은 #30에서 처음 도입 — 이 웹 프로세스의 인메모리 상태를 다루는 백그라운드 태스크
 전용이다. 주기 잡은 rag/worker.py의 arq cron 몫.)
(과거 STT/화자분리/요약 실험 코드는 2026-07-18 제거, 테스트 UI(/ui 정적 서빙)는
 2026-07-23 제거 — FE가 별도 앱으로 분리됨. 필요 시 git 이전 백업 참조.)
"""
import asyncio
import contextlib
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.responses import Response

from config import settings
from rag import cancellation
from rag.otel import init_tracing
from routers.conversations import router as conversation_router
from routers.documents import router as document_router
from routers.faqs import router as faq_router
from routers.folders import router as folder_router
from routers.kms import router as kms_router
from routers.stats import router as stats_router

init_tracing()   # OTel(#7) — otel_endpoint 미설정이면 no-op


@asynccontextmanager
async def lifespan(app: FastAPI):
    """앱 수명 동안 도는 백그라운드 태스크 기동·정리 (#30).

    여기 두는 건 **이 웹 프로세스의 인메모리 상태를 다뤄야 하는 작업**뿐이다 —
    취소 레지스트리(rag/cancellation.py)가 프로세스 로컬이라 그렇다. 주기 스윕(캐시 청소 등)은
    rag/worker.py의 arq cron 몫이고 별 프로세스라 여기와 무관하다.

    주의: 테스트의 httpx ASGITransport는 lifespan을 호출하지 않는다(실측) — 그래서 본문은
    subscribe_forever를 부르는 얇은 배선만 두고, 테스트는 그 함수를 직접 띄워 검증한다.
    """
    subscriber = asyncio.create_task(cancellation.subscribe_forever())
    try:
        yield
    finally:
        subscriber.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await subscriber


app = FastAPI(lifespan=lifespan)

# CORS — FE(React)가 별도 origin에서 API를 부를 때 브라우저 차단 해제 (게이트웨이 경유 시엔 불발동).
# X-Tenant-Id 커스텀 헤더 때문에 preflight(OPTIONS)가 항상 발생 → allow_headers 필수.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.cors_allow_origins.split(',') if o.strip()],
    allow_methods=['*'],
    allow_headers=['*'],
)

# Prometheus 스크레이프 (#129) — 앱 지표(rag/metrics.py: 체감 TTFT·finish_reason)와
# prometheus_client 기본 프로세스 지표를 노출. 인증 없음 — 사내망 개발계 전제, 외부 노출
# 경로가 생기면 접근 제어 재검토. rag/metrics.py는 routers→rag.streaming 경유로 이미
# import되므로 여기서 따로 불러올 필요 없다.
# mount(make_asgi_app()) 대신 직접 라우트인 이유: mount는 /metrics→/metrics/ 307을 낸다(실측).
@app.get('/metrics', include_in_schema=False)
async def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


app.include_router(kms_router)
app.include_router(document_router)
app.include_router(conversation_router)
app.include_router(folder_router)
app.include_router(faq_router)
app.include_router(stats_router)
