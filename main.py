"""KMS 상담 지식 어시스턴트 — FastAPI 앱 부트스트랩.

라우터 조립 + CORS만 담당한다. 도메인 로직은 rag/, 라우팅은 routers/.
(과거 STT/화자분리/요약 실험 코드는 2026-07-18 제거, 테스트 UI(/ui 정적 서빙)는
 2026-07-23 제거 — FE가 별도 앱으로 분리됨. 필요 시 git 이전 백업 참조.)
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from rag.otel import init_tracing
from routers.conversations import router as conversation_router
from routers.documents import router as document_router
from routers.faqs import router as faq_router
from routers.folders import router as folder_router
from routers.kms import router as kms_router
from routers.stats import router as stats_router

init_tracing()   # OTel(#7) — otel_endpoint 미설정이면 no-op

app = FastAPI()

# CORS — FE(React)가 별도 origin에서 API를 부를 때 브라우저 차단 해제 (게이트웨이 경유 시엔 불발동).
# X-Tenant-Id 커스텀 헤더 때문에 preflight(OPTIONS)가 항상 발생 → allow_headers 필수.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.cors_allow_origins.split(',') if o.strip()],
    allow_methods=['*'],
    allow_headers=['*'],
)

app.include_router(kms_router)
app.include_router(document_router)
app.include_router(conversation_router)
app.include_router(folder_router)
app.include_router(faq_router)
app.include_router(stats_router)
