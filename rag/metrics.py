"""Prometheus 앱 지표 (#129) — vLLM·TEI 자체 /metrics와의 역할 분담이 정의의 기준이다.

vLLM(worker15)·TEI는 서버 레벨 지표(vllm:time_to_first_token_seconds, kv_cache_usage_perc,
prefix_cache_* 등)를 이미 직접 노출한다 — Prometheus가 그쪽을 각자 스크레이프하므로 여기에
중복 정의하지 않는다. 이 모듈에는 **서버가 볼 수 없는 앱 관점 값**만 둔다:

  TTFT_SECONDS          체감 TTFT — HTTP 요청 도착(t_request, routers/kms.py)부터 첫 델타
                        토큰까지. prepare()(인텐트·condense·검색·리랭크·큐잉)를 포함하므로
                        vLLM 자체 TTFT보다 항상 크다 — 그 차이가 RAG 파이프라인 비용이다.
  FINISH_REASON_TOTAL   vLLM 스트림 종료 사유. 'length'가 늘면 max_tokens에 잘렸다는 뜻 —
                        출처 꼬리까지 잘릴 위험 신호(kms.tail_truncated 스팬 속성과 같은
                        문제를 집계 축으로 본다). 완주한 스트림만 집계된다 — 취소·예외
                        중단은 TurnStatus 축(messages.status)이 담당하므로 여기 안 넣는다.

라벨 규율: route(knowledge|other 등 PreparedRag.route 어휘)·reason(stop|length)만 —
tenant_id·conversation_id 같은 고카디널리티 값 금지(Prometheus 카디널리티 폭발).
테넌트별 비즈니스 지표는 routers/stats.py(/kms/stats, DB 집계) 소관으로 이미 분리돼 있다.

레지스트리는 prometheus_client 기본 전역 REGISTRY — config.settings·otel._tracer와 같은
"모듈 import=1회 등록" 싱글톤 관례. uvicorn 단일 프로세스 전제(현 docker-compose.yml,
--workers 미지정)다. 멀티프로세스로 바뀌면 프로세스별 REGISTRY가 갈라져 과소집계된다 —
그때는 prometheus_client 멀티프로세스 모드(PROMETHEUS_MULTIPROC_DIR)로 재설계할 것.

이 모듈은 0층(다른 rag/ 모듈 import 금지). 기록 주체는 rag/streaming.py(진입점) 한 곳이다 —
llm.py(0층)는 이 모듈을 모르고 콜백으로 값만 올려보낸다(계층 규칙, llm.astream docstring).
"""
from prometheus_client import Counter, Histogram

TTFT_SECONDS = Histogram(
    'kms_ttft_seconds',
    '체감 TTFT(초) — 요청 도착부터 첫 델타 토큰까지 (prepare 포함, vLLM 서버측 TTFT와 다름)',
    labelnames=('route',),
    # 잠정 버킷 — 실측 없이 정한 초기값. 상한 60초는 #101 부하 실측(동시 32에서 p95 63초
    # 붕괴)을 관측 가능하게 덮기 위함. 트래픽이 쌓이면 분포 보고 재조정할 것.
    buckets=(0.25, 0.5, 1, 2, 3, 5, 8, 13, 21, 34, 60),
)

FINISH_REASON_TOTAL = Counter(
    'kms_llm_finish_reason_total',
    'vLLM 스트림 종료 사유 (완주분만 — 취소·예외 중단 제외). length=max_tokens 잘림 경보',
    labelnames=('route', 'reason'),
)
