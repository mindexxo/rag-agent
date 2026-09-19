"""Prometheus 앱 지표 (#129) — vLLM·TEI 자체 /metrics와의 역할 분담이 정의의 기준이다.

vLLM(worker15)·TEI는 서버 레벨 지표(vllm:time_to_first_token_seconds, kv_cache_usage_perc,
prefix_cache_* 등)를 이미 직접 노출한다 — Prometheus가 그쪽을 각자 스크레이프하므로 여기에
중복 정의하지 않는다. 이 모듈에는 **서버가 볼 수 없는 앱 관점 값**만 둔다:

  TTFT_SECONDS          체감 TTFT — HTTP 요청 도착(t_request, routers/kms.py)부터 첫 델타
                        토큰까지. prepare()(인텐트·condense·검색·리랭크·큐잉)를 포함하므로
                        vLLM 자체 TTFT보다 항상 크다 — 그 차이가 RAG 파이프라인 비용이다.
  SEARCH_INDEX_SYNC_TOTAL  외부 검색 색인(#139) 동기화 결과. 이중 쓰기는 **실패해도 예외를
                        올리지 않는다**(PG는 이미 커밋됐고, 여기서 500을 내면 OpenSearch
                        장애가 업로드 장애가 된다) — 그 대가로 실패가 조용해지므로 여기서
                        드러낸다. result='error'가 늘면 색인이 낡고 있다는 뜻이고, 복구는
                        `python -m rag.os_reconcile --apply`다.
  INDEX_DURATION_SECONDS  문서 1건 인제스션 소요 — 단계별(parse·embed·index·total)·형식별.
                        **성공한 건만** 기록된다: 실패 건수는 위 SEARCH_INDEX_SYNC_TOTAL의
                        op='index_document', result='error'가 이미 세므로 중복 정의하지 않는다.
                        파싱은 형식마다 성격이 완전히 다르다(PDF만 레이아웃 ML, #143) —
                        ext 라벨이 없으면 PDF의 초 단위가 md·txt의 밀리초에 묻혀 p95가 무의미해진다.
  INDEX_TOTAL           문서 인제스션 결과 — **문서 단위**. 위 SEARCH_INDEX_SYNC_TOTAL도 실패를
                        세지만 그건 **재시도 단위**라(한 문서가 5번 재시도되면 error 5) 실패율의
                        분모로 쓸 수 없다. result는 셋: ok(성공) / retry(실패, 재시도 남음) /
                        failed(MAX_ATTEMPTS 도달 — 문서가 failed 확정, 사용자가 FE에서 본다).
                        최종 실패율 = failed / (ok + failed). retry만 느는 것은 "버티는 중" 신호다.
  INDEX_PICTURE_PLACEHOLDER_TOTAL
                        캡션이 안 붙어 자리표시로 남은 그림 수(#162). **docling이 VLM 실패를
                        예외 없이 삼키므로 이것이 유일한 관측 창이다** — 이 값이 늘면 VLM이
                        죽었거나 느린 것이다. 성공한 캡션은 셀 수 없다
                        (일반 본문과 구분되지 않게 들어간다) — 성공률이 아니라 실패 신호다.
  FINISH_REASON_TOTAL   vLLM 스트림 종료 사유. 'length'가 늘면 max_tokens에 잘렸다는 뜻 —
                        출처 꼬리까지 잘릴 위험 신호(kms.tail_truncated 스팬 속성과 같은
                        문제를 집계 축으로 본다). 완주한 스트림만 집계된다 — 취소·예외
                        중단은 TurnStatus 축(messages.status)이 담당하므로 여기 안 넣는다.

라벨 규율: route(knowledge|other 등 PreparedRag.route 어휘)·reason(stop|length)·
stage·ext(확장자 6종으로 고정 — _ext가 그 밖을 'other'로 접는다)만.
tenant_id·conversation_id·filename 같은 고카디널리티 값 금지(Prometheus 카디널리티 폭발).
테넌트별 비즈니스 지표는 routers/stats.py(/kms/stats, DB 집계) 소관으로 이미 분리돼 있다.

레지스트리는 prometheus_client 기본 전역 REGISTRY — config.settings·otel._tracer와 같은
"모듈 import=1회 등록" 싱글톤 관례. uvicorn 단일 프로세스 전제(현 docker-compose.yml,
--workers 미지정)다. 멀티프로세스로 바뀌면 프로세스별 REGISTRY가 갈라져 과소집계된다 —
그때는 prometheus_client 멀티프로세스 모드(PROMETHEUS_MULTIPROC_DIR)로 재설계할 것.

이 모듈은 0층(다른 rag/ 모듈 import 금지). 기록 주체는 **두 곳**이고 프로세스가 서로 다르다:
  - 웹 프로세스: rag/streaming.py(진입점) — TTFT·finish_reason.
    llm.py(0층)는 이 모듈을 모르고 콜백으로 값만 올려보낸다(계층 규칙, llm.astream docstring).
  - 워커 프로세스: rag/documents.py(인제스션)·rag/outbox.py(대기열) — 인제스션 소요·동기화 결과.
레지스트리는 프로세스마다 따로이므로 **지표도 프로세스별로 갈라져 노출된다** — 인제스션 지표는
워커의 /metrics(WORKER_METRICS_PORT, #151)에만, TTFT는 앱의 /metrics(main.py)에만 나온다.
프로메테우스가 두 타깃(kms_app·kms_worker)을 각각 긁어 합친다.
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

_EXT_KNOWN = ('pdf', 'docx', 'xlsx', 'md', 'txt')


def ext_label(filename: str) -> str:
    """파일명 → ext 라벨. 아는 확장자 5종 + 'other'로 접어 카디널리티를 고정한다."""
    ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
    return ext if ext in _EXT_KNOWN else 'other'


INDEX_DURATION_SECONDS = Histogram(
    'kms_index_duration_seconds',
    '문서 1건 인제스션 소요(초) — 단계별·형식별 (성공 건만)',
    labelnames=('stage', 'ext'),      # stage=parse|embed|index|total
    # 상한 300초는 docling_document_timeout_seconds와 같은 눈금이다 — 그 위는 어차피 실패로 끝난다.
    # 아래쪽이 촘촘한 이유: md·txt·docx는 밀리초대라 1초 버킷만 있으면 전부 첫 버킷에 뭉친다.
    buckets=(0.1, 0.25, 0.5, 1, 2, 4, 8, 15, 30, 60, 120, 300),
)

INDEX_TOTAL = Counter(
    'kms_index_total',
    '문서 인제스션 결과 (문서 단위 — 재시도 단위인 kms_search_index_sync_total과 다르다)',
    labelnames=('ext', 'result'),     # result=ok|retry|failed
)

INDEX_PICTURE_PLACEHOLDER_TOTAL = Counter(
    'kms_index_picture_placeholder_total',
    '캡션이 안 붙어 자리표시로 남은 그림 수 (#162) — 늘면 VLM에 못 닿는다는 신호',
    labelnames=('ext',),
)

SEARCH_INDEX_SYNC_TOTAL = Counter(
    'kms_search_index_sync_total',
    '외부 검색 색인 동기화 (#139), 재시도 단위. error가 늘면 색인이 낡는다 — rag.os_reconcile로 복구',
    labelnames=('op', 'result'),      # op=rag/outbox.py의 op 어휘 (index_document|drop_documents|…)
)                                     # result=ok|error  (라벨 카디널리티 고정 — 규율 참조)

SEARCH_INDEX_FAILED_TOTAL = Counter(
    'kms_search_index_failed_total',
    'outbox 행이 failed로 확정된 수 (#185, 확정 단위 — 위 error는 재시도마다 오른다). '
    '0이 아니면 PG↔엔진 불일치가 사람 손 없이는 안 맞는 상태다 — 알람은 rag/outbox.py _on_failed에서',
    labelnames=('op',),
)
