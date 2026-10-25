"""Prometheus 앱 지표 — vLLM·TEI가 자체 /metrics로 볼 수 없는 앱 관점 값만 둔다.

  TTFT_SECONDS          체감 TTFT — HTTP 요청 도착부터 첫 델타까지. prepare()를 포함하므로
                        vLLM 자체 TTFT와의 차이가 RAG 파이프라인 비용이다.
  SEARCH_INDEX_SYNC_TOTAL  색인 동기화 결과(재시도 단위). 동기화 실패는 예외를 올리지 않아 조용하므로
                        여기서 드러낸다 — error가 늘면 `python -m rag.os_reconcile --apply`로 복구.
  INDEX_DURATION_SECONDS  문서 1건 인제스션 소요(성공 건만). 형식마다 파싱 비용이 크게 달라 ext로 나눈다.
  INDEX_TOTAL           인제스션 결과(문서 단위) — ok / retry(재시도 남음) / failed(확정).
                        최종 실패율 = failed / (ok + failed).
  INDEX_PICTURE_PLACEHOLDER_TOTAL
                        캡션 대상에서 빠져(면적 하한 미만) 자리표시로 남은 그림 수. VLM 실패는 문서 재시도로 간다.
  FINISH_REASON_TOTAL   LLM 스트림 종료 사유(완주분만). 'length'가 늘면 출처 꼬리까지 잘릴 위험 신호.

라벨 규율: tenant_id·conversation_id·filename 같은 고카디널리티 값을 라벨로 쓰지 않는다(카디널리티 폭발).
ext는 ext_label로 고정 집합에 접는다. 테넌트별 비즈니스 지표는 routers/stats.py(DB 집계) 소관이다.

전역 REGISTRY는 프로세스별이다 — 단일 uvicorn 프로세스 전제(멀티프로세스면 PROMETHEUS_MULTIPROC_DIR 모드 필요).
웹(rag/streaming.py: TTFT·finish_reason)과 워커(rag/documents.py·rag/outbox.py: 인제스션·동기화)가
각자 기록하므로 지표는 앱 /metrics와 워커 /metrics(WORKER_METRICS_PORT)에 나뉘어 노출된다.
"""

from prometheus_client import Counter, Histogram

TTFT_SECONDS = Histogram(
    'kms_ttft_seconds',
    '체감 TTFT(초) — 요청 도착부터 첫 델타 토큰까지 (prepare 포함, vLLM 서버측 TTFT와 다름)',
    labelnames=('route',),
    # 잠정 버킷 — 상한 60초는 고동시 부하의 p95 붕괴 구간을 덮기 위함. 트래픽이 쌓이면 재조정
    buckets=(0.25, 0.5, 1, 2, 3, 5, 8, 13, 21, 34, 60),
)

FINISH_REASON_TOTAL = Counter(
    'kms_llm_finish_reason_total',
    'vLLM 스트림 종료 사유 (완주분만 — 취소·예외 중단 제외). length=max_tokens 잘림 경보',
    labelnames=('route', 'reason'),
)

_EXT_KNOWN = ('pdf', 'docx', 'xlsx', 'md', 'txt')


def ext_label(filename: str) -> str:
    """아는 확장자 외는 'other'로 접어 ext 라벨 카디널리티를 고정한다."""
    ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
    return ext if ext in _EXT_KNOWN else 'other'


INDEX_DURATION_SECONDS = Histogram(
    'kms_index_duration_seconds',
    '문서 1건 인제스션 소요(초) — 단계별·형식별 (성공 건만)',
    labelnames=('stage', 'ext'),  # stage=parse|embed|index|total
    # 상한 900초 = docling_document_timeout_seconds(그 위는 실패라 기록 안 됨). 아래가 촘촘한 건 md·txt·docx가 밀리초대라서
    buckets=(0.1, 0.25, 0.5, 1, 2, 4, 8, 15, 30, 60, 120, 300, 600, 900),
)

INDEX_TOTAL = Counter(
    'kms_index_total',
    '문서 인제스션 결과 (문서 단위 — 재시도 단위인 kms_search_index_sync_total과 다르다)',
    labelnames=('ext', 'result'),  # result=ok|retry|failed
)

INDEX_PICTURE_PLACEHOLDER_TOTAL = Counter(
    'kms_index_picture_placeholder_total',
    '캡션 대상에서 빠져 자리표시로 남은 그림 수 (#162·#223) — VLM 실패는 문서 재시도로 간다',
    labelnames=('ext',),
)

EVIDENCE_BUDGET_TRIM_TOTAL = Counter(
    'kms_evidence_budget_trim_total',
    '근거 예산 초과로 뺀 산문 청크 수 (#222) — 엑셀 상한 6,000토큰이면 평소 0. 늘면 예산·상한 재검토',
)

SEARCH_INDEX_SYNC_TOTAL = Counter(
    'kms_search_index_sync_total',
    '외부 검색 색인 동기화 (#139), 재시도 단위. error가 늘면 색인이 낡는다 — rag.os_reconcile로 복구',
    labelnames=('op', 'result'),  # op=rag/outbox.py의 op 어휘
)  # result=ok|error

SEARCH_INDEX_FAILED_TOTAL = Counter(
    'kms_search_index_failed_total',
    'outbox 행이 failed로 확정된 수 (#185, 확정 단위 — 위 error는 재시도마다 오른다). '
    '0이 아니면 PG↔엔진 불일치가 사람 손 없이는 안 맞는 상태다 — 알람은 rag/outbox.py _on_failed에서',
    labelnames=('op',),
)
