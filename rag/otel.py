"""OTel 트레이싱 헬퍼 (#7) — Phoenix(OpenInference) 규격 수동 계측.

설계 원칙:
- otel_endpoint 미설정이면 완전 no-op — TracerProvider를 설치하지 않으므로 기본
  Proxy 트레이서가 NonRecordingSpan만 만든다 (속성 세팅 전 is_recording 가드로
  문자열 조립 비용까지 회피). 운영 기본 무영향.
- 계측 코드는 벤더 중립 — OTel 표준 API + OpenInference '속성 이름 문자열'만 사용
  (openinference 패키지 의존 없음). 백엔드 교체는 endpoint 값 변경으로 끝.
- 전송은 BatchSpanProcessor 백그라운드 스레드 — 수집기(Phoenix) 장애는 스팬 유실로
  끝나고 앱 경로를 막지 않는다.
- SSE 배경 태스크: OTel 컨텍스트는 contextvars 기반이라 asyncio.create_task 시점에
  자동 복사 → 요청에서 연 루트 스팬 아래로 배경 스팬이 붙는다 (별도 전파 코드 없음).

스팬 텍스트 정책(#7 결정): 질문·재작성·변형은 전문, 청크 본문·답변은
otel_text_limit(기본 500자) 절단 — 개발계 진단 가치와 개인정보 노출 면적의 절충.
"""
from contextlib import contextmanager

from opentelemetry import context as otel_context
from opentelemetry import trace

from config import settings

# ── OpenInference 속성 이름 (규격 문자열 — Phoenix 전용 뷰가 이 이름을 렌더링) ──
SPAN_KIND = 'openinference.span.kind'          # CHAIN|LLM|RETRIEVER|RERANKER|GUARDRAIL ...
INPUT_VALUE = 'input.value'
OUTPUT_VALUE = 'output.value'
LLM_MODEL = 'llm.model_name'
RERANK_QUERY = 'reranker.query'
RERANK_TOP_K = 'reranker.top_k'

_tracer = trace.get_tracer('consult-agent')    # Proxy — init 전/미설정이면 no-op 스팬 생성


def init_tracing() -> bool:
    """앱 기동 시 1회 호출. endpoint 미설정이면 아무것도 설치하지 않는다(False)."""
    if not settings.otel_endpoint:
        return False
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    provider = TracerProvider(resource=Resource.create({'service.name': 'consult-agent'}))
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=settings.otel_endpoint)))
    trace.set_tracer_provider(provider)
    return True


def clip(text: str | None) -> str:
    """청크 본문·답변용 절단 — 스팬 텍스트 정책의 단일 적용 지점."""
    if not text:
        return ''
    limit = settings.otel_text_limit
    return text if len(text) <= limit else text[:limit] + '…[절단]'


@contextmanager
def span(name: str, kind: str):
    """OpenInference kind가 달린 현재 스팬 컨텍스트. no-op이면 비용 최소."""
    with _tracer.start_as_current_span(name) as sp:
        if sp.is_recording():
            sp.set_attribute(SPAN_KIND, kind)
        yield sp


def start_turn(name: str = 'kms.query'):
    """턴 루트 스팬 시작 — 컨텍스트 매니저를 못 쓰는 자리용 (SSE 생성 경로는 핸들러
    반환 후 백그라운드 태스크가 끝나야 턴이 끝나므로, 스팬 종료를 핸드오프해야 한다).

    반환 (span, token): token은 핸들러 finally에서 항상 detach_turn으로 해제하고,
    span.end()는 턴이 실제로 끝나는 쪽(핸들러 or 백그라운드 태스크)이 1회 호출한다.
    create_task가 컨텍스트를 복사하므로 detach 후에도 배경 스팬은 이 루트의 자식으로 붙는다.
    """
    sp = _tracer.start_span(name)
    if sp.is_recording():
        sp.set_attribute(SPAN_KIND, 'CHAIN')
    token = otel_context.attach(trace.set_span_in_context(sp))
    return sp, token


def detach_turn(token) -> None:
    otel_context.detach(token)


def mark_error(sp, exc: BaseException) -> None:
    """예상외 예외만 ERROR로 — 정상 클라이언트 오류(404 등)는 호출부가 이 함수를 건너뛴다."""
    if sp.is_recording():
        sp.record_exception(exc)
        sp.set_status(trace.StatusCode.ERROR)


def set_attrs(sp, attrs: dict) -> None:
    """None 값은 걸러서 일괄 세팅 (OTel은 None 속성 비허용)."""
    if not sp.is_recording():
        return
    for k, v in attrs.items():
        if v is not None:
            sp.set_attribute(k, v)


def set_documents(sp, chunks, prefix: str = 'retrieval.documents') -> None:
    """검색 결과 청크를 OpenInference 문서 속성으로 평탄화 (순서·본문만).

    점수는 의도적으로 뺀다 — chunks의 rrf_score는 리랭크 '이전' 값이라 리랭크 후
    순서와 어긋나 진단을 오도한다. 실제 채택 점수는 rerank 스팬이 기록한다.
    """
    if not sp.is_recording():
        return
    for i, c in enumerate(chunks):
        sp.set_attribute(f'{prefix}.{i}.document.id', str(c.chunk_id))
        sp.set_attribute(f'{prefix}.{i}.document.content', clip(c.text))


def current_span():
    """루트 스팬에 속성을 뒤늦게 달 때(핸들러에서 prepare 결과 반영) 사용."""
    return trace.get_current_span()
