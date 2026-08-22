"""SSE 이벤트 포맷 단위 테스트 — streaming.sse_event / _meta_event.

FE 파서가 이 봉투 형식(event/data/빈 줄)에 의존한다.
"""
import json

from rag.retriever import RetrievalResult
from rag.service import PreparedRag
from rag.streaming import _meta_event, sse_event


def test_봉투_형식():
    out = sse_event('delta', {'text': '안녕'})
    assert out == 'event: delta\ndata: {"text": "안녕"}\n\n'


def test_한글_이스케이프_안함():
    assert '안녕' in sse_event('delta', {'text': '안녕'})  # ensure_ascii=False


def test_done_페이로드_형식():
    """done은 빈 객체가 아니라 최종 상태를 싣는다 (#56) — 필드 셋이 FE 계약."""
    from rag.streaming import TurnResult, _done_payload
    from schemas.kms import SourceCitation
    result = TurnResult(answer='답', citations=[SourceCitation(document_id=5, filename='정책.pdf', version=2)],
                        finish_reason='done', latency_ms=123)
    assert _done_payload(result) == {
        'finish_reason': 'done',
        'latency_ms': 123,
        'citations': [{'document_id': 5, 'filename': '정책.pdf', 'version': 2}],
    }


def test_meta_필드():
    p = PreparedRag(conversation_id=3, original_query='q', standalone_query='q',
                    prior_turns=[], retrieval=RetrievalResult(chunks=[], no_evidence=False, reason=None),
                    sources=[], source_doc_ids=[], assistant_message_id=42)
    out = _meta_event(p)
    payload = json.loads(out.split('data: ')[1])
    assert payload == {
        'conversation_id': 3,
        'assistant_message_id': 42,   # FE 재접속 폴링이 이 필드에 의존
        'cached': False,
        'cache_kind': None,
        'reason': 'ok',
    }


def test_meta_캐시_히트():
    # cached/cache_kind 하드코딩 회귀 방지 — 캐시 미스 상태만 테스트하면 상수화 뮤테이션을 못 잡는다
    p = PreparedRag(conversation_id=1, original_query='q', standalone_query='q',
                    prior_turns=[], retrieval=RetrievalResult(chunks=[], no_evidence=False, reason=None),
                    sources=[], source_doc_ids=[], assistant_message_id=1,
                    cached_answer='캐시된 답', cache_kind='semantic')
    payload = json.loads(_meta_event(p).split('data: ')[1])
    assert payload['cached'] is True
    assert payload['cache_kind'] == 'semantic'


def test_meta_no_evidence_reason():
    p = PreparedRag(conversation_id=1, original_query='q', standalone_query='q',
                    prior_turns=[], retrieval=RetrievalResult(chunks=[], no_evidence=True, reason='no_results'),
                    sources=[], source_doc_ids=[], assistant_message_id=1)
    payload = json.loads(_meta_event(p).split('data: ')[1])
    assert payload['reason'] == 'no_evidence'
