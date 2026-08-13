"""SSE 이벤트 포맷 단위 테스트 — streaming.sse_event / _meta_event.

FE 파서가 이 봉투 형식(event/data/빈 줄)에 의존한다.
"""
import json

from rag.retriever import RetrievalResult
from rag.service import PreparedRag
from rag.streaming import _meta_event, sse_event


def test_봉투_형식():
    out = sse_event('token', {'text': '안녕'})
    assert out == 'event: token\ndata: {"text": "안녕"}\n\n'


def test_한글_이스케이프_안함():
    assert '안녕' in sse_event('token', {'text': '안녕'})  # ensure_ascii=False


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
                    sources=[], source_doc_ids=[], cached_answer='캐시된 답', cache_kind='semantic')
    payload = json.loads(_meta_event(p).split('data: ')[1])
    assert payload['cached'] is True
    assert payload['cache_kind'] == 'semantic'


def test_meta_no_evidence_reason():
    p = PreparedRag(conversation_id=1, original_query='q', standalone_query='q',
                    prior_turns=[], retrieval=RetrievalResult(chunks=[], no_evidence=True, reason='no_results'),
                    sources=[], source_doc_ids=[])
    payload = json.loads(_meta_event(p).split('data: ')[1])
    assert payload['reason'] == 'no_evidence'
