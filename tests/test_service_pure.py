"""service 순수 로직 단위 테스트 — _source_doc_ids / PreparedRag 상태 속성.

_source_doc_ids의 음수 네임스페이스는 FAQ 캐시 무효화 키와 짝 — 깨지면 무효화가 조용히 어긋난다.
needs_generation은 SSE 백그라운드 생성 분기와 정확히 일치해야 한다.
"""
from rag.retriever import RetrievalResult, RetrievedChunk
from rag.service import PreparedRag, _source_doc_ids


def _chunk(document_id=None, faq_id=None) -> RetrievedChunk:
    return RetrievedChunk(chunk_id=1, document_id=document_id, text='x', heading_path=[],
                          page=None, rrf_score=0.03, branches=['dense'],
                          filename='f.pdf', version=1, faq_id=faq_id)


def _prepared(**kw) -> PreparedRag:
    base = dict(conversation_id=1, original_query='q', standalone_query='q',
                prior_turns=[], retrieval=RetrievalResult(chunks=[], no_evidence=False, reason=None),
                sources=[], source_doc_ids=[])
    return PreparedRag(**{**base, **kw})


class TestSourceDocIds:
    def test_문서는_양수_FAQ는_음수(self):
        ids = _source_doc_ids([_chunk(document_id=5), _chunk(faq_id=7)])
        assert set(ids) == {5, -7}

    def test_중복_제거(self):
        ids = _source_doc_ids([_chunk(document_id=5), _chunk(document_id=5)])
        assert list(ids) == [5]

    def test_같은_번호의_문서와_FAQ는_구분(self):
        # doc 5와 faq 5가 하나로 접히면 캐시 무효화 키가 어긋난다
        ids = _source_doc_ids([_chunk(document_id=5), _chunk(document_id=5), _chunk(faq_id=5)])
        assert set(ids) == {5, -5}

    def test_빈_입력(self):
        assert list(_source_doc_ids([])) == []


class TestNeedsGeneration:
    def test_blocked는_즉시(self):
        assert _prepared(route='blocked').needs_generation is False

    def test_other는_생성(self):
        assert _prepared(route='other', retrieval=None).needs_generation is True

    def test_캐시_히트는_즉시(self):
        assert _prepared(cached_answer='캐시된 답').needs_generation is False

    def test_근거없음_첨부없음은_즉시(self):
        r = RetrievalResult(chunks=[], no_evidence=True, reason='no_results')
        assert _prepared(retrieval=r).needs_generation is False

    def test_근거없어도_첨부있으면_생성(self):
        r = RetrievalResult(chunks=[], no_evidence=True, reason='no_results')
        p = _prepared(retrieval=r, attachments=[{'filename': 'a.pdf', 'text': 'x'}])
        assert p.needs_generation is True

    def test_정상_knowledge는_생성(self):
        assert _prepared().needs_generation is True


class TestShouldCache:
    def test_정상_신규_응답은_캐시(self):
        assert _prepared().should_cache is True

    def test_캐시_히트는_저장_안함(self):
        assert _prepared(cached_answer='x').should_cache is False

    def test_근거없음은_저장_안함(self):
        r = RetrievalResult(chunks=[], no_evidence=True, reason='no_results')
        assert _prepared(retrieval=r).should_cache is False

    def test_첨부_대화는_저장_안함(self):
        p = _prepared(attachments=[{'filename': 'a.pdf', 'text': 'x'}])
        assert p.should_cache is False

    def test_검색_없는_경로는_저장_안함(self):
        assert _prepared(route='other', retrieval=None).should_cache is False
