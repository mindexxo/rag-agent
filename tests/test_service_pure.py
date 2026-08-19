"""service 순수 로직 단위 테스트 — _source_doc_ids / PreparedRag 상태 속성.

_source_doc_ids의 음수 네임스페이스는 FAQ 캐시 무효화 키와 짝 — 깨지면 무효화가 조용히 어긋난다.
needs_generation은 SSE 백그라운드 생성 분기와 정확히 일치해야 한다 — #36부터 resolved_answer
한 곳에서 파생되므로, 그 파생 관계와 생성자 불변식(route ↔ retrieval)을 여기서 고정한다.
"""
import pytest

from rag.retriever import RetrievalResult, RetrievedChunk
from rag.service import PreparedRag, _source_doc_ids


def _chunk(document_id=None, faq_id=None) -> RetrievedChunk:
    return RetrievedChunk(chunk_id=1, document_id=document_id, text='x', heading_path=[],
                          page=None, filename='f.pdf', version=1, faq_id=faq_id)


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
        assert _prepared(route='blocked', retrieval=None).needs_generation is False

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


class TestRouteRetrievalInvariant:
    """route와 retrieval의 짝을 생성자가 강제한다 (#36).

    should_cache가 route를 안 보고 `retrieval is not None`으로 blocked/other를 걸러내므로,
    이 짝이 어긋나면 차단된 턴이 캐시에 들어가는 식으로 **조용히** 틀린다.
    이전엔 _routed() 헬퍼의 관례로만 지켜졌다.
    """

    def test_blocked인데_retrieval이_있으면_거부(self):
        with pytest.raises(ValueError, match='불변식 위반'):
            _prepared(route='blocked')          # base의 retrieval이 채워진 채로

    def test_other인데_retrieval이_있으면_거부(self):
        with pytest.raises(ValueError, match='불변식 위반'):
            _prepared(route='other')

    def test_knowledge인데_retrieval이_없으면_거부(self):
        with pytest.raises(ValueError, match='불변식 위반'):
            _prepared(retrieval=None)           # route 기본값 knowledge

    def test_올바른_짝은_통과(self):
        assert _prepared(route='blocked', retrieval=None).route == 'blocked'
        assert _prepared().route == 'knowledge'


class TestResolvedAnswer:
    """needs_generation과 generate()가 함께 보는 단일 판정점 (#36)."""

    def test_blocked는_고정문구(self):
        from rag.prompt_texts import BLOCKED_INPUT_ANSWER
        assert _prepared(route='blocked', retrieval=None).resolved_answer == BLOCKED_INPUT_ANSWER

    def test_other는_생성_필요(self):
        # OTHER는 검색 없이도 LLM이 대화성 응답을 만든다 — cached_answer보다 먼저 걸러져야 한다
        assert _prepared(route='other', retrieval=None).resolved_answer is None

    def test_캐시히트는_저장된_답변(self):
        assert _prepared(cached_answer='캐시된 답').resolved_answer == '캐시된 답'

    def test_근거없음은_고정문구(self):
        from rag.prompt_texts import NO_EVIDENCE_ANSWER
        r = RetrievalResult(chunks=[], no_evidence=True, reason='no_results')
        assert _prepared(retrieval=r).resolved_answer == NO_EVIDENCE_ANSWER

    def test_정상경로는_생성_필요(self):
        assert _prepared().resolved_answer is None

    def test_needs_generation은_resolved_answer의_파생값(self):
        for p in (_prepared(route='blocked', retrieval=None), _prepared(cached_answer='x'),
                  _prepared(retrieval=RetrievalResult(chunks=[], no_evidence=True, reason='no_results')),
                  _prepared(route='other', retrieval=None), _prepared()):
            assert p.needs_generation == (p.resolved_answer is None)


class TestGenerateQuerySlot:
    """generate()가 "질문:" 자리에 원 질문을, 재작성 질의는 참고로 넘기는지 (#48).

    조립 자체는 test_prompts.py가 검증한다. 여기서 고정하는 건 **호출부가 두 값을 뒤바꿔
    넘기지 않는다**는 것 — 뒤바뀌어도 답변은 그럴듯하게 나오므로 어떤 테스트도 못 잡는다.
    """

    @pytest.mark.asyncio
    async def test_원질문과_재작성질의를_각각_넘긴다(self, monkeypatch):
        captured = {}

        def fake_build(original_query, chunks, *, standalone_query=None, prior_turns=None,
                       attachments=None, domain_hint=None):
            captured.update(original_query=original_query, standalone_query=standalone_query,
                            prior_turns=prior_turns, domain_hint=domain_hint)
            return [{'role': 'system', 'content': 's'}, {'role': 'user', 'content': 'u'}]

        monkeypatch.setattr('rag.service.build_knowledge_generation_prompt', fake_build)

        class _Llm:
            async def astream(self, prompt, extra_body=None):
                yield 'ok'

        from rag.service import RagService
        svc = RagService(tenant_id='t', session=None)
        svc._llm = _Llm()
        prepared = _prepared(original_query='그건 한 마리에 얼마예요?',
                             standalone_query='냉장 생닭 한 마리 가격은?',
                             prior_turns=[{'q': '유통기한?', 'a': '5일입니다'}],
                             domain_hint='식품 상담')

        assert ''.join([t async for t in svc.generate(prepared)]) == 'ok'
        assert captured['original_query'] == '그건 한 마리에 얼마예요?'
        assert captured['standalone_query'] == '냉장 생닭 한 마리 가격은?'
        assert captured['prior_turns'] == [{'q': '유통기한?', 'a': '5일입니다'}]
        assert captured['domain_hint'] == '식품 상담'
