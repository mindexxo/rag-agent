"""프롬프트 조립 단위 테스트 — 컨텍스트 라벨과 블록 구조.

라벨은 인용 형식을 견인한다(라벨=인용형식 설계) — 형식이 바뀌면 인용 전체가 회귀.
"""
from rag.prompt_texts import DEFAULT_DOMAIN_HINT
from rag.prompts import (
    SYSTEM_PROMPT,
    build_attachment_blocks,
    build_chat_prompt,
    build_classify_user_message,
    build_condense_user_message,
    build_context_blocks,
    build_intent_guard_prompt,
    build_other_system_prompt,
    build_other_user_message,
    build_system_prompt,
    build_user_message,
)
from rag.retriever import RetrievedChunk


def _chunk(**kw) -> RetrievedChunk:
    base = dict(chunk_id=1, document_id=1, text='본문 내용', heading_path=['3. 보상', '3.2 기준'],
                page=2, filename='배송정책.pdf', version=1)
    return RetrievedChunk(**{**base, **kw})


class TestContextBlocks:
    def test_블록_전체_형식(self):
        # 번호·표시명·섹션·페이지·본문·구분자 전체 고정 — [번호]는 출처 꼬리의 인용 번호다
        out = build_context_blocks([_chunk()])
        assert out == '[1] 배송정책.pdf v1 섹션: 3. 보상 > 3.2 기준 / 페이지: 2\n본문 내용\n---'

    def test_같은_문서의_청크들은_같은_번호(self):
        # 인용은 문서 단위 — 블록 '사이' 구분자까지 전체 고정
        out = build_context_blocks([_chunk(text='첫 본문'), _chunk(chunk_id=2, text='둘째 본문')])
        block = '[1] 배송정책.pdf v1 섹션: 3. 보상 > 3.2 기준 / 페이지: 2\n{}\n---'
        assert out == block.format('첫 본문') + '\n' + block.format('둘째 본문')

    def test_다른_문서는_등장_순서대로_번호(self):
        out = build_context_blocks([
            _chunk(), _chunk(chunk_id=2, document_id=9, filename='환불정책.pdf'),
            _chunk(chunk_id=3, text='같은 문서 재등장')])
        assert '[1] 배송정책.pdf v1' in out and '[2] 환불정책.pdf v1' in out
        assert out.count('[1] 배송정책.pdf v1') == 2   # 재등장 청크도 같은 번호

    def test_FAQ_청크는_버전없는_FAQ_표시(self):
        out = build_context_blocks([_chunk(document_id=None, faq_id=7, filename='FAQ')])
        assert '[1] FAQ' in out
        assert 'FAQ v' not in out                # 버전 붙으면 안 됨

    def test_FAQ는_문서들_뒤_번호를_받는다(self):
        # sources_from_chunks가 FAQ를 마지막에 접는다 — 등장이 앞서도 번호는 뒤
        out = build_context_blocks([
            _chunk(chunk_id=1, document_id=None, faq_id=7, filename='FAQ', text='FAQ 본문'),
            _chunk(chunk_id=2)])
        assert '[2] FAQ' in out and '[1] 배송정책.pdf v1' in out

    def test_헤딩_없고_페이지_없으면_대시(self):
        out = build_context_blocks([_chunk(heading_path=[], page=None)])
        assert '페이지: -' in out

    def test_빈_목록(self):
        assert build_context_blocks([]) == ''


class TestAttachmentBlocks:
    def test_첨부_블록_전체_형식(self):
        out = build_attachment_blocks([{'filename': '영수증.pdf', 'text': '첨부 본문'}], start=2)
        assert out == '<첨부 문서>\n[2] 첨부: 영수증.pdf\n첨부 본문\n---\n</첨부 문서>\n\n'

    def test_없으면_빈_문자열(self):
        assert build_attachment_blocks([]) == ''


class TestCitationGrammar:
    def _regex(self, sources_n: int, attachments: list[str]):
        from rag.prompts import build_citation_grammar
        sources = [object()] * sources_n           # 문법은 개수만 쓴다 — 내용 무관
        return build_citation_grammar(sources, attachments)['structured_outputs']['regex']

    def test_유효_번호만_허용(self):
        import re
        pat = re.compile(self._regex(2, ['첨부.pdf']))     # 후보 3
        from rag.prompt_texts import TAIL_END, TAIL_START
        assert pat.fullmatch(f'답변 본문 {TAIL_START}1,3{TAIL_END}')
        assert pat.fullmatch(f'거절 {TAIL_START}{TAIL_END}')          # 빈 목록 허용
        assert not pat.fullmatch(f'답변 {TAIL_START}4{TAIL_END}')     # 범위 밖
        assert not pat.fullmatch('꼬리 없는 답변')                    # 꼬리는 필수

    def test_후보_없으면_빈_꼬리만(self):
        import re
        from rag.prompt_texts import TAIL_END, TAIL_START
        pat = re.compile(self._regex(0, []))
        assert pat.fullmatch(f'답변 {TAIL_START}{TAIL_END}')
        assert not pat.fullmatch(f'답변 {TAIL_START}1{TAIL_END}')


class TestUserMessage:
    def test_이력_없으면_맥락_블록_생략(self):
        out = build_user_message('환불 기간은?', [_chunk()])
        assert '이전 맥락' not in out
        assert '<문서>' in out and '질문: 환불 기간은?' in out

    def test_이력_있으면_참고용_명시(self):
        out = build_user_message('그럼 교환은?', [_chunk()], prior_turns=[{'q': 'q1', 'a': 'a1'}])
        assert '이전 맥락(참고용, 근거 아님):' in out
        assert '- Q: q1' in out and '- A: a1' in out
        assert '- A: a1\n\n<문서>' in out   # 이력 블록과 문서 블록 사이 빈 줄 (붙으면 경계 소실)

    def test_첨부는_문서블록_뒤_질문_앞에_배치_번호는_출처_다음(self):
        out = build_user_message('질문?', [_chunk()], attachments=[{'filename': 'r.pdf', 'text': '첨부내용'}])
        # 검색 출처 1건([1]) 다음 번호 — 문법·파서의 후보 순서와 같은 규칙
        assert out.index('</문서>') < out.index('[2] 첨부: r.pdf') < out.index('질문: 질문?')
        assert '첨부내용' in out

    def test_템플릿_구조_고정(self):
        out = build_user_message('질문?', [_chunk()])
        assert '<문서>' in out and '</문서>' in out
        assert out.rstrip().endswith('위 규칙을 지켜 한국어로 답하십시오.')


class TestCondenseUserMessage:
    def test_이력_역할_라벨과_구조(self):
        history = [{'role': 'user', 'content': '배송 3일 넘으면?'},
                   {'role': 'assistant', 'content': '보상 안내가 필요합니다.'}]
        out = build_condense_user_message('그럼 5일은?', history)
        assert '사용자: 배송 3일 넘으면?' in out
        assert '상담도우미: 보상 안내가 필요합니다.' in out
        assert out.startswith('이전 대화:\n')
        assert '현재 질문:\n그럼 5일은?' in out

    def test_빈_content_메시지는_이력에서_제외(self):
        # generating 자리표시(content='')가 condense 입력에 새지 않는 방어선
        history = [{'role': 'user', 'content': 'q'}, {'role': 'assistant', 'content': ''}]
        out = build_condense_user_message('후속?', history)
        assert '상담도우미' not in out

    def test_이력_없으면_없음_표기(self):
        out = build_condense_user_message('질문', [])
        assert '(이전 대화 없음)' in out


class TestChatPromptAndClassify:
    def test_chat_prompt_역할_순서(self):
        msgs = build_chat_prompt('시스템 규칙', '유저 입력')
        assert msgs == [{'role': 'system', 'content': '시스템 규칙'},
                        {'role': 'user', 'content': '유저 입력'}]

    def test_classify_형식(self):
        assert build_classify_user_message('  환불 돼요?  ') == '입력: 환불 돼요?\n출력:'


class TestOtherUserMessage:
    def test_이력_없으면_없음_표기(self):
        out = build_other_user_message('안녕')
        assert '(이전 대화 없음)' in out
        assert '현재 입력: 안녕' in out

    def test_이력_주입_형식(self):
        out = build_other_user_message('요약해줘', prior_turns=[{'q': '환불?', 'a': '7일입니다'}])
        assert '사용자: 환불?' in out and '상담도우미: 7일입니다' in out


_SYSTEM_BUILDERS = (build_intent_guard_prompt, build_system_prompt, build_other_system_prompt)


class TestDomainHint:
    def test_힌트가_3개_프롬프트에_주입(self):
        for build in _SYSTEM_BUILDERS:
            out = build('보험 약관·청구 절차 상담')
            assert '보험 약관·청구 절차 상담' in out
            assert '__DOMAIN_HINT__' not in out   # 마커 잔존 = 치환 누락 회귀

    def test_빈값은_중립_폴백(self):
        for build in _SYSTEM_BUILDERS:
            for hint in (None, '', '   '):
                out = build(hint)
                assert DEFAULT_DOMAIN_HINT in out
                assert '__DOMAIN_HINT__' not in out

    def test_상수는_기본_렌더링과_동일(self):
        # eval/generation 등 정적 참조(SYSTEM_PROMPT)가 기본 빌드와 어긋나면 평가·운영 프롬프트가 갈라진다
        assert SYSTEM_PROMPT == build_system_prompt()
