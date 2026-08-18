"""프롬프트 조립 — 텍스트 템플릿에 런타임 값을 채워 LLM 메시지를 만든다.

문구 자체는 rag/prompt_texts.py, 거절 판정은 rag/answer_check.py (#36 분리).
인용은 답변 끝 출처 꼬리(#56) — 번호 순서는 rag/citation_labels.py 단일 정의점,
꼬리 강제 문법(build_citation_grammar)도 여기서 조립한다(런타임 값 → LLM 입력이라는 같은 성격).
"""
import re

from rag.citation_labels import (TAIL_END, TAIL_START, attachment_display,
                                 source_display, sources_from_chunks)
from rag.prompt_texts import (
    DEFAULT_DOMAIN_HINT,
    USER_TEMPLATE,
    _INTENT_GUARD_SYSTEM_PROMPT_TEMPLATE,
    _OTHER_SYSTEM_PROMPT_TEMPLATE,
    _SYSTEM_PROMPT_TEMPLATE,
)
from rag.retriever import RetrievedChunk


def _resolve_domain_hint(domain_hint: str | None) -> str:
    """domain_hint 정규화 + 폴백 — 3개 빌더가 공유하는 단일 폴백 지점."""
    return (domain_hint or '').strip() or DEFAULT_DOMAIN_HINT


def build_system_prompt(domain_hint: str | None = None) -> str:
    """KNOWLEDGE 생성 시스템 프롬프트 — 지식 범위 슬롯 치환."""
    return _SYSTEM_PROMPT_TEMPLATE.replace('__DOMAIN_HINT__', _resolve_domain_hint(domain_hint))


SYSTEM_PROMPT = build_system_prompt()   # 기본(중립) 렌더링 — eval/generation 등 힌트 없는 정적 참조용


def build_intent_guard_prompt(domain_hint: str | None = None) -> str:
    """입력 검사·인텐트 분류 시스템 프롬프트 — KNOWLEDGE 정의의 지식 범위 슬롯 치환."""
    return _INTENT_GUARD_SYSTEM_PROMPT_TEMPLATE.replace('__DOMAIN_HINT__', _resolve_domain_hint(domain_hint))


def build_other_system_prompt(domain_hint: str | None = None) -> str:
    """OTHER 경로 생성 시스템 프롬프트 — <역할 안내>의 지식 범위 슬롯 치환."""
    return _OTHER_SYSTEM_PROMPT_TEMPLATE.replace('__DOMAIN_HINT__', _resolve_domain_hint(domain_hint))


def build_chat_prompt(system_content: str, user_content: str) -> list[dict]:
    """LLM chat/completions 호출용 messages를 조립한다.

    system_content는 LLM의 역할과 규칙이고,
    user_content는 해당 작업에 필요한 입력 본문이다.
    """
    return [
        {'role': 'system', 'content': system_content},
        {'role': 'user', 'content': user_content},
    ]


def build_context_blocks(chunks: list[RetrievedChunk]) -> str:
    """RetrievedChunk 리스트 -> <문서> 블록 안에 들어갈 텍스트 조립.
    각 블록 앞 [번호]가 출처 꼬리의 인용 번호다 — 번호 순서는 sources_from_chunks가
    단일 정의점(citation_labels)이라, 문법·파서와 어긋날 수 없다. 같은 문서의 여러
    청크는 같은 번호를 받는다(인용은 문서 단위). FAQ 청크는 retriever가 filename='FAQ'로
    강제하므로(F99) 'FAQ' 후보 하나로 접힌다.
    """
    numbers = {s.document_id if s.filename != 'FAQ' else 'FAQ': i
               for i, s in enumerate(sources_from_chunks(chunks), start=1)}
    blocks = []
    for chunk in chunks:
        heading = ' > '.join(chunk.heading_path) if chunk.heading_path else ''
        page = chunk.page or '-'
        n = numbers['FAQ' if chunk.faq_id is not None else chunk.document_id]
        blocks.append(
            f'[{n}] {source_display(chunk)} 섹션: {heading} / 페이지: {page}\n'
            f'{chunk.text}\n---'
        )
    return '\n'.join(blocks)

def build_attachment_blocks(attachments: list[dict], start: int = 1) -> str:
    """채팅 첨부 문서 -> <첨부 문서> 블록 조립. 없으면 빈 문자열.
    [번호]는 검색 출처 번호에 이어 start부터 매긴다 (번호 불변식 — citation_labels).
    attachments: [{"filename": "...", "text": "..."}, ...]
    """
    if not attachments:
        return ''
    blocks = [
        f"[{start + i}] {attachment_display(a['filename'])}\n{a['text']}\n---"
        for i, a in enumerate(attachments)
    ]
    return '<첨부 문서>\n' + '\n'.join(blocks) + '\n</첨부 문서>\n\n'


def build_citation_grammar(sources, attachment_filenames: list[str]) -> dict:
    """출처 꼬리를 강제하는 guided decoding 정규식 — LlmClient extra_body용 (#56).

    꼬리는 **필수**다: 선택(`(...)?`)으로 두면 본문만으로도 문법이 완성돼 EOS가 항상
    허용되고, 강제력이 0이 된다(설계 검토에서 확인). 필수라서 모델은 꼬리를 쓰기 전엔
    끝낼 수 없고, 꼬리 안에는 유효 범위의 번호 목록(또는 빈 목록)만 올 수 있다.
    거절 답변도 꼬리는 붙는다 — 빈 목록으로 (프롬프트 규칙 7이 같은 것을 지시).

    첨부를 빠뜨리면 첨부 인용이 문법에 막힌다(#41의 함정) — 후보 수에 반드시 포함.
    vLLM v0.12+ structured_outputs 형식. 서버가 미지원(400)이면 호출부가 문법 없이
    재시도한다(fail-open) — 파서의 번호 범위 검증은 문법 유무와 무관하게 동일.
    """
    count = len(sources) + len(attachment_filenames)
    nums = '|'.join(str(i) for i in range(1, count + 1))
    inner = f'(?:(?:{nums})(?:,(?:{nums}))*)?' if count else ''
    tail = f'{re.escape(TAIL_START)}{inner}{re.escape(TAIL_END)}'
    return {"structured_outputs": {"regex": rf'[\s\S]*{tail}'}}


def build_user_message(
        query: str,
        chunks: list[RetrievedChunk],
        prior_turns: list[dict] | None = None,
        attachments: list[dict] | None = None,
) -> str:
    """유저 메시지 전체 조립.
    prior_turns: [{"q": "...", "a": "..."}, ...] 형태. Stage E.1(멀티턴)에서 채워짐.
    attachments: 채팅 첨부 문서 [{"filename", "text"}]. 없으면 블록 생략.
    """
    if prior_turns:
        lines = ['이전 맥락(참고용, 근거 아님):']
        for t in prior_turns:
            lines.append(f"- Q: {t['q']}")
            lines.append(f"- A: {t['a']}")
        prior_turns_block = "\n".join(lines) + "\n\n"
    else:
        prior_turns_block = ''

    return USER_TEMPLATE.format(
        prior_turns_block=prior_turns_block,
        context_blocks=build_context_blocks(chunks),
        # 첨부 번호는 검색 출처 번호 다음부터 — 문법·파서의 후보 순서(sources + 첨부)와 동일
        attachment_blocks=build_attachment_blocks(attachments or [], start=len(sources_from_chunks(chunks)) + 1),
        query=query
    )


def build_classify_user_message(query: str, has_attachments: bool = False) -> str:
    """분류 LLM에 넘길 사용자 메시지 — 현재 입력 (+ 첨부 존재 신호).

    첨부 신호가 없으면 "요약해줘"가 대화 요약(OTHER)으로 오분류돼 첨부 요약이
    막힌다 (교차 기능 갭 — 2026-07-19). 형식은 시스템 프롬프트 few-shot과 동일.
    """
    prefix = "상황: 첨부 문서 있음\n" if has_attachments else ""
    return f"{prefix}입력: {query.strip()}\n출력:"


def build_other_user_message(query: str, prior_turns: list[dict] | None = None) -> str:
    """'그 외'(OTHER) 경로 유저 메시지 — 이전 대화 + 현재 입력.
    요약·회상·되묻기가 가능하도록 이력을 싣는다. 이력은 서비스 사실의 근거가 아니다.
    prior_turns: [{"q": "...", "a": "..."}, ...] (build_prior_turns 산출물)
    """
    if prior_turns:
        lines = ['<이전 대화>']
        for t in prior_turns:
            lines.append(f"사용자: {t['q']}")
            lines.append(f"상담도우미: {t['a']}")
        lines.append('</이전 대화>')
        history_block = "\n".join(lines) + "\n\n"
    else:
        history_block = '<이전 대화>\n(이전 대화 없음)\n</이전 대화>\n\n'

    return f"{history_block}현재 입력: {query.strip()}"


def build_condense_user_message(query: str, history: list[dict]) -> str:
    """condense LLM에 넘길 사용자 메시지를 조립한다.

    history는 최근 대화 메시지 목록이며
    [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    형태를 기대한다.

    이 함수는 LLM이 답변을 생성하지 않고, 현재 질문을 검색 가능한 독립 질문으로
    재작성하는 데 필요한 이전 대화와 현재 질문만 제공한다.
    """
    history_lines = []

    for message in history:
        role = "사용자" if message["role"] == "user" else "상담도우미"
        content = message["content"].strip()
        if content:
            history_lines.append(f"{role}: {content}")

    history_block = "\n".join(history_lines) if history_lines else "(이전 대화 없음)"

    # 끝 라벨은 "출력:" — 구 "검색용 독립 질문:"은 프로즈 출력을 유도하던 완성 트릭인데
    # JSON 스키마 강제(#43) 하에서는 혼선 신호다. build_classify_user_message와 같은 관례.
    return f"""이전 대화:
{history_block}

현재 질문:
{query.strip()}

출력:"""
