"""프롬프트 조립 — 텍스트 템플릿에 런타임 값을 채워 LLM 메시지를 만든다.

문구 자체는 rag/prompt_texts.py, 답변 판정(거절·인용)은 rag/answer_check.py (#36 분리).
LLM에게 "문서 블록만 근거로 답해라, 인용은 [파일명 v1] 형식"을 강제하는 규칙은 텍스트 쪽에 있다.
"""
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
    각 블록 앞 라벨을 인용 형식 [파일명 vN] 그대로 둔다 — 모델이 눈앞 라벨을
    그대로 흉내 내므로, 라벨=인용형식이어야 규칙 6대로 [파일명 vN]으로 인용한다.
    """
    blocks = []
    for chunk in chunks:
        heading = ' > '.join(chunk.heading_path) if chunk.heading_path else ''
        page = chunk.page or '-'
        # FAQ 청크는 버전 없는 [FAQ] 라벨 (파일·버전 개념이 없음 — F3)
        label = '[FAQ]' if getattr(chunk, 'faq_id', None) else f'[{chunk.filename} v{chunk.version}]'
        blocks.append(
            f'{label} 섹션: {heading} / 페이지: {page}\n'
            f'{chunk.text}\n---'
        )
    return '\n'.join(blocks)

def build_attachment_blocks(attachments: list[dict]) -> str:
    """채팅 첨부 문서 -> <첨부 문서> 블록 조립. 없으면 빈 문자열.
    라벨을 인용 형식 [첨부: 파일명] 그대로 둔다 (build_context_blocks와 같은 원리).
    attachments: [{"filename": "...", "text": "..."}, ...]
    """
    if not attachments:
        return ''
    blocks = [
        f"[첨부: {a['filename']}]\n{a['text']}\n---"
        for a in attachments
    ]
    return '<첨부 문서>\n' + '\n'.join(blocks) + '\n</첨부 문서>\n\n'


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
        attachment_blocks=build_attachment_blocks(attachments or []),
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

    return f"""이전 대화:
{history_block}

현재 질문:
{query.strip()}

검색용 독립 질문:"""
