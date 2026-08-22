"""프롬프트 조립 — 텍스트 템플릿에 런타임 값을 채워 LLM 메시지를 만든다.

문구 자체는 rag/prompt_texts.py (#36 분리). 근거 유무 판정은 rag/citation_tail.py의
실인용 개수가 담당한다 — 거절 문구 판정은 #61에서 폐기됐다.
인용은 답변 끝 출처 꼬리(#56) — 번호 순서는 rag/citation_labels.py 단일 정의점,
꼬리 강제 제약(build_citation_constraint)도 여기서 조립한다(런타임 값 → LLM 입력이라는 같은 성격).
"""
from rag.citation_labels import (TAIL_END, TAIL_START, attachment_display,
                                 source_display, sources_from_chunks)
from rag.prompt_texts import (
    DEFAULT_DOMAIN_HINT,
    PRIOR_TURNS_LABEL,
    STANDALONE_QUERY_LABEL,
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
    단일 정의점(citation_labels)이라, 제약·파서와 어긋날 수 없다. 같은 문서의 여러
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


def build_citation_constraint(sources, attachment_filenames: list[str]) -> dict:
    """출처 꼬리 안쪽을 유효 후보 번호로 강제한다 — LlmClient extra_body용 (#56, #65 재구현).

    반환값은 vLLM의 `response_format` + `structural_tag`다. `triggers`에 든 문자열이 생성
    도중 나타나면 그 지점부터 `end`까지 `schema`를 강제한다 — **본문은 자유, 꼬리만 제약**이
    필요한 우리 경우에 정확히 맞는 기능이다. 그래서 꼬리 안쪽이 JSON 정수 배열이 된다
    (형식 상수는 rag/prompt_texts.py의 TAIL_EXAMPLE_* — 프롬프트 예시와 같은 정의점).

    ⚠ **정규식으로 되돌리지 말 것.** #56의 원래 구현은 `[\\s\\S]*««1|2»»` 형태의 정규식이었고
    vLLM 0.24.0에서 **강제되지 않았다**(2026-08-20 실측, #61에서 발견 → #65). 파라미터는
    받아들여지고 예외도 안 나는데 제약이 안 걸린다. 사다리 실측:

        ««(?:1)?»»              강제됨   ← 제약이 문자열 시작 위치
        ««(?:1)?»»[\\s\\S]*      강제됨   ← 제약이 앞, 수량자는 뒤
        [\\s\\S]*««(?:1)?»»      무효    ← 옛 구현의 형태
        [^«]*««(?:1)?»»         무효
        [\\s\\S]{0,200}««…»»     무효    ← 유계로 바꿔도 마찬가지
        + / .* / [^«»]*          무효

    규칙: 앞에 "아무 문자" 구간이 오면 뒤쪽 제약이 무효다(유한·무한 무관). 이건 서버 버전
    문제가 아니라 "꼬리가 답변 **끝**에 온다"는 구조와 정규식 가드 디코딩의 충돌이라 재현성이
    있다 — 백엔드 3종(xgrammar·guidance·outlines)과 GBNF까지 전부 같은 결과였다.

    structural_tag가 강제된다는 근거(반증 테스트): 후보가 1개뿐인 문맥에 enum=[2]를 주면
    모델이 3/3 `[2]`를 쓴다 — 문맥상 말이 안 되는 값을 강제로 쓴 것이다. 스트리밍
    (astream, 운영 경로)에서도 강제된다. 옛 정규식은 같은 문항에서 5/5 위반했다.

    후보 0개면 `maxItems: 0`으로 빈 배열만 허용한다 — 운영에선 도달 불가(후보 0 = no_evidence
    = 즉시 경로라 LLM을 안 탄다)지만 eval oracle/retrieved 모드에서는 도달한다. `enum: []`을
    쓰지 않은 이유: 선택지 0개 enum을 백엔드가 어떻게 컴파일하는지 미확인이라, 검증된 형태를 쓴다.

    거절 답변도 꼬리는 붙는다 — 빈 배열로(프롬프트의 인용 표시 규칙이 같은 것을 지시). 후보가 있어도
    모델이 빈 배열을 낼 수 있다(minItems 미지정) — 답할 수 없는 질문에 후보 4개를 주고 확인,
    2/2 `[]`. 즉 강제는 "없는 번호를 못 쓰게" 할 뿐 거절의 자유를 빼앗지 않는다.

    첨부를 빠뜨리면 안 되는 이유와 그 실패 방향(#41의 함정, #65에서 누락→오답으로 바뀜)은
    rag/citation_labels.py 모듈 docstring — 후보 목록 파생은 PreparedRag.citation_candidates
    한 곳이다.

    서버가 structural_tag를 미지원하면 호출부가 제약 없이 재시도한다(fail-open) — 파서의 번호
    범위 검증은 제약 유무와 무관하게 동일하다(rag/citation_tail.py의 원칙).
    """
    count = len(sources) + len(attachment_filenames)
    schema = ({"type": "array", "items": {"type": "integer", "enum": list(range(1, count + 1))}}
              if count else {"type": "array", "maxItems": 0})
    return {"response_format": {
        "type": "structural_tag",
        "triggers": [TAIL_START],
        "structures": [{"begin": TAIL_START, "schema": schema, "end": TAIL_END}],
    }}


def build_user_message(
        query: str,
        chunks: list[RetrievedChunk],
        prior_turns: list[dict] | None = None,
        attachments: list[dict] | None = None,
        standalone_query: str | None = None,
) -> str:
    """유저 메시지 전체 조립.
    query: "질문:" 슬롯에 표시할 **원 질문**(original_query).
    prior_turns: [{"q": "...", "a": "..."}, ...] 형태. Stage E.1(멀티턴)에서 채워짐.
    attachments: 채팅 첨부 문서 [{"filename", "text"}]. 없으면 블록 생략.
    standalone_query: 검색에 실제로 쓴 재작성 질의. query와 다를 때만 참고 줄로 병기한다 —
      같으면(단일턴) 중복이라 생략. 왜 교체가 아니라 병기인지는 prompt_texts.py의
      STANDALONE_QUERY_LABEL 주석에 실측과 함께 있다 (#48).
    """
    if prior_turns:
        lines = [PRIOR_TURNS_LABEL]
        for t in prior_turns:
            lines.append(f"- Q: {t['q']}")
            lines.append(f"- A: {t['a']}")
        prior_turns_block = "\n".join(lines) + "\n\n"
    else:
        prior_turns_block = ''

    # 양쪽을 strip해서 비교한다 — 스키마가 query의 앞뒤 공백을 벗기지 않으므로(schemas/kms.py),
    # 공백 차이만으로 "다르다"고 보고 같은 문장을 두 번 싣는 일을 막는다. 공백만인
    # standalone_query가 빈 참고 줄로 실리는 것도 여기서 걸러진다.
    standalone = (standalone_query or '').strip()
    standalone_line = (f'\n{STANDALONE_QUERY_LABEL} {standalone}'
                       if standalone and standalone != (query or '').strip() else '')

    return USER_TEMPLATE.format(
        prior_turns_block=prior_turns_block,
        context_blocks=build_context_blocks(chunks),
        # 첨부 번호는 검색 출처 번호 다음부터 — 제약·파서의 후보 순서(sources + 첨부)와 동일
        attachment_blocks=build_attachment_blocks(attachments or [], start=len(sources_from_chunks(chunks)) + 1),
        query=query,
        standalone_line=standalone_line,
    )


def build_knowledge_generation_prompt(
        original_query: str,
        chunks: list[RetrievedChunk],
        *,
        standalone_query: str | None = None,
        prior_turns: list[dict] | None = None,
        attachments: list[dict] | None = None,
        domain_hint: str | None = None,
) -> list[dict]:
    """KNOWLEDGE 답변 생성용 messages 조립 — 운영과 eval이 공유하는 단일 조립점.

    이전엔 rag/service.py와 eval/generation.py가 각자 system+user를 조립했고, 그래서
    eval 쪽에 prior_turns를 넘기지 않는 누락이 생겨도 어디에도 걸리지 않았다(#48에서 발견 —
    이력 없이 측정한 결과가 "재작성 오염 때문에 오답"이라는 오진으로 이어졌다). 조립점을
    하나로 모아, 인자가 늘어날 때 두 호출부가 함께 드러나게 한다.

    선택 인자를 키워드 전용(*)으로 강제한 이유: original_query와 standalone_query는 둘 다
    질문 문자열이라 위치로 넘기면 뒤바꿔도 타입 오류가 안 나고 답변도 그럴듯하게 나온다 —
    어떤 테스트도 못 잡는 종류의 사고다. 이름을 쓰게 만들어 그 여지를 없앤다.
    """
    return build_chat_prompt(
        build_system_prompt(domain_hint),
        build_user_message(original_query, chunks, prior_turns=prior_turns,
                           attachments=attachments, standalone_query=standalone_query),
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
