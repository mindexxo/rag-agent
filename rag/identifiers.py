"""식별자(번호 붙은 헤딩) — 정규화·구간 오리기의 정의점.

"11조 뭐야" 같은 이름표 질문은 의미 검색·Nori가 '11'·'조'로 쪼개 제11장·제111조와 못 가린다. 그래서 헤딩 번호를
쪼개지 않는 한 토큰('11조'·'43조의2'·'3장')으로 정규화해 정확 일치시킨다. 색인(헤딩)과 질문에 같은 함수를 쓴다.
패턴은 실문서에 있고 질문 사례가 있는 것만 켠다 — '2.1절'·'별표 1' 같은 꼴은 사례가 생기면 _RULES·_TOKEN에 추가한다.
"""

import re

# 정규화 규칙 — 순서가 의미 있다(긴 꼴 먼저). 치환 결과는 아래 _TOKEN이 뽑는 꼴이어야 한다.
_RULES = (
    # 제43조의3의2 / 제 43 조 3의 2(…)  → 43조의3의2
    (re.compile(r'제?\s*(\d+)\s*조\s*(?:의\s*)?(\d+)\s*의\s*(\d+)(?=\s*[(（])'), r'\1조의\2의\3'),
    (re.compile(r'제?\s*(\d+)\s*조\s*의\s*(\d+)\s*의\s*(\d+)'), r'\1조의\2의\3'),
    # 제43조의2 / 43조의2  → 43조의2
    (re.compile(r'제?\s*(\d+)\s*조\s*의\s*(\d+)'), r'\1조의\2'),
    # '의' 생략 표기 '제 43 조 2(태아검진…)' — 괄호 제목이 뒤따를 때만 가지로 읽는다('제9조 2항'은 참조다).
    (re.compile(r'제\s*(\d+)\s*조\s*(\d+)(?=\s*[(（])'), r'\1조의\2'),
    # 제11조 / 제 11 조 / 11조  → 11조. '조건'·'조정'·'조치'… 같은 단어의 '조'는 제외.
    (re.compile(r'제?\s*(\d+)\s*조(?!의\s*\d)'), r'\1조'),
    # 제 3 장  → 3장
    (re.compile(r'제\s*(\d+)\s*장'), r'\1장'),
)
# 토큰 추출 — '조' 뒤에 단어가 이어지면('3조건'·'1조원'·'조정') 식별자가 아니다. '11조의 내용'의 '의'는 조사라 허용.
_TOKEN = re.compile(r'\d+조(?:의\d+)*(?![건정사치직합기성회항원])|\d+장')


def normalize_identifiers(text: str) -> list[str]:
    """텍스트 속 식별자 토큰 — 등장 순서, 중복 제거.

    헤딩 한 줄·질문 한 문장용이다. 본문 전체에 돌리지 마라 — "제18조의2에 따른 …" 같은 인용이 토큰으로 잡힌다.
    """
    for rx, rep in _RULES:
        text = rx.sub(rep, text)
    return list(dict.fromkeys(_TOKEN.findall(text)))


# 조 제목 줄의 머리 — 줄 머리 조 번호(가지 조 포함) 다음 괄호가 열린다. 괄호 짝과 그 뒤는 split_heading_line이 본다.
# 가지 번호 반복은 소유 수량자(`*+`)로 둘 것 — 붙은 숫자열의 분할이 지수적이라 일반 수량자면 역추적으로 색인 워커가 멈춘다.
_HEADING_HEAD = re.compile(r'^제?\s*\d+\s*조(?:\s*의?\s*\d+)*+\s*[(（]')


def split_heading_line(line: str) -> tuple[str, str] | None:
    """조 제목 줄이면 (제목 머리, 같은 줄에 이어진 본문), 아니면 None — 조 제목 판정의 정의점.

    제목 = 줄 머리 조 번호 + 괄호 제목, 짝 맞는 닫는 괄호 뒤가 줄끝이거나 공백(공백 뒤는 같은 문단의 본문)이다.
    괄호는 짝을 세어 넘고(전각 포함), 줄 안에서 안 닫히면 제목이다(레이아웃 모델이 닫는 괄호를 다음 줄로 떨어뜨린 꼴).
    인용으로 보는 것: 괄호 없는 꼴, 닫는 괄호에 조사·기호가 바로 붙는 꼴, 괄호 안이 '이하'로 시작하는 정의 꼴.
    알려진 한계(테스트가 고정): "제98조(보상) 규정에 따라"는 제목이 되고, 괄호 없는 진짜 제목('제8조 목적')은 제목이 안 된다.
    """
    s = line.strip()
    m = _HEADING_HEAD.match(s)
    if not m or s[m.end() :].lstrip().startswith('이하'):
        return None
    depth = 1
    for i in range(m.end(), len(s)):
        if s[i] in '(（':
            depth += 1
        elif s[i] in ')）':
            depth -= 1
            if depth == 0:
                rest = s[i + 1 :]
                return (s[: i + 1], rest.strip()) if not rest or rest[0].isspace() else None
    return s, ''


def is_heading_line(line: str) -> bool:
    """조 제목 줄인가 — 판정은 split_heading_line."""
    return split_heading_line(line) is not None


def effective_headings(headings: list[str], text: str) -> list[str]:
    """조각의 헤딩 줄 — 청킹이 기록한 것 ∪ 본문의 헤딩 꼴 줄. 본문 등장 순, 중복 제거.

    기록만 믿으면 빠진다 — PDF 레이아웃 모델이 같은 꼴의 조 제목을 어떤 줄은 본문으로 분류한다.
    """
    recorded = {h.strip() for h in headings or ()}
    out: list[str] = []
    for line in text.split('\n'):
        line_ = line.strip()
        if line_ and (line_ in recorded or is_heading_line(line_)):
            out.append(line_)
    out.extend(h.strip() for h in headings or () if h.strip() not in out)
    return list(dict.fromkeys(out))


def _heading_identifier(line: str) -> str | None:
    """헤딩 한 줄의 식별자 — 줄 머리의 것 하나만. 없으면 None.

    헤딩 줄에 본문이 붙어 오면("제93조 (…) 각 지점은 이 규정 제10조에 …") 본문 속 인용까지 이름표가 되므로 머리만 본다.
    """
    tokens = normalize_identifiers(line)
    return tokens[0] if tokens and re.match(r'^\s*제?\s*\d', line) else None


def chunk_identifiers(heading_path: list[str], headings: list[str]) -> list[str]:
    """조각의 식별자 사전 — 제목 경로 + 헤딩 줄(effective_headings의 결과)에서 줄마다 줄 머리 식별자 하나씩."""
    out = [_heading_identifier(h) for h in [*(heading_path or []), *(headings or [])]]
    return list(dict.fromkeys(t for t in out if t))


def cut_identifier_span(text: str, headings: list[str], token: str) -> str:
    """묶음 조각 본문에서 `token` 헤딩의 구간(그 헤딩 줄부터 다음 기록 헤딩 직전까지)만 남긴다.

    묶음을 통째로 주면 모델이 옆 조를 읽고 답한다. 헤딩이 본문 줄에 없으면 원문 그대로다.
    """
    lines = text.split('\n')
    heads = {h.strip() for h in headings or ()}
    idx = [i for i, line in enumerate(lines) if line.strip() in heads]
    for n, i in enumerate(idx):
        if _heading_identifier(lines[i]) == token:
            j = idx[n + 1] if n + 1 < len(idx) else len(lines)
            return '\n'.join(lines[i:j])
    return text
