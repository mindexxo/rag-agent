"""식별자(번호 붙은 헤딩) — 정규화·구간 오리기의 **정의점** (#238).

"취업규칙 11조 뭐야" 같은 질문은 의미가 아니라 **이름표**로 묻는다. 의미 검색은 숫자를 거의 안 보고,
키워드 검색(Nori)은 "제11조"를 '11'·'조'로 쪼개 제11장·제111조·개정 날짜 속 '21'과 구분하지 못했다
(실측 2026-10-15: "제11조" 질문에 "제11장" 소속 조각이 상위 5개를 독식, 리랭커도 제목의 장 번호를
조 번호로 봄 — 제목 "제3장"→"제9장"만 바꾸면 0.90→0.09). 그래서 헤딩의 번호를 **쪼개지 않는 한 토큰**
('11조', '43조의2', '3장')으로 정규화해 정확 일치시킨다.

규칙은 여기 한 곳이고 **색인(헤딩)과 질문에 같은 함수를 적용한다** — 질문을 따로 해석하는 로직은 없다.
질문에서 나온 토큰이 색인 사전에 없으면 아무 일도 일어나지 않는다(일반 질문은 영향 0).

쓰는 곳: rag/os_index.py(색인 필드 `identifiers`·`headings` = effective_headings) · rag/retriever.py(후보 주입·가산·오리기) ·
eval/retrieval_article.py(정답 정의).
0층 leaf — `rag/` 아무것도 import하지 않는다.

조문 전용이 아니다 — 번호 붙은 헤딩 일반을 겨냥하되, 패턴은 실문서에 있고 질문 사례가 있는 것만 켠다
(결정 11·12 #238). '2.1절'·'별표 1' 같은 꼴은 질문 사례가 생기면 _RULES·_TOKEN에 한 줄씩 추가한다.
"""
import re

# 정규화 규칙 — 순서가 의미 있다(긴 꼴 먼저). 치환 결과는 아래 _TOKEN이 뽑는 꼴이어야 한다.
_RULES = (
    # 제43조의3의2 / 제 43 조 3의 2(…)  → 43조의3의2
    (re.compile(r'제?\s*(\d+)\s*조\s*(?:의\s*)?(\d+)\s*의\s*(\d+)(?=\s*[(（])'), r'\1조의\2의\3'),
    (re.compile(r'제?\s*(\d+)\s*조\s*의\s*(\d+)\s*의\s*(\d+)'), r'\1조의\2의\3'),
    # 제43조의2 / 43조의2  → 43조의2
    (re.compile(r'제?\s*(\d+)\s*조\s*의\s*(\d+)'), r'\1조의\2'),
    # '의' 생략 표기 '제 43 조 2(태아검진…)' — 사내 취업규칙 원본이 실제로 이렇게 적는다(#238 결정 10).
    # **괄호 제목이 뒤따를 때만** 가지로 읽는다 — '제9조 2항'은 참조다. DOCX 폴백도 같은 판정(split_heading_line)이라 제목으로 올린다(#250·#233).
    (re.compile(r'제\s*(\d+)\s*조\s*(\d+)(?=\s*[(（])'), r'\1조의\2'),
    # 제11조 / 제 11 조 / 11조  → 11조. '조건'·'조정'·'조치'… 같은 단어의 '조'는 제외.
    (re.compile(r'제?\s*(\d+)\s*조(?!의\s*\d)'), r'\1조'),
    # 제 3 장  → 3장
    (re.compile(r'제\s*(\d+)\s*장'), r'\1장'),
)
# 토큰 추출 — '조' 뒤에 단어가 이어지면('3조건'·'1조원'·'조정') 식별자가 아니다. '11조의 내용'의 '의'는 조사라 허용.
_TOKEN = re.compile(r'\d+조(?:의\d+)*(?![건정사치직합기성회항원])|\d+장')


def normalize_identifiers(text: str) -> list[str]:
    """텍스트 속 식별자 토큰 — 등장 순서, 중복 제거. 없으면 [].

    헤딩 한 줄이든 질문 한 문장이든 같은 함수다. 본문 전체에 돌리지 마라 — "제18조의2에 따른 …" 같은
    인용이 토큰으로 잡혀 정답 아닌 조각이 그 조로 등재된다(#238 실측: 참조 문구 오인 3건).
    """
    for rx, rep in _RULES:
        text = rx.sub(rep, text)
    return list(dict.fromkeys(_TOKEN.findall(text)))


# 조 제목 줄의 머리 — 줄 머리 조 번호(가지 조 포함) 다음 괄호가 열린다. 괄호 짝과 그 뒤는 split_heading_line이 본다.
# 가지 번호 반복은 **소유 수량자(`*+`)** 다 — 공백·'의'가 둘 다 생략 가능해 붙은 숫자열("1111…")을 나누는 방법이 지수적으로
# 많고, 뒤에 괄호가 없으면 엔진이 그걸 전부 되돌려 본다(#233 리뷰: 조 번호 뒤 숫자 22자 0.56초, 2자마다 ×4 — 한 줄로 색인
# 워커가 멈춘다). 되돌려 나눠도 숫자 뒤에 괄호가 생길 수는 없으므로 판정은 같다(tests/test_identifiers 동치 검사).
_HEADING_HEAD = re.compile(r'^제?\s*\d+\s*조(?:\s*의?\s*\d+)*+\s*[(（]')


def split_heading_line(line: str) -> tuple[str, str] | None:
    """조 제목 줄이면 (제목 머리, 같은 줄에 이어진 본문), 아니면 None — **조 제목 판정의 정의점**(#233).

    식별자 사전(effective_headings)과 DOCX 조문 폴백(rag/chunking._docx_fallback_headings)이 이 함수 하나를 쓴다.
    제목 = 줄 머리 조 번호 + 괄호 제목이고, 짝 맞는 닫는 괄호 **뒤가 줄끝이거나 공백**이다(공백이면 그 뒤는 같은 문단에 이어진
    본문 — "제8조(역할과 책임) ① 운영관리자는 …" 워드 꼴). 괄호는 짝을 세어 넘는다(전각 포함): '제2조(시향 예약(단체))'.
    괄호가 줄 안에서 안 닫히면 제목이다 — 레이아웃 모델이 닫는 괄호를 다음 줄로 떨어뜨린 실문서 꼴('제 7 조 ( 징계종류').
    인용으로 보는 것: 괄호 없는 꼴("제98조에 의한", "제98조 및 제99조에", "제98조 제2항의"), 닫는 괄호에 조사가 바로 붙는 꼴
    ("제98조(보상)에 따라"), 괄호 안이 '이하'로 시작하는 정의 꼴("제98조(이하 '보상조항'이라 한다)에 따라").

    알려진 한계(#233 결정 2 — 테스트가 고정한다): 닫는 괄호 뒤가 공백이면 인용도 제목이 된다("제98조(보상) 규정에 따라") —
    워드 꼴 제목+본문과 글자 꼴로 가를 수 없다. 괄호 없이 적은 진짜 제목('제8조 목적')은 제목으로 보지 않는다 — 인용
    ("제98조 및 …")과 가를 수 없다. 닫는 괄호 뒤에 공백 없이 괄호·기호가 붙은 제목('제12조(휴가)(연차)', '제5조(목적)[개정 2020]')은
    인용으로 본다 — 조사가 붙은 인용과 같은 규칙에 걸린다(#233 리뷰, 실문서 사례 없음). 어느 쪽이든 글은 본문이나 제목 경로에
    남는다(원문 소실 0 — eval/chunk_invariants).
    """
    s = line.strip()
    m = _HEADING_HEAD.match(s)
    if not m or s[m.end():].lstrip().startswith('이하'):
        return None
    depth = 1
    for i in range(m.end(), len(s)):
        if s[i] in '(（':
            depth += 1
        elif s[i] in ')）':
            depth -= 1
            if depth == 0:
                rest = s[i + 1:]
                return (s[:i + 1], rest.strip()) if not rest or rest[0].isspace() else None
    return s, ''


def is_heading_line(line: str) -> bool:
    """조 제목 줄인가 — 판정은 split_heading_line."""
    return split_heading_line(line) is not None


def effective_headings(headings: list[str], text: str) -> list[str]:
    """조각의 헤딩 줄 — 청킹이 기록한 것(rag/chunking meta['headings']) ∪ 본문의 헤딩 꼴 줄. 본문 등장 순, 중복 제거.

    기록만 믿으면 빠진다(#238 결정 15-3): PDF 레이아웃 모델(docling)이 같은 꼴의 조 제목을 어떤 줄은 헤딩으로, 어떤 줄은
    본문으로 분류한다 — 실측 회계 규정 112조 중 35조(31%)가 기록에 없었다. 본문 꼴 줄로 보충한다. 기록된 헤딩 중 본문에
    없는 것(제목 경로로 올라간 것)은 그대로 둔다.
    """
    recorded = {h.strip() for h in headings or ()}
    out: list[str] = []
    for line in text.split('\n'):
        line_ = line.strip()
        if line_ and (line_ in recorded or is_heading_line(line_)):
            out.append(line_)
    out.extend(h.strip() for h in headings or () if h.strip() not in out)
    return list(dict.fromkeys(out))


def heading_identifier(line: str) -> str | None:
    """헤딩 한 줄의 식별자 — **줄 머리의 것 하나만**. 없으면 None.

    헤딩 줄에 제목과 본문이 붙어 오는 문서가 있다: "제93조 (지점서류의 인계 및 보관) 각 지점은 이 규정 제10조에 의하여 …".
    줄 전체를 정규화하면 본문 속 인용 '10조'까지 이 조각의 이름표가 되어, "10조" 질문에 엉뚱한 조각이 가산을 받고
    오리기도 이 줄에서 잘린다(#238 골드 작성 중 실측, 회계 규정). 질문은 이 함수가 아니라 normalize_identifiers를 쓴다.
    """
    tokens = normalize_identifiers(line)
    return tokens[0] if tokens and re.match(r'^\s*제?\s*\d', line) else None


def chunk_identifiers(heading_path: list[str], headings: list[str]) -> list[str]:
    """조각의 식별자 사전 — 제목 경로 + 헤딩 줄(effective_headings의 결과)에서 줄마다 줄 머리 식별자 하나씩."""
    out = [heading_identifier(h) for h in [*(heading_path or []), *(headings or [])]]
    return list(dict.fromkeys(t for t in out if t))


def cut_identifier_span(text: str, headings: list[str], token: str) -> str:
    """묶음 조각 본문에서 `token`에 해당하는 헤딩의 구간(그 헤딩 줄부터 다음 기록 헤딩 직전까지)만 남긴다.

    모델이 묶음(제15조 표+제16조+제17조)을 통째로 받으면 옆 조를 읽고 답한다(#238 실측: 정답을 1위로 올린 뒤에도
    묶음이면 옆 조로 답함). 구간만 넘기면 23건 중 18→21 정답, 회귀 0. 색인·순위는 건드리지 않는다 — 넘기는 글만.
    헤딩이 본문 줄에 없으면(조 하나가 조각 하나라 제목 경로에만 있는 경우) 원문 그대로다.
    """
    lines = text.split('\n')
    heads = {h.strip() for h in headings or ()}
    idx = [i for i, line in enumerate(lines) if line.strip() in heads]
    for n, i in enumerate(idx):
        if heading_identifier(lines[i]) == token:
            j = idx[n + 1] if n + 1 < len(idx) else len(lines)
            return '\n'.join(lines[i:j])
    return text
