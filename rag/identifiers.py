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
    # **괄호 제목이 뒤따를 때만** 가지로 읽는다 — '제9조 2항'은 참조다. 운영 DOCX 폴백의 같은 결함은 #250.
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


# 헤딩 꼴 줄 — 줄 머리가 식별자이고 **괄호 제목이 뒤따른다**. 인용("제10조 2항에 의거", "제18조의2제1항을 위반")은
# 괄호 제목이 없어 걸리지 않는다(실측 585문항·실질문 23건 오인식 0, 2026-10-17).
_HEADING_LINE = re.compile(r'^제?\s*\d+\s*조(?:\s*의?\s*\d+)*\s*[(（]')


def is_heading_line(line: str) -> bool:
    return bool(_HEADING_LINE.match(line.strip()))


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


def chunk_identifiers(heading_path: list[str], headings: list[str]) -> list[str]:
    """조각의 식별자 사전 — 제목 경로 + 헤딩 줄(effective_headings의 결과를 넘긴다). 본문 전체는 보지 않는다."""
    out: list[str] = []
    for h in [*(heading_path or []), *(headings or [])]:
        out.extend(normalize_identifiers(h))
    return list(dict.fromkeys(out))


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
        if token in normalize_identifiers(lines[i]):
            j = idx[n + 1] if n + 1 < len(idx) else len(lines)
            return '\n'.join(lines[i:j])
    return text
