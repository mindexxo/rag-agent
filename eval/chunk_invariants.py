"""청킹 불변식 — **"원문 소실 0"의 정의점** (#233 결정 1).

제목 판정(rag/identifiers.is_heading_line)은 규칙이라 틀릴 수 있다. 그래서 약속은 "판정이 틀려도 원문 글자는 하나도
사라지지 않는다"이고, 그 약속을 여기 한 곳에서 정의해 두 층이 같이 쓴다:
  - 조립 단위 — `build_sections`에 넣은 헤딩·본문이 섹션의 제목 경로나 본문 어딘가에 남는가(sections_lost)
  - 문서 단위 — 문서를 운영 청킹(chunk_file)에 돌린 조각의 제목 경로·본문 어딘가에 원문 줄이 남는가(chunks_lost)
eval/_golden_chunks.py --diff의 "사라진 줄"도 같은 판정(missing_lines)을 쓴다. 여기(eval/)에 두는 이유: 테스트와 도구가 같이
쓰는 정의점이고, 의존 방향은 tests → eval이 저장소 관례다(반대 방향은 없다, #233 리뷰 지적 10).

"남는다"는 글자 비교다. 공백·표 구분자('|')는 청킹이 모양을 바꾸므로(표 렌더·줄바꿈·SentenceSplitter) 무시한다.
조각 경계에 걸친 줄(SentenceSplitter 분할)은 본문을 이어 붙인 글에서 찾는다. 제목 경로는 원소 하나씩만 본다.
"""
import re

_IGNORED = re.compile(r'[\s|]+')


def norm(text: str) -> str:
    """비교용 정규화 — 공백·표 구분자 제거."""
    return _IGNORED.sub('', text)


def _stitch(pieces: list[str]) -> str:
    """이웃 조각을 겹침(SentenceSplitter overlap) 없이 잇는다 — 앞 글의 꼬리와 뒤 글의 머리가 같은 가장 긴 부분을 한 번만."""
    out = ''
    for p in pieces:
        k = next((k for k in range(min(len(out), len(p)), 0, -1) if out.endswith(p[:k])), 0)
        out += p[k:]
    return out


def missing_lines(lines, heads: list[str], bodies: list[str]) -> list[str]:
    """`lines` 중 제목 경로 원소(heads)·본문(bodies) 어디에도 없는 것 — 아래 두 함수와 _golden_chunks --diff의 공통 판정."""
    pieces = [norm(h) for h in (*heads, *bodies)]
    # 경계에 걸친 줄은 본문끼리만 이어 붙여 찾는다 — 제목 경로까지 이으면 "제98조(보상)"(제목으로 잘못 올라감) + "에 따라
    # 지급한다"(본문)가 이어져 **깨진 문장이 멀쩡해 보인다**.
    joined = _stitch([norm(b) for b in bodies])
    out = []
    for line in lines:
        n = norm(line)
        if n and not any(n in p for p in pieces) and n not in joined:
            out.append(line)
    return out


def sections_lost(items, sections) -> list[str]:
    """`build_sections` 입력 항목 중 결과 섹션의 제목 경로·본문 어디에도 없는 글. 표 블록은 렌더한 글로 본다."""
    texts = [t.render() if hasattr(t, 'render') else t for _, t, _ in items]
    return missing_lines(texts, [h for s in sections for h in s.heading_path], [s.body for s in sections])


def chunks_lost(lines, chunks) -> list[str]:
    """원문 줄 중 조각(ChunkData)들의 제목 경로·본문 어디에도 없는 줄."""
    return missing_lines(lines, [h for c in chunks for h in c.heading_path], [c.text for c in chunks])
