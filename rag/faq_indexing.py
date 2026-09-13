"""FAQ 항목 → 검색 청크 텍스트 조립 (F3).

항목 하나가 청크 하나다. 색인은 라우터가 같은 트랜잭션에 남긴 outbox 행(INDEX_FAQ)을
워커가 처리하며 한다(rag/opensearch.index_faq_chunks — 이 텍스트를 다시 조립해 임베딩·색인).
PG에는 청크를 쓰지 않는다(#139) — 이 모듈에 남은 것은 텍스트 조립 규약 하나다.
"""


def build_faq_chunk_text(question: str, variants: list[str], answer: str) -> str:
    """질문+유사질문+답변을 한 청크로 — 질문 표현들이 dense 매칭을 견인한다.

    임베딩·어휘 채널 입력 모두 이 텍스트 **그대로**다(문서 청크와 달리 '파일명>헤딩'
    프리픽스가 없다 — rag/index_text.py의 비대칭). 형태를 바꾸면 전 FAQ 재색인이다.
    """
    lines = [f"Q: {question}"]
    clean = [v.strip() for v in variants if v.strip()]
    if clean:
        lines.append(f"(유사 질문: {', '.join(clean)})")
    lines.append(f"A: {answer}")
    return "\n".join(lines)
