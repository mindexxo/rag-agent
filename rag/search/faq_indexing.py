"""FAQ 항목 하나 → 검색 청크 텍스트 하나의 조립 규약."""


def build_faq_chunk_text(question: str, variants: list[str], answer: str) -> str:
    """임베딩·어휘 채널 입력이 이 텍스트 그대로다(프리픽스 없음) — 형태를 바꾸면 전 FAQ 재색인이다."""
    lines = [f"Q: {question}"]
    clean = [v.strip() for v in variants if v.strip()]
    if clean:
        lines.append(f"(유사 질문: {', '.join(clean)})")
    lines.append(f"A: {answer}")
    return "\n".join(lines)
