"""토큰 근사 추정 (F100).

정확한 토크나이저 대신 문자 기반 근사. 30K 천장 대비 실사용(~9K)이 작아
정밀 카운트가 막아줄 초과 사고가 없으므로 근사로 충분하다.
천장에 바짝 붙여 쓰는 설계로 가면 그때 실제 토크나이저로 교체.
"""


def estimate_tokens(text: str) -> int:
    """한국어 ~1.5자/토큰. 초과 방지 방향으로 보수적(1.4로 나눠 과대 추정)."""
    return int(len(text) / 1.4) + 1


# 보수적 추정 계수 — 표 전체(헤더·구분선 포함) 실측 글자/토큰(생성 모델 Qwen3 /tokenize, 2026-10-12, #222):
# 실제·모의 엑셀 131표 최소 1.223·중앙 1.348, 숫자만 든 합성 표 최소 1.142. 최소보다 낮게 잡아 **추정 ≥ 실제**가 되게 한다.
# 위 estimate_tokens(÷1.4)는 산문 대화 이력용 근사라 숫자·표가 많은 근거에선 실제보다 20% 넘게 적게 센다 — 상한·예산
# 판정에는 이쪽을 쓴다(엑셀 크기 상한 rag.xlsx_chunking, 근거 예산 rag.retriever.prepare_evidence). 강제: tests/test_xlsx_token_estimate.
CONSERVATIVE_CHARS_PER_TOKEN = 1.1


def estimate_tokens_conservative(text: str) -> int:
    """상한·예산 판정용 보수적 추정 — 글자 수 ÷ 1.1(올림). 실측상 어떤 표든 실제 토큰 이상이다."""
    return -(-len(text) * 10 // 11)   # ceil(len / 1.1) — 부동소수 없이
