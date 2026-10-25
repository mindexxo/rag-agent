"""문자 수 기반 토큰 근사 추정 — 컨텍스트 천장 대비 실사용이 작아 실제 토크나이저 없이 근사로 충분하다."""


def estimate_tokens(text: str) -> int:
    """산문 대화 이력용 근사 — 한국어 ~1.5자/토큰을 1.4로 나눠 약간 과대 추정."""
    return int(len(text) / 1.4) + 1


# 상한·예산 판정용 계수 — 표 계열 근거의 실제 글자/토큰 최솟값보다 낮게 잡아 **추정 ≥ 실제**가 되게 한다.
# estimate_tokens(÷1.4)는 숫자·표가 많은 근거를 적게 세므로 엑셀 크기 상한·근거 예산은 이쪽을 쓴다.
# 강제: tests/test_xlsx_token_estimate.
CONSERVATIVE_CHARS_PER_TOKEN = 1.1


def estimate_tokens_conservative(text: str) -> int:
    return -(-len(text) * 10 // 11)  # ceil(len / 1.1) — 부동소수 없이
