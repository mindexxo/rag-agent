"""어절 내 문자 bigram 토크나이저 — 어휘 bigram 필드의 정의점.

`rag/search/os_index.py`가 이 함수로 `lex_bigram` 필드를 만든다(공백 결합 — 엔진은 공백으로만 다시 자른다).
운영 어휘 채널은 Nori 필드를 쓰고 이 필드는 대조군·실험용이지만, 색인문과 질의문이 같은 방식으로 잘리도록 한 곳에 둔다.
"""


def bigrams(text: str) -> list[str]:
    """어절 내 문자 bigram. 1글자 어절은 그대로 토큰.

    어절 안에서만 자른다(어절 경계를 넘는 가짜 조각 방지). 중복을 유지한다 — BM25의 tf·dl이 중복을 센다.
    """
    out = []
    for w in text.lower().split():
        if len(w) < 2:
            out.append(w)
        else:
            out.extend(w[i : i + 2] for i in range(len(w) - 1))
    return out or [text.lower()]
