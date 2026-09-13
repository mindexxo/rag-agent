"""어절 내 문자 bigram 토크나이저 — 어휘 필드의 **정의점**. 0층 leaf (다른 rag 모듈을 안 씀).

색인 시 `rag/opensearch.build_doc`이 이 함수로 `lex_bigram` 필드를 만든다(공백 결합 문자열,
엔진은 공백으로만 다시 자른다 — 매핑 주석). 운영 어휘 채널은 엔진의 Nori 필드를 쓰고
bigram 필드는 대조군·실험용이지만, 토크나이저가 코드 한 곳에 있어야 색인문과 질의문이
같은 방식으로 잘린다.

토크나이저 = 어절 내 문자 bigram. kiwi(형태소)와의 A/B 실측(#133, 450문항)에서 단독 리콜
96.9% vs 97.6%·최종 동률로 사실상 무차이였고, kiwi는 문맥 의존 절단('아더에러'가 문장에선
통째, 단독에선 오분할)·사전 관리 비용이 있어 bigram 확정.

(구) PG FTS용 tsquery 이스케이프·앱 BM25 산식은 #139 OpenSearch 도입으로 제거했다 —
BM25는 엔진(Lucene, k1·b는 rag/opensearch.py 매핑)이 계산한다.
"""


def bigrams(text: str) -> list[str]:
    """어절 내 문자 bigram. 1글자 어절은 그대로 토큰.

    어절 **안에서만** 자른다 — 공백 제거 후 절단(구판 어블레이션)은 "반품 기간"→"품기"
    같은 어절 경계를 넘는 가짜 조각을 만들었다(#133에서 교정). 중복 토큰을 유지한
    리스트를 반환한다 — BM25의 tf(빈도)·dl(길이)이 중복을 세기 때문.
    """
    out = []
    for w in text.lower().split():
        if len(w) < 2:
            out.append(w)
        else:
            out.extend(w[i:i + 2] for i in range(len(w) - 1))
    return out or [text.lower()]
