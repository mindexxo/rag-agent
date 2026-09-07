"""어휘 채널(#135) — bigram 토큰화 + Okapi BM25 재점수. 0층 leaf (다른 rag 모듈을 안 씀).

토크나이저·BM25 산식·tsquery 이스케이프의 **정의점**이다. 운영(retriever·documents·
faq_indexing)과 측정(eval/_hybrid_ablation)이 이 함수들을 같이 써야 어블레이션 눈금이
운영을 그대로 예언한다 — 두 벌로 갈라지면 조립 검증(#128 규율)이 무너진다.

구조(설계 정본은 이슈 #133·#135):
- 색인 시: 임베딩과 동일한 index_text(문서='파일명>헤딩'+본문, FAQ=원문)를 bigram으로
  잘라 chunks.lex_tsv(토큰 집합, GIN)·lex_len(토큰 총수=dl)에 저장. 통계 테이블은 없다 —
  df·N·avgdl은 질의 시 SQL·후보에서 계산한다(#135 B안: 유지보수 0·드리프트 원리적 불가,
  승격 트리거는 config.hybrid_lexical_enabled 주석 참조).
- 질의 시: 질의 토큰과 하나라도 겹치는 청크(=score>0 가능한 전체)를 후보로 회수한 뒤
  앱에서 BM25 재점수. **df는 후보만으로 정확하다** — 질의 토큰 t를 가진 청크는 전부
  후보에 들어오므로(후보 = 어느 토큰이든 매칭) df(t)=후보 중 t 보유 수가 곧 코퍼스 df다.

토크나이저 = 어절 내 문자 bigram. kiwi(형태소)와의 A/B 실측(#133, 450문항)에서 단독 리콜
96.9% vs 97.6%·최종 동률로 사실상 무차이였고, kiwi는 문맥 의존 절단('아더에러'가 문장에선
통째, 단독에선 오분할 — 색인문↔질의문 비대칭 매칭 실패 위험)·사전 관리 비용이 있어 bigram
확정. 교체는 이 함수 하나 + 재색인(백필) 1회다.
"""
import math


def bigrams(text: str) -> list[str]:
    """어절 내 문자 bigram. 1글자 어절은 그대로 토큰.

    어절 **안에서만** 자른다 — 공백 제거 후 절단(구판 어블레이션)은 "반품 기간"→"품기"
    같은 어절 경계를 넘는 가짜 조각을 만들었다(#133에서 교정). 중복 토큰을 유지한
    리스트를 반환한다 — BM25의 tf(빈도)·dl(길이)이 중복을 세기 때문. tsvector용
    집합화는 tsvector_lexemes가 담당.
    """
    out = []
    for w in text.lower().split():
        if len(w) < 2:
            out.append(w)
        else:
            out.extend(w[i:i + 2] for i in range(len(w) - 1))
    return out or [text.lower()]


def tsvector_lexemes(tokens: list[str]) -> list[str]:
    """array_to_tsvector 입력용 — 중복 제거·정렬(결정적 저장). PG도 dedup하지만
    바인딩 크기를 줄이고 값 검증(테스트의 저장값 대조)을 결정적으로 만들기 위함."""
    return sorted(set(tokens))


def tsquery_or(tokens: list[str]) -> str:
    """토큰들을 OR 결합한 tsquery 문자열 — `'s-' | '아더' | ...` 형태.

    따옴표 리터럴이라 PG 파서가 토큰을 재해석하지 않는다(bigram의 구두점 조각 `s-`도
    lexeme 그대로 매칭 — plainto_tsquery류는 재토큰화해서 못 쓴다). 이스케이프는
    `\\`→`\\\\` 먼저, `'`→`''` 다음 — tsquery 따옴표 리터럴 안에서 역슬래시가 이스케이프
    문자라서다(개발계 PG 16.14에서 적대 토큰 8종 값 검증, #135).
    빈 tokens는 호출부가 걸러라 — 빈 tsquery는 문법 오류다.
    """
    return " | ".join(
        "'" + t.replace("\\", "\\\\").replace("'", "''") + "'" for t in dict.fromkeys(tokens)
    )


def bm25_rank(
    query: str,
    cand_tokens: dict[int, list[str]],
    n_docs: int,
    avgdl: float,
    *,
    k1: float = 1.5,
    b: float = 0.75,
    tokenize=bigrams,
) -> list[int]:
    """후보(id→토큰열)를 Okapi BM25로 재점수해 점수 내림차순 id를 반환.

    n_docs·avgdl은 **코퍼스 전체**(검색 가능 청크) 기준으로 호출부가 넘긴다 —
    df만 후보에서 세도 정확하다(모듈 docstring). 산식은 eval/_hybrid_ablation.Bm25와
    동일해야 한다(테스트 test_hybrid_lexical이 점수 단위로 대조).
    idf = ln(1 + (N-df+0.5)/(df+0.5)), score = Σ idf·tf·(k1+1)/(tf + k1·(1-b+b·dl/avgdl)).
    """
    tf_map: dict[int, dict[str, int]] = {}
    dl: dict[int, int] = {}
    df: dict[str, int] = {}
    for cid, toks in cand_tokens.items():
        tf: dict[str, int] = {}
        for t in toks:
            tf[t] = tf.get(t, 0) + 1
        tf_map[cid] = tf
        dl[cid] = len(toks)
        for t in tf:
            df[t] = df.get(t, 0) + 1

    avgdl = avgdl or 1.0
    scores: dict[int, float] = {}
    for t in tokenize(query):          # 질의 토큰 중복 유지 — 반복 토큰은 그만큼 가산 (측정과 동일)
        d = df.get(t)
        if not d:
            continue
        idf = math.log(1 + (n_docs - d + 0.5) / (d + 0.5))
        for cid, tf in tf_map.items():
            f = tf.get(t)
            if not f:
                continue
            denom = f + k1 * (1 - b + b * dl[cid] / avgdl)
            scores[cid] = scores.get(cid, 0.0) + idf * f * (k1 + 1) / denom
    return [cid for cid, _ in sorted(scores.items(), key=lambda x: -x[1])]
