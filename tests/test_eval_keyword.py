"""키워드형 질문 검색축(#248) 순수 로직 — 토큰 추출·정답 집합·표본·질문 문체. DB·엔진 불필요.

코드·용어는 전부 지어낸 것이다(저장소 공개). Nori 출력은 실제 응답(2026-10-18 스파이크)과 같은 모양의 가짜다.
"""
import random

import pytest

from eval.axis_common import evidence_has, norm_text
from eval.retrieval_keyword import (CAP_PER_TENANT, KINDS, TAILS, build_items, extract_acronyms, extract_codes,
                                    gold_of, make_query, noun_terms, sample_tokens, select_candidates)


@pytest.mark.parametrize("text,expected", [
    ("원료 AR-2003은 머스크 계열", {"AR-2003"}),
    ("농가 HF-101, 모델 SB-1000·P001", {"HF-101", "SB-1000", "P001"}),
    ("HPG-A 등급", set()),                       # 숫자 없음 — 코드 아님(약어 쪽 규칙도 'HPG'만 본다)
    ("2026-10-18 기준 30일", set()),              # 날짜·숫자만 — 문자가 없다
    ("ABC123DEF", set()),                         # 앞뒤가 문자에 붙은 덩어리 — 한 덩어리 코드가 아니다
    ("D-1 마감", set()),                          # 3자 미만
    ("문서번호 SUM-MB-012 참조", {"SUM-MB-012"}),  # 이어진 문자 마디는 통째로 — 'MB-012'만 떼지 않는다
    ("Ver1.8 적용, v1.2 개정", set()),             # 버전 표기는 코드가 아니다
    ("하이픈 없는 SB1000 모델", {"SB1000"}),
])
def test_코드는_문자와_숫자가_함께인_덩어리만(text, expected):
    assert extract_codes(text) == expected


@pytest.mark.parametrize("text,expected", [
    ("CTI 연동과 DBMS 백업, HACCP 인증", {"CTI", "DBMS", "HACCP"}),
    ("첨부 PDF·XLSX 파일", set()),                # 파일 형식명은 묻는 대상이 아니다
    ("Matching Grant", set()),                    # 대문자 시작 단어는 약어가 아니다
    ("ABCDEFG", set()),                           # 7자 이상
    ("문서 HP-OPS-015", set()),                   # 코드의 일부 — 약어가 아니다
    ("모델 SB1000·HF101", set()),                 # 숫자가 붙은 코드의 앞부분 — 약어가 아니다
])
def test_약어는_대문자_2에서_6자(text, expected):
    assert extract_acronyms(text) == expected


def _tok(token, start, end, pos, ptype="MORPHEME"):
    return {"token": token, "start_offset": start, "end_offset": end, "leftPOS": pos, "posType": ptype}


def test_명사는_붙은_형태소를_잇고_복합어만_남긴다():
    # "시향일지에서 무향 스트립 묶음" — 실제 Nori 모양: 시향+일지(붙음), 묶음(COMPOUND)+묶+음.
    # 형태소 하나짜리 명사(무향·스트립)는 빠진다 — 첫 검수에서 대부분 일반어였다
    tokens = [_tok("시향", 0, 2, "NNG(General Noun)"), _tok("일지", 2, 4, "NNG(General Noun)"),
              _tok("에서", 4, 6, "J(Ending Particle)"),
              _tok("무향", 7, 9, "NNG(General Noun)"), _tok("스트립", 10, 13, "NNG(General Noun)"),
              _tok("묶음", 14, 16, "NNG(General Noun)", "COMPOUND"), _tok("묶", 14, 15, "NNG(General Noun)"),
              _tok("음", 15, 16, "NNG(General Noun)")]
    assert noun_terms(tokens) == {"시향일지", "묶음"}


def test_한_형태소_명사_뒤에도_다음_복합어는_제대로_잡힌다():
    # 버퍼 비우기 회귀 — 앞의 단일 명사가 남아 다음 명사와 잘못 이어지면 안 된다
    tokens = [_tok("향", 0, 1, "NNG(General Noun)"), _tok("을", 1, 2, "J(Ending Particle)"),
              _tok("유산", 3, 5, "NNG(General Noun)"), _tok("기부", 5, 7, "NNG(General Noun)")]
    assert noun_terms(tokens) == {"유산기부"}


def test_명사가_아닌_형태소와_한_글자_명사는_빠진다():
    tokens = [_tok("반반", 0, 2, "MAG(General Adverb)"), _tok("차", 2, 3, "VV(Verb)"),
              _tok("향", 4, 5, "NNG(General Noun)"), _tok("HACCP", 6, 11, "SL(Foreign language)")]
    assert noun_terms(tokens) == set()


def test_정규화는_공백과_대소문자를_무시한다():
    assert norm_text("Ar - 2003\n원료") == "ar-2003원료"
    texts = {1: norm_text("패치 테스트를 한다"), 2: norm_text("PATCH test"), 3: norm_text("무관")}
    assert gold_of("패치테스트", texts) == [1]
    assert gold_of("patch Test", texts) == [2]


def test_정답이_1에서_3조각인_토큰만_후보가_된다():
    texts = {i: norm_text(t) for i, t in enumerate(["AR-1 키트", "AR-1 키트", "AR-1 키트", "AR-1 키트 HF-9", "기타"])}
    out = select_candidates({"AR-1": "code", "HF-9": "code", "없는말": "noun", "키트": "noun"}, texts)
    assert out == {"HF-9": ("code", [3])}          # AR-1·키트는 4조각, 없는말은 0조각


def test_명사만_다른_테넌트에_나오면_뺀다():
    texts = {1: norm_text("유산기부 안내, 연말정산 서류, AR-1 코드")}
    other = norm_text("연말정산은 다른 회사도 한다 AR-1")
    out = select_candidates({"유산기부": "noun", "연말정산": "noun", "AR-1": "code"}, texts, other)
    assert set(out) == {"유산기부", "AR-1"}       # 코드는 다른 회사에 같은 글자가 있어도 남는다


def test_대소문자만_다른_토큰은_하나로():
    texts = {1: norm_text("CTI 연동")}
    assert len(select_candidates({"CTI": "acronym", "cti": "acronym"}, texts)) == 1


def _selected(n_per_kind):
    return {f"{k}{i}": (k, [i]) for k, n in n_per_kind.items() for i in range(n)}


def test_표본은_시드가_같으면_같고_상한과_종류_배분을_지킨다():
    sel = _selected({"code": 50, "acronym": 50, "noun": 50})
    a = sample_tokens(sel, CAP_PER_TENANT, random.Random(1))
    b = sample_tokens(sel, CAP_PER_TENANT, random.Random(1))
    assert a == b and len(a) == CAP_PER_TENANT
    assert {k: sum(x[0] == k for x in a) for k in KINDS} == {k: CAP_PER_TENANT // 3 for k in KINDS}


def test_모자란_종류의_몫은_다른_종류로_채운다():
    picked = sample_tokens(_selected({"code": 5, "acronym": 0, "noun": 100}), 60, random.Random(1))
    assert len(picked) == 60 and sum(x[0] == "code" for x in picked) == 5


def test_후보가_상한보다_적으면_전부():
    assert len(sample_tokens(_selected({"code": 3, "acronym": 2, "noun": 1}), 60, random.Random(1))) == 6


def test_문체는_세_가지를_번갈아_토큰_글자는_그대로():
    qs = [make_query("AR-2003", i) for i in range(9)]
    assert qs[0] == "AR-2003" and qs[2] == "AR-2003 뭐야"
    assert [qs[i] for i in (1, 4, 7)] == [f"AR-2003 {t}" for t in TAILS]
    assert all(q.startswith("AR-2003") for q in qs)


def test_문항은_토큰과_정답을_그대로_싣는다():
    items = build_items("t1", [("code", "AR-2003", [5, 7]), ("noun", "시향", [9])])
    assert [(it["qid"], it["type"], it["query"], it["gold_ids"]) for it in items] == [
        ("kw_t1_000", "keyword_code", "AR-2003", [5, 7]), ("kw_t1_001", "keyword_noun", f"시향 {TAILS[0]}", [9])]


def test_근거_판정은_조각_id가_아니라_넘기는_글로_본다():
    # 표 시트 접기: 정답 행이 든 형제 조각은 id로는 빠지지만, 1위 조각이 시트 완성본으로 바뀌어 그 행을 싣는다
    from rag.search.retrieval_types import RetrievedChunk
    sheet = RetrievedChunk(chunk_id=1, document_id=1, text="| AR-2001 | 벤조인 |\n| AR-2021 | 만다린 |", heading_path=[],
                           page=None, filename="원료.xlsx", version=1, is_table=True)
    assert evidence_has([sheet], "ar-2021") == 1.0
    assert evidence_has([sheet], "AR-2099") == 0.0
