"""조 번호 질문 검색축(#238) 순수 로직 — 문서명 정리·조 추출·정답 집합·질문 생성. DB·엔진 불필요.

파일명은 실문서와 **같은 모양의 가짜**다 — 저장소가 공개라 실문서 이름을 박지 않는다.
"""
import random
import re

import pytest

from eval.retrieval_article import (_typo, article_keys, build_gold, build_items, classify_mix,
                                    clean_doc_name, make_query, pick_bare)
from rag.search.retrieval_types import RetrievedChunk


@pytest.mark.parametrize("filename,expected", [
    ("가나_복무규칙_20260701.pdf", "가나 복무규칙"),
    ("[인티큐브] 보상규정 Ver1.8.pdf", "보상규정"),
    ("[인티큐브] 상벌규정 Ver1.4 (1).pdf", "상벌규정"),
    ("[총무 _ 표준] 출장 표준.pdf", "출장 표준"),
    ("4_한솔인티큐브 장비관리 지침서_8.0.docx", "장비관리 지침서"),
    ("경비처리규정.pdf", "경비처리규정"),
])
def test_clean_doc_name(filename, expected):
    assert clean_doc_name(filename) == expected


def test_article_keys_heading_and_packed_headings():
    # 묶음 조각: 제목 경로의 조 + 청킹이 기록한 묶음 헤딩 — 전부 이 조각의 조다. '장'은 조가 아니다
    assert article_keys(["제5장 휴가", "제 47 조(경조휴가)"], ["제48조(장기근속휴가)", "제 48 조 2(특별휴가)"]) == ["47조", "48조", "48조의2"]


def test_article_keys_ignores_inline_reference():
    # 본문은 보지 않는다 — "제48조에 따라" 같은 참조는 기록 헤딩이 아니라 사전에 못 들어간다
    assert article_keys(["부칙"], []) == []


def test_build_gold_collects_all_chunks_of_split_article():
    chunks = [
        {"chunk_id": 1, "document_id": 10, "heading_path": ["제3조(채용)"], "text": "앞부분"},
        {"chunk_id": 2, "document_id": 10, "heading_path": ["제3조(채용)"], "text": "뒷부분"},   # 512토큰 분할 이어짐
        {"chunk_id": 3, "document_id": 20, "heading_path": ["제3조(정의)"], "text": "다른 규정"},
    ]
    gold, keys_of = build_gold(chunks)
    assert gold[(10, "3조")] == {1, 2}
    assert gold[(20, "3조")] == {3}       # 같은 번호라도 문서가 다르면 다른 정답
    assert keys_of[1] == ["3조"]


def test_make_query_never_alters_article_number():
    rng = random.Random(0)
    for style in (0, 1, 2):
        for _ in range(50):
            q = make_query("복무규칙", "76조의2", style, rng)
            assert re.search(r"76조의2", q)


def test_make_query_styles():
    rng = random.Random(0)
    assert make_query("복무규칙", "48조", 0, rng) == "복무규칙 제48조의 내용을 알려주세요"
    assert make_query("복무규칙", "48조", 1, rng) == "복무규칙 48조 뭐야"


def test_typo_changes_one_syllable_tail_only():
    rng = random.Random(1)
    out = _typo("복무규칙", rng)
    assert len(out) == 4 and sum(a != b for a, b in zip(out, "복무규칙")) == 1


def test_build_items_deterministic_and_rotates_styles():
    chunks = [{"chunk_id": i, "document_id": 10, "filename": "가나_복무규칙_20260701.pdf",
               "heading_path": [f"제{i}조(항목)"], "text": ""} for i in range(1, 7)]
    gold, _ = build_gold(chunks)
    a, b = build_items(chunks, gold, "t"), build_items(chunks, gold, "t")
    assert [x["query"] for x in a] == [x["query"] for x in b]
    assert [x["style"] for x in a] == [0, 1, 2, 0, 1, 2]
    assert [x["article"] for x in a] == [f"{i}조" for i in range(1, 7)]   # 숫자 순(문자열 순 아님)


def _ch(cid, doc):
    return RetrievedChunk(chunk_id=cid, document_id=doc, text="", heading_path=[], page=None,
                          filename="f", version=1)


def test_classify_mix_same_doc_adjacent_and_other_doc():
    item = {"gold_ids": [1], "document_id": 10, "article": "48조"}
    keys_of = {1: ["48조"], 2: ["49조"], 3: ["60조"], 4: ["48조"]}
    top = [_ch(1, 10), _ch(2, 10), _ch(3, 10), _ch(4, 20)]
    assert classify_mix(item, top, keys_of) == {"mix_same_doc": 2, "mix_adjacent": 1, "mix_other_doc": 1}


def test_pick_bare_only_numbers_shared_by_several_docs():
    gold = {(10, "1조"): {1}, (20, "1조"): {2}, (10, "2조"): {3}}
    assert pick_bare(gold) == ["1조"]
