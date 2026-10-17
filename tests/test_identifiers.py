"""식별자 채널(#238) 순수 로직 — 정규화·사전·구간 오리기·가산 재정렬·근거 준비·청킹 헤딩 기록. DB·엔진 불필요.

문서 텍스트는 전부 가짜다(저장소 공개). 실문서에서 관측된 **표기 꼴**만 옮겼다.
"""
import pytest
from llama_index.core.node_parser import SentenceSplitter

from rag import chunking
from rag.identifiers import chunk_identifiers, cut_identifier_span, effective_headings, is_heading_line, normalize_identifiers
from rag.retrieval_types import RetrievedChunk
from rag.retriever import apply_identifier_boost, prepare_evidence


# ----- 정규화: 색인(헤딩)과 질문에 같은 함수 ---------------------------------------

@pytest.mark.parametrize("text,tokens", [
    ("취업규칙 11조 뭐야", ["11조"]),
    ("취업규칙 제11조의 내용을 알려주세요", ["11조"]),         # '조의 내용'의 '의'는 조사
    ("제 11 조(사무인계 의무)", ["11조"]),                        # 헤딩 띄어쓰기
    ("제43조의2(태아검진 시간의 허용)", ["43조의2"]),
    ("제 43 조 2(태아검진 시간의 허용)", ["43조의2"]),           # '의' 생략 표기 — 괄호 제목이 붙을 때만 가지
    ("제 43 조 3의 2(배우자 유산·사산휴가)", ["43조의3의2"]),
    ("제10조의 2 (증빙)", ["10조의2"]),
    ("제9조 2항", ["9조"]),                                      # 괄호 제목 없음 — 참조, 가지 아님
    ("제 3 장 복 무", ["3장"]),
    ("남녀고용평등법 18조의2 확인좀", ["18조의2"]),
    ("3조건이 있다", []),                                        # '조'가 단어의 일부
    ("매출 1조원", []),
    ("연차 며칠이야", []),                                       # 일반 질문 — 채널 미발동
    ("제1장 총칙 제1조(목적)", ["1장", "1조"]),                  # 순서 보존, 중복 제거
])
def test_normalize(text, tokens):
    assert normalize_identifiers(text) == tokens


def test_chunk_identifiers_는_제목_경로와_기록_헤딩만_본다():
    # 본문 속 "제18조의2에 따른"은 인수로 넘어오지 않는다 — 인용이 사전에 들어가면 안 된다(참조 문구 오인)
    assert chunk_identifiers(["가나 규칙", "제 2 장 인 사"], ["제 9 조(고용관계)", "제 11 조(사무인계)"]) == ["2장", "9조", "11조"]
    assert chunk_identifiers(["제38조(양벌규정)"], []) == ["38조"]
    assert chunk_identifiers(["가나 규칙 Ver1.8"], []) == []


@pytest.mark.parametrize("line,ok", [
    ("제10조 (장부와 전표 및 증빙서류의 보관)", True),
    ("제80조(삭제)", True),
    ("제 43 조 2(태아검진 시간의 허용)", True),
    ("제 43 조 3의 2(배우자 유산·사산휴가)", True),
    ("제10조의 2 (증빙)", True),
    ("제10조 2항에 의거 보관한다.", False),        # 인용 — 괄호 제목 없음
    ("제18조의2제1항을 위반하여", False),
    ("① 사원이 다음에 해당할 경우(정년)", False),  # 줄 머리가 식별자가 아님
])
def test_is_heading_line(line, ok):
    assert is_heading_line(line) is ok


def test_effective_headings_는_기록과_본문_꼴을_합친다():
    text = "제8조 (책임)\n본문\n제10조 (보관)\n1. 보존은 회계부서\n제10조 2항에 의거"   # 제10조는 docling이 본문으로 분류한 경우
    assert effective_headings(["제8조 (책임)"], text) == ["제8조 (책임)", "제10조 (보관)"]
    assert effective_headings([], "① 본문뿐") == []
    assert effective_headings(["제 3 장 복 무"], "본문") == ["제 3 장 복 무"]       # 기록은 본문에 없어도 유지


# ----- 구간 오리기 ---------------------------------------------------------------

_PACKED = "제 15 조(기준)\n| a | b |\n| 1 | 2 |\n제 16조(운영기준)\n정기 승격은 심의를 거친다\n제 17 조(기준일)\n기준일은 3월 1일"
_HEADS = ["제 15 조(기준)", "제 16조(운영기준)", "제 17 조(기준일)"]


def test_cut_은_그_조_구간만_남긴다():
    assert cut_identifier_span(_PACKED, _HEADS, "16조") == "제 16조(운영기준)\n정기 승격은 심의를 거친다"
    assert cut_identifier_span(_PACKED, _HEADS, "17조") == "제 17 조(기준일)\n기준일은 3월 1일"      # 마지막 구간
    assert cut_identifier_span(_PACKED, _HEADS, "15조").startswith("제 15 조(기준)\n| a | b |")      # 표 줄은 구간 안에


def test_cut_은_헤딩이_본문에_없으면_원문_그대로():
    assert cut_identifier_span("① 본문뿐", [], "16조") == "① 본문뿐"           # 조 하나짜리 조각 — 제목 경로에만 있음
    assert cut_identifier_span(_PACKED, _HEADS, "99조") == _PACKED               # 토큰 불일치


# ----- 가산 재정렬 + 근거 준비 ----------------------------------------------------

def _ch(cid, idents=(), heads=(), text="", doc=10):
    return RetrievedChunk(chunk_id=cid, document_id=doc, text=text, heading_path=["문서", "제 2 장"], page=None,
                          filename="f.pdf", version=1, identifiers=list(idents), headings=list(heads))


def test_boost_는_정확_일치만_올리고_표시를_남긴다():
    # 리랭커 순서: 제11장 조각(0.97) > 정답 묶음(0.72). '111조'·'11장'은 '11조'와 다른 토큰이라 가산 없음
    chunks = [_ch(1, ["11장", "111조"]), _ch(2, ["9조", "10조", "11조"])]
    out, scores, qi = apply_identifier_boost(chunks, [0.97, 0.72], [0, 0], ["11조"], 0.3)
    assert [c.chunk_id for c in out] == [2, 1]
    assert scores == [pytest.approx(1.02), 0.97]
    assert out[0].matched_ident == "11조" and out[1].matched_ident is None


def test_boost_는_리랭커_확신을_뒤집지_않는다():
    chunks = [_ch(1, ["32조의1"]), _ch(2, ["32조"])]
    out, _, _ = apply_identifier_boost(chunks, [0.95, 0.50], [0, 0], ["32조"], 0.3)   # 0.45 차이 > 0.3
    assert [c.chunk_id for c in out] == [1, 2]


def test_prepare_evidence_는_걸린_묶음만_오린다():
    packed = _ch(2, ["15조", "16조", "17조"], _HEADS, _PACKED)
    hit = apply_identifier_boost([packed, _ch(3, ["16조"], [], "다른 규정 16조 본문")], [0.6, 0.9], [0, 0], ["16조"], 0.3)[0]
    ev = prepare_evidence(hit, 5)
    cut = next(c for c in ev if c.chunk_id == 2)
    assert cut.text == "제 16조(운영기준)\n정기 승격은 심의를 거친다"
    assert cut.heading_path == ["문서", "제 2 장"]                                   # 라벨 불변(결정 18)
    assert next(c for c in ev if c.chunk_id == 3).text == "다른 규정 16조 본문"      # 헤딩 없는 조각은 불변


def test_식별자_없는_질문은_아무것도_안_바꾼다():
    chunks = [_ch(1, ["11장"]), _ch(2, ["11조"], _HEADS, _PACKED)]
    out, scores, _ = apply_identifier_boost(chunks, [0.9, 0.7], [0, 0], [], 0.3)
    assert [c.chunk_id for c in out] == [1, 2] and scores == [0.9, 0.7]
    assert all(c.matched_ident is None for c in out)
    assert prepare_evidence(out, 5)[1].text == _PACKED


# ----- 청킹: 묶음 헤딩 기록 + 고아 헤딩 ---------------------------------------------

def _sec(path, body):
    return chunking._Section(path, None, body)


def test_pack_sections_는_내려보낸_헤딩을_기록한다():
    secs = [_sec(["문서", "제 2 장", "제 9 조(고용)"], "본문9"), _sec(["문서", "제 2 장", "제 10 조(발령)"], "본문10")]
    packed = chunking._pack_sections(secs)
    assert len(packed) == 1
    assert packed[0].headings == ["제 9 조(고용)", "제 10 조(발령)"]
    assert packed[0].body == "제 9 조(고용)\n본문9\n제 10 조(발령)\n본문10"
    assert chunking._pack_sections([secs[0]])[0].headings == []     # 안 묶이면 비어 있다


def test_split_section_은_기록된_헤딩을_조각_끝에_두지_않는다():
    class _Splitter:
        def split_text(self, body):          # 경계가 헤딩 바로 뒤에 떨어지는 분할기 흉내
            return ["본문9 첫 줄\n제 10 조(발령)", "본문10 긴 줄"]
    sec = chunking._Section(["문서", "제 2 장"], None, "본문9 첫 줄\n제 10 조(발령)\n본문10 긴 줄",
                            headings=["제 10 조(발령)"])
    assert chunking._split_section(sec, _Splitter()) == ["본문9 첫 줄", "제 10 조(발령)\n본문10 긴 줄"]
    # 다음 조각이(겹침으로) 이미 그 줄을 갖고 있으면 앞에서만 뗀다
    class _Overlap(_Splitter):
        def split_text(self, body):
            return ["본문9 첫 줄\n제 10 조(발령)", "제 10 조(발령)\n본문10 긴 줄"]
    assert chunking._split_section(sec, _Overlap()) == ["본문9 첫 줄", "제 10 조(발령)\n본문10 긴 줄"]
    # 기록된 헤딩이 없으면 분할기 출력 그대로
    plain = chunking._Section(["문서"], None, "a\nb")
    assert chunking._split_section(plain, _Splitter()) == ["본문9 첫 줄\n제 10 조(발령)", "본문10 긴 줄"]


def test_chunk_file_은_조각마다_실린_헤딩을_meta에_싣는다(tmp_path):
    # md 경로 — 같은 상위(##) 아래 ### 형제가 묶인다. 묶음 조각의 meta['headings']에 ### 제목이 그대로 남는다.
    p = tmp_path / "r.md"
    p.write_text("# 가나 규칙\n\n## 제 2 장 인사\n\n### 제 9 조(고용관계)\n고용은 계약에 의한다.\n\n### 제 10 조(인사발령)\n발령은 공고로 한다.\n",
                 encoding="utf-8")
    chunks = chunking.chunk_file(p)
    packed = [c for c in chunks if c.meta and c.meta.get("headings")]
    assert packed and packed[0].meta["headings"] == ["제 9 조(고용관계)", "제 10 조(인사발령)"]
    assert chunk_identifiers(packed[0].heading_path, packed[0].meta["headings"]) == ["2장", "9조", "10조"]
