"""#232 표 문항 축 — 표 조각 판정(eval.axis_common.is_table_chunk·table_kind)과 전수 조사 집계(eval.table_census.census).

`is_table`만 보면 산문 조각 안의 작은 표를 놓친다(실문서 PDF 작은 표 38 vs 큰 표 3) — 그 판정을 여기서 고정한다.
"""

from types import SimpleNamespace as NS

from eval.axis_common import is_table_chunk, table_kind
from eval.table_census import census, summary

MD = "제3조(휴가)\n| 근속기간 | 휴가일수 |\n| --- | --- |\n| 5년 | 3일 |\n| 10년 | 5일 |\n"


def test_표_조각_판정은_큰_표_표식과_마크다운_구분선_둘_다_본다():
    assert is_table_chunk(NS(is_table=True, text="| a | b |"))  # 큰 표 묶음(엑셀·PDF·DOCX 2묶음 이상)
    assert is_table_chunk(NS(is_table=False, text=MD))  # 산문 안 작은 표
    assert not is_table_chunk(NS(is_table=False, text="가 | 나 | 다"))  # 파이프만 있는 산문은 표가 아니다
    assert not is_table_chunk(NS(is_table=False, text=None))


def test_정답_조각_묶음의_표_종류():
    big, small, prose = (
        NS(is_table=True, text="| a |\n| --- |\n| 1 |"),
        NS(is_table=False, text=MD),
        NS(is_table=False, text="산문"),
    )
    assert table_kind([prose]) == ""
    assert table_kind([big, prose]) == "big"
    assert table_kind([small]) == "small"
    assert table_kind([big, small]) == "big+small"


def test_전수_조사는_큰_표를_table_key로_작은_표를_구분선으로_센다():
    chunks = [
        {"filename": "a.pdf", "text": MD, "is_table": False},  # 작은 표 1, 행 2
        {"filename": "a.pdf", "text": "산문", "is_table": False},
        {
            "filename": "b.xlsx",
            "text": "| h |\n| --- |\n| 1 |\n| 2 |",
            "is_table": True,
            "table_key": "시트1",
        },  # 큰 표 묶음 1
        {
            "filename": "b.xlsx",
            "text": "| h |\n| --- |\n| 3 |",
            "is_table": True,
            "table_key": "시트1",
        },  # 같은 표 묶음 2
    ]
    rows = {r["file"]: r for r in census("t", chunks)}
    assert rows["a.pdf"] == {
        "tenant": "t",
        "file": "a.pdf",
        "ext": "pdf",
        "chunks": 2,
        "table_chunks": 1,
        "big": 0,
        "small": 1,
        "rows": 2,
    }
    assert rows["b.xlsx"]["big"] == 1 and rows["b.xlsx"]["rows"] == 3 and rows["b.xlsx"]["table_chunks"] == 2
    md = summary(list(rows.values()))
    assert "| 모의 | pdf | 1 | 1 | 0 | 1 | 1 | 2 | 2 |" in md and "| 모의 | xlsx | 1 | 1 | 1 | 0 | 2 | 2 | 3 |" in md
