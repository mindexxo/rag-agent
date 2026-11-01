"""docling 표 grid → 마크다운 그리기 규칙(docling_grid_markdown) — 가짜 grid를 먹인다(칸 글자 정리가 docling_core를 지연 import하므로 docling은 필요하다)."""

from types import SimpleNamespace

import pytest

from rag.chunking import docling_grid_markdown

pytest.importorskip('docling')


def _cell(text, r, c):
    return SimpleNamespace(text=text, start_row_offset_idx=r, start_col_offset_idx=c)


def _table(grid):
    return SimpleNamespace(data=SimpleNamespace(grid=grid))


class TestGridMarkdown:
    def test_가로_병합은_칸마다_반복한다(self):
        # 가로 병합은 전부 반복한다 — 첫 열에만 두면 걸친 칸 값이 다른 칸에서 빈칸으로 읽힌다
        span = _cell('1 년', 1, 1)
        md = docling_grid_markdown(
            _table(
                [
                    [_cell('구분', 0, 0), _cell('7년 이상', 0, 1), _cell('7년 미만', 0, 2)],
                    [_cell('육아휴직', 1, 0), span, span],
                ]
            )
        )
        assert md.splitlines()[2] == '| 육아휴직 | 1 년 | 1 년 |'

    def test_세로_병합은_행마다_반복한다(self):
        # 세로 병합도 반복한다 — 첫 행에만 두면 표가 청크 경계에서 잘릴 때 아래 행이 분류어를 잃는다
        cat = _cell('발생원인', 1, 0)
        md = docling_grid_markdown(
            _table(
                [
                    [_cell('구분', 0, 0), _cell('유형', 0, 1)],
                    [cat, _cell('기반구조 장애', 1, 1)],
                    [cat, _cell('기술적 장애', 2, 1)],
                ]
            )
        )
        assert md.splitlines()[2:] == ['| 발생원인 | 기반구조 장애 |', '| 발생원인 | 기술적 장애 |']

    def test_값이_하나도_없는_행은_빠진다(self):
        md = docling_grid_markdown(_table([[_cell('h', 0, 0)], [_cell('', 1, 0)], [_cell('v', 2, 0)]]))
        assert md.splitlines() == ['| h |', '| --- |', '| v |']

    def test_빈_표는_None(self):
        assert docling_grid_markdown(_table([])) is None
        assert docling_grid_markdown(_table([[_cell('', 0, 0)]])) is None
