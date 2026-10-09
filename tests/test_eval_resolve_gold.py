"""resolve_gold의 매칭 규칙(match_expected_chunks) — 색인 없이 튜플만으로 (#209)."""
from eval.retrieval import match_expected_chunks


def _rows(*texts, hp=None):
    return [(i + 1, hp or [], t) for i, t in enumerate(texts)]


class TestMatchExpectedChunks:
    def test_중복_문장_청크는_전부_채점_집합에_들고_주_정답은_첫_매치(self):
        rows = _rows('5쪽 ( 승인자 : PM > 팀장 )', '6쪽 ( 승인자 : PM > 팀장 )', '무관')
        primary, scoring, missing = match_expected_chunks(rows, [{'filename': 'x', 'snippet': '승인자 : PM > 팀장'}])
        assert primary == [1] and scoring == {1, 2} and missing == []

    def test_heading_path_일치가_주_정답_우선순위(self):
        rows = [(1, ['A'], '본문 같은 문장'), (2, ['B'], '본문 같은 문장')]
        primary, scoring, _ = match_expected_chunks(rows, [{'filename': 'x', 'snippet': '같은 문장', 'heading_path': ['B']}])
        assert primary == [2] and scoring == {1, 2}

    def test_공백과_파이프를_무시하고_매칭(self):
        rows = _rows('| 환산 | 타임레포트 최종 승인된 실적 기반 |')
        primary, scoring, _ = match_expected_chunks(rows, [{'filename': 'x', 'snippet': '환산 타임레포트  최종\n승인된 실적 기반'}])
        assert primary == [1] and scoring == {1}

    def test_못_찾은_스니펫은_missing으로_나가고_나머지는_계속(self):
        rows = _rows('있는 문장')
        primary, scoring, missing = match_expected_chunks(rows, [{'filename': 'x', 'snippet': '없는 문장'}, {'filename': 'x', 'snippet': '있는 문장'}])
        assert primary == [1] and scoring == {1} and missing == ['없는 문장']

    def test_복수_스니펫은_주_정답_순서를_지키고_채점_집합은_합집합(self):
        rows = [(1, [], '첫 문장 A'), (2, [], '둘째 문장 B'), (3, [], '첫 문장 A 반복')]
        primary, scoring, _ = match_expected_chunks(rows, [{'filename': 'x', 'snippet': '둘째 문장 B'}, {'filename': 'x', 'snippet': '첫 문장 A'}])
        assert primary == [2, 1] and scoring == {1, 2, 3}

    def test_heading_path가_안_맞으면_첫_매치로_폴백(self):
        rows = [(1, ['A'], '같은 문장'), (2, ['B'], '같은 문장')]
        primary, _, _ = match_expected_chunks(rows, [{'filename': 'x', 'snippet': '같은 문장', 'heading_path': ['없음']}])
        assert primary == [1]

    def test_expected_chunks가_비면_전부_빈값(self):
        assert match_expected_chunks([(1, [], '본문')], []) == ([], set(), [])
