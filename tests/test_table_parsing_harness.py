"""표 파싱 검증 하네스의 부속 — 문서 빌드·갤러리·질문·E2E 집계·LLM 재판정·실패 진단·문맥 문서. 서버 없이 돈다.

채점 규칙(기대 렌더·markdown 표 파싱·채점)은 tests/test_table_parsing_eval.py가 본다.
"""

from eval.table_parsing.expected import expected_tables
from eval.table_parsing.specs import BY_ID, SPECS


class TestBuild:
    def test_정의_하나가_4경로_문서로_나오고_병합이_파일에_실린다(self, tmp_path):
        import openpyxl
        from docx import Document

        from eval.table_parsing.build import build_all

        made = build_all(tmp_path, {'S02_colspan'})
        assert sorted(p for _, p, _ in made) == ['docx', 'img', 'pdf', 'xlsx']
        assert all(f.exists() and f.stat().st_size > 0 for _, _, f in made)
        ws = openpyxl.load_workbook(tmp_path / 'S02_colspan.xlsx.xlsx').active
        assert {str(m) for m in ws.merged_cells.ranges} == {'A1:A2', 'B1:C1', 'D1:E1'}
        t = Document(tmp_path / 'S02_colspan.docx.docx').tables[0]
        assert [c.text for c in t.rows[0].cells] == ['직급', '국내 출장', '국내 출장', '해외 출장', '해외 출장']
        assert (tmp_path / 'png' / 'S02_colspan.png').exists()

    def test_paths_인자로_경로를_좁힌다(self, tmp_path):
        from eval.table_parsing.build import build_all

        assert [p for _, p, _ in build_all(tmp_path, {'S01_basic'}, {'xlsx'})] == ['xlsx']


class TestGallery:
    def test_샘플_갤러리는_축마다_카드_하나와_이미지를_만든다(self, tmp_path):
        from eval.table_parsing.gallery import build_gallery

        page = build_gallery(tmp_path)
        html = page.read_text()
        assert html.count('class="c"') == 23 and len(list((tmp_path / 'png').glob('*.png'))) == 23


class TestQuestions:
    def test_정의마다_사실_2개_경로마다_다른_질문이고_정답이_표의_칸_값이다(self):
        from eval.table_parsing.expected import norm
        from eval.table_parsing.questions import ALL_QUESTIONS, FACTS
        from eval.table_parsing.render_image import cell_text

        for s in SPECS:
            assert len(FACTS[s['id']]) == 2, s['id']
            cells = {norm(cell_text(c)) for t in expected_tables(s) for row in t for c in row if c is not None}
            texts = set()
            for qs in ALL_QUESTIONS[s['id']].values():
                for q in qs:
                    assert any(norm(q.expected) in c for c in cells), (s['id'], q.expected)  # 정답은 표 안 글자
                    texts.add(q.text)
            n_paths = len(ALL_QUESTIONS[s['id']])
            assert len(texts) == 2 * n_paths, (s['id'], '경로마다 질문이 달라야 한다')

    def test_판정은_띄어쓰기_쉼표_원_날짜표기를_무시한다(self):
        from eval.table_parsing.questions import judge

        assert judge('임원의 국내 출장 일비는 50,000원입니다.', '50,000')
        assert judge('지원기준은 재직 1년이상입니다', '재직 1년 이상')  # OCR 띄어쓰기 소실
        assert judge('₩3,250,000입니다', '₩3,250,000') and judge('3250000원', '₩3,250,000')
        assert judge('복귀일은 2026년 11월 5일입니다', '2026-11-05')
        assert not judge('해당 내용은 문서에서 확인할 수 없습니다.', '500,000원')


class TestE2EPure:
    def test_SSE_파서는_delta와_done을_뽑고_깨진_줄은_건너뛴다(self):
        from eval.table_parsing.e2e import _sse

        text = (
            'event: meta\ndata: {"conversation_id": 1}\n\n'
            'event: delta\ndata: {"text": "임원"}\n\n'
            ': ping\n\n'
            'event: delta\ndata: {"text": "은 50,000원"}\n\n'
            'event: done\ndata: {"finish_reason": "done", "citations": [{"filename": "a.pdf"}]}\n'
        )
        ev = _sse(text)
        assert ''.join(d['text'] for e, d in ev if e == 'delta') == '임원은 50,000원'
        assert next(d for e, d in ev if e == 'done')['citations'][0]['filename'] == 'a.pdf'

    def test_리뷰_요약은_축_경로별_정답_인용_수를_센다(self, tmp_path):
        from eval.table_parsing.e2e import phase_report

        manifest = [
            dict(
                id='S02_colspan',
                path='pdf',
                filename='S02_colspan.pdf.pdf',
                document_id=1,
                status='ready',
                chunks=[{'text': '| a |', 'heading_path': []}],
            ),
            dict(id='S02_colspan', path='docx', filename='x', document_id=2, status='failed', chunks=[]),
        ]
        answers = [
            dict(
                id='S02_colspan',
                path='pdf',
                document_id=1,
                question='q1',
                expected='50,000',
                manual=False,
                answer='50,000원',
                cited=['S02_colspan.pdf.pdf'],
                cited_ok=True,
                value_ok=True,
            ),
            dict(
                id='S02_colspan',
                path='pdf',
                document_id=1,
                question='q2',
                expected='40,000',
                manual=False,
                answer='모름',
                cited=[],
                cited_ok=False,
                value_ok=False,
            ),
        ]
        (tmp_path / 'docs' / 'png').mkdir(parents=True)
        phase_report(tmp_path, manifest, answers)
        summary = (tmp_path / 'summary.md').read_text()
        assert '색인 1/1 · 생성 1/2(기계 1) · 인용 1/2' in summary and '색인 0/1 · 생성 0/0' in summary
        assert (tmp_path / 'review.html').read_text().count('class="c"') == 2


class TestLlmJudge:
    def test_재판정은_기계_실패분만_묻고_결과를_행에_쓴다(self):
        import asyncio

        from eval.table_parsing import llm_judge
        from eval.table_parsing.llm_judge import AnswerJudgment, final_ok, rejudge

        class FakeLlm:
            calls = []

            async def acomplete(self, messages, **_):
                FakeLlm.calls.append(messages[1]['content'])
                return AnswerJudgment(
                    label='correct' if '30일' in messages[1]['content'] else 'ungrounded', reason='이유'
                ).model_dump_json()

        rows = [
            dict(
                id='S03_rowspan',
                path='pdf',
                question='무급 병가 한도?',
                expected='연 30일',
                answer='1년에 30일까지',
                value_ok=False,
                manual=False,
            ),
            dict(
                id='S20_symbols',
                path='img',
                question='백업 완료?',
                expected='☑',
                answer='완료 상태입니다',
                value_ok=False,
                manual=True,
            ),
            dict(
                id='S01_basic',
                path='pdf',
                question='x',
                expected='500,000원',
                answer='500,000원',
                value_ok=True,
                manual=False,
            ),
        ]
        chunks = {
            ('S03_rowspan', 'pdf'): [{'text': '| 병가 | 무급 | 연 30일 |'}],
            ('S20_symbols', 'img'): [{'text': '| 백업확인 |  | 운영팀 |'}],
        }
        done = asyncio.run(rejudge(rows, chunks, llm=FakeLlm()))
        assert len(done) == 2 and len(FakeLlm.calls) == 2  # 기계가 맞다고 한 행은 안 묻는다
        assert '| 병가 | 무급 | 연 30일 |' in FakeLlm.calls[0]  # 청크를 같이 준다
        assert rows[0]['llm_label'] == 'correct' and final_ok(rows[0])
        assert rows[1]['llm_label'] == 'ungrounded' and not final_ok(rows[1])  # 근거 없는 정답은 최종 성공이 아니다
        assert final_ok(rows[2]) and 'llm_label' not in rows[2]
        assert rows[0]['judge'] == llm_judge.judge_label()

    def test_판정_실패는_기계_판정을_유지하고_오류를_남긴다(self):
        import asyncio

        from eval.table_parsing.llm_judge import final_ok, rejudge

        class BrokenLlm:
            async def acomplete(self, messages, **_):
                raise RuntimeError('한도')

        rows = [dict(id='a', path='pdf', question='q', expected='e', answer='x', value_ok=False, manual=False)]
        asyncio.run(rejudge(rows, {}, llm=BrokenLlm()))
        assert rows[0]['llm_label'] is None and '한도' in rows[0]['llm_error'] and not final_ok(rows[0])


class TestDiagnosis:
    def test_청킹_손실은_청크에_없는_칸_글자만(self):
        from eval.table_parsing.e2e import chunk_loss

        spec = BY_ID['V14_vertical']
        chunks = [
            {
                'text': '| 0 | 1 | 2 |\n| --- | --- | --- |\n|  |  |  |\n| 교통비 | 150,000 | 매월25일 |\n| 통신비 | 50,000 | 매월 25일 |\n| 식대 | 200,000 | 매월 25일 |'
            }
        ]
        assert chunk_loss(spec, 'img', chunks) == [
            '정액 항목',
            '월 지급액',
            '지급일',
        ]  # 세로쓰기 머리글만 없음(띄어쓰기는 무시)

    def test_실패_원인은_인용·청크·답변_모양으로_가른다(self):
        from eval.table_parsing.e2e import diagnose

        ch = [{'text': '| 식대 | 200,000 | 매월 25일 |'}]
        base = dict(expected='매월 25일', cited=[], cited_ok=False, final_ok=False, llm_label=None)
        assert '후보 0건' in diagnose(dict(base, answer='해당 내용은 제공된 문서에서 확인할 수 없습니다.'), ch)
        assert '파싱 단계 손실' in diagnose(
            dict(base, answer='확인할 수 없습니다.'), [{'text': '| 식대 | 200,000 |  |'}]
        )
        assert '추정' in diagnose(dict(base, answer='완료', llm_label='ungrounded'), ch)
        assert '파싱 손실' in diagnose(
            dict(base, answer='매월 30일', llm_label='wrong'), [{'text': '| 식대 | 200,000 | 매월 30일 |'}]
        )
        assert '생성 모델 오류' in diagnose(
            dict(base, answer='매월 30일', cited=['x'], cited_ok=True, llm_label='wrong'), ch
        )
        assert diagnose(dict(base, final_ok=True, answer='매월 25일'), ch) == ''


class TestContext:
    """--context 문서 — 문맥이 표 값을 흘리면 '문맥 유무'만 비교한다는 전제가 깨진다."""

    def test_모든_정의에_문맥이_있다(self):
        from eval.table_parsing.context import CONTEXT
        from eval.table_parsing.specs import SPECS

        assert {s['id'] for s in SPECS} == set(CONTEXT)

    def test_문맥에는_표의_값_칸이_없다(self):
        import re

        from eval.table_parsing.context import blocks
        from eval.table_parsing.expected import expected_cells, flatten_nested
        from eval.table_parsing.specs import SPECS, tables_of

        for s in SPECS:
            b = blocks(s['id'])
            # 시행일 줄('2026년 10월 1일')은 모든 문서 공통이라 '월 1일' 같은 짧은 칸에 부분 일치한다(S03) — 표 사실과 무관해 뺀다
            text = ' '.join(v for k, v in b.items() if k != 'meta')
            for rows in tables_of(s):
                if any(isinstance(c, dict) and c.get('nested') for r in rows for c in r):
                    rows = flatten_nested(rows)
                for cell in expected_cells(rows):
                    t = cell.text.strip()
                    if re.search(r'\d', t) and len(t) >= 3:  # 숫자가 든 칸(금액·날짜·수량)이 문맥에 있으면 안 된다
                        assert t not in text, (s['id'], t)
