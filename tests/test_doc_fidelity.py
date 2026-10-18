"""실문서 파싱 충실도 도구(#247·#231) — 판정 기준·분류·표본·집계의 순수 로직. DB·VLM·Claude 없이 돈다."""
import pytest

from eval.doc_fidelity import coverage as cov
from eval.doc_fidelity.judge import parse
from eval.doc_fidelity.pictures import norm
from eval.doc_fidelity.review import select_for_human, summarize
from eval.doc_fidelity.rubric import PROMPT, Verdict, grade_of
from eval.doc_fidelity.synthetic import score


def _v(**k):
    base = dict(type='흐름도', text_units=20, missing=0, misread=0, number_errors=0, relation_errors=0,
                hallucinations=0, chatter=0, grade='정확')
    return Verdict(**{**base, **k})


class TestRubric:
    def test_등급은_결정6_규칙대로_개수에서_계산된다(self):
        assert grade_of(_v()) == '정확'
        assert grade_of(_v(chatter=2)) == '정확'                       # 잡담은 허용
        assert grade_of(_v(missing=2)) == '경미'                       # 2/20 = 10% 이하
        assert grade_of(_v(missing=2, misread=1)) == '오류'            # 3/20 > 10%
        assert grade_of(_v(relation_errors=1)) == '오류'
        assert grade_of(_v(number_errors=1)) == '위험'
        assert grade_of(_v(hallucinations=1, missing=10)) == '위험'    # 위험이 오류보다 우선

    def test_사진_장식은_판정에서_빠지고_캡션_없는_글자_그림은_오류(self):
        assert grade_of(_v(type='사진·장식', text_units=0)) is None
        assert grade_of(_v(text_units=5, missing=5)) == '오류'          # 캡션 없음 = 전부 누락
        assert grade_of(_v(text_units=0)) == '정확'                     # 잃은 글자가 없다

    def test_프롬프트에_출력_키를_전부_적는다(self):
        # #253 교훈: 키를 적지 않으면 판정자가 다른 이름으로 답한다
        for key in Verdict.model_fields:
            assert key in PROMPT


class TestJudgeParse:
    def test_앞뒤_잡담이_있어도_JSON을_뽑는다(self):
        raw = '판정입니다.\n{"type":"차트","text_units":4,"missing":0,"misread":0,"number_errors":1,' \
              '"relation_errors":null,"hallucinations":0,"chatter":0,"grade":"위험","ambiguous":false,"evidence":["숫자"]}\n끝'
        v = parse(raw)
        assert v.type == '차트' and v.relation_errors is None and grade_of(v) == '위험'

    def test_없는_유형이나_키_누락은_실패로_올린다(self):
        with pytest.raises(Exception):
            parse('{"type":"지도","text_units":1}')


class TestCoverage:
    def test_정규화는_공백과_표_구분자를_지운다(self):
        assert norm('| 육아 휴직 |\n 1 년 |') == '육아휴직1년'

    def test_문장은_줄과_종결어미에서_나뉜다(self):
        assert cov._sentences('첫 문장이다. 둘째 문장임\n셋째') == ['첫 문장이다.', '둘째 문장임', '셋째']

    def test_본문_표_헤딩_부분_유실후보로_분류한다(self):
        chunks = [{'text': '휴가는 연 15일이다.\n| 구분 | 일수 |\n| 연차 | 15일 |', 'heading_path': ['문서', '제 3 조(휴가)']},
                  {'text': '병가는 연 60일까지 쓸 수 있다', 'heading_path': ['문서']}]
        units = [cov.Unit(1, 'pdf', '본문', 1, t) for t in
                 ('휴가는 연 15일이다.', '연차 15일', '제 3 조(휴가)', '병가는 연60일까지 쓸수 있 다', '병가는 연 60일까지 쓸 수 있다만', '전혀 없는 문장이다', '12')]
        got = {u['text']: u['class'] for u in cov.classify(units, chunks)}
        assert got['휴가는 연 15일이다.'] == '본문' and got['연차 15일'] == '표' and got['제 3 조(휴가)'] == '헤딩'
        assert got['병가는 연60일까지 쓸수 있 다'] == '본문'            # 띄어쓰기 차이는 무시
        assert got['병가는 연 60일까지 쓸 수 있다만'] == '부분'   # 바이그램 80% 이상 겹침
        assert got['전혀 없는 문장이다'] == '유실 후보'
        assert '12' not in got                                         # MIN_CHARS 미만은 세지 않는다


class TestReview:
    def _recs(self):
        recs = []
        for i in range(10):
            recs.append({'picture_id': f'1-{i:03d}', 'in_index': True, 'captioned': True,
                         'verdict': {'type': '흐름도', 'grade': '정확'}})
        recs += [{'picture_id': '1-010', 'in_index': True, 'captioned': True, 'verdict': {'type': '차트', 'grade': '위험', 'hallucinations': 1}},
                 {'picture_id': '1-011', 'in_index': False, 'captioned': True, 'verdict': {'type': '차트', 'grade': '정확'}},
                 {'picture_id': '1-012', 'in_index': None, 'captioned': False, 'verdict': {'type': '사진·장식', 'grade': None}}]
        return recs

    def test_적으면_전부_리뷰하고_장식도_확인용으로_넣는다(self):
        picked = select_for_human(self._recs())
        assert len(picked) == 13 and {r['review_reason'] for r in picked} == {'전수', '플래그', '장식 확인'}

    def test_플래그는_전부_정상은_20퍼센트_시드고정_장식은_제외(self, monkeypatch):
        import eval.doc_fidelity.review as rv
        monkeypatch.setattr(rv, 'ALL_BELOW', 5)
        picked = select_for_human(self._recs())
        ids = [r['picture_id'] for r in picked]
        assert {'1-010', '1-011'} <= set(ids) and '1-012' not in ids          # 위험·색인 불일치는 전부, 장식은 제외
        assert sum(r['review_reason'] == '정상 표본' for r in picked) == 2      # 10 × 20%
        assert ids == [r['picture_id'] for r in select_for_human(self._recs())]   # 같은 입력 = 같은 표본

    def test_사람_등급이_최종이고_뒤집힌_환각은_환각률에서_빠진다(self):
        s = summarize(self._recs(), {'1-010': {'grade': '정확', 'memo': '판정자 오판'}, '1-000': {'grade': '오류'}})
        assert s['excluded'] == 1 and s['judged'] == 12
        assert s['hallucination_rate'] == 0                                    # 사람이 위험 → 정확으로 뒤집음
        assert s['judge_human_agree'] == 0 and s['judge_missed'] == 1           # 판정자 '정확'을 사람이 '오류'로


def test_합성_캡션_채점은_칸값과_숫자칸을_따로_센다():
    r = score(['구분', '단가', '3,500원', '2026-11-04'], '구분 단가\n3,500 원')
    assert (r['values_hit'], r['numeric'], r['numeric_hit']) == (3, 2, 1) and r['missing'] == ['2026-11-04']


def test_색인_대조는_청크_경계에서_나뉜_긴_캡션도_일치로_본다():
    from eval.doc_fidelity.pictures import index_line_ratio
    cap = '실적 관리\n1주차 계획: 40.0 Hr\n2주차 계획: 44.0 Hr\n사후특근 등록'
    chunks = norm('실적 관리\n1주차 계획: 40.0 Hr\n2주차 계획: 44.0') + norm('2주차 계획: 44.0 Hr\n사후특근 등록')   # 겹침 경계
    assert norm(cap) not in chunks and index_line_ratio(cap, chunks) == 1.0
    assert index_line_ratio(cap + '\n지어낸 줄입니다', chunks) == 0.8


class TestSyntheticPictures:
    def test_짝은_같은_줄이나_빈_줄로_나뉜_블록_안이면_맞다(self):
        from eval.doc_fidelity.synthetic import score_picture
        spec = {'texts': ['상암', '수원'], 'numbers': ['24', '31'], 'pairs': [('상암', '24'), ('수원', '31')]}
        assert score_picture(spec, '상암\n정규직 24\n\n수원\n정규직 31')['pairs_hit'] == 2
        assert score_picture(spec, '상암\n수원\n24\n31')['pairs_hit'] == 0           # 항목과 값이 떨어지면 관계를 잃은 것

    def test_순서는_첫_등장_위치로_본다(self):
        from eval.doc_fidelity.synthetic import score_picture
        spec = {'texts': [], 'numbers': [], 'sequence': ['신고', '점검', '해결']}
        assert score_picture(spec, '신고\n점검\n해결?')['sequence_ok'] is True
        assert score_picture(spec, '점검\n신고\n해결?')['sequence_ok'] is False

    def test_짝짓기는_캡션이_아니라_그림_모양으로_한다(self, tmp_path):
        from eval.doc_fidelity.picture_specs import PICTURES
        from eval.doc_fidelity.synthetic import match_pictures
        specs = [(k, PICTURES[k]) for k in ('P06_edge05', 'P01_bar')]
        PICTURES['P01_bar']['draw']().save(tmp_path / 'a.png')
        PICTURES['P06_edge05']['draw']().resize((600, 214)).save(tmp_path / 'b.png')   # 크기가 달라도 같은 그림
        recs = [{'picture_id': 'x-000', 'crop': 'a.png', 'caption': '엉뚱한 캡션 2 1'}, {'picture_id': 'x-001', 'crop': 'b.png', 'caption': ''}]
        got = dict((sid, (r or {}).get('picture_id')) for sid, r in match_pictures(specs, recs, tmp_path))
        assert got == {'P06_edge05': 'x-001', 'P01_bar': 'x-000'}


def test_근거가_문자열_하나여도_목록으로_받는다():
    v = parse('{"type":"화면 캡처","text_units":3,"missing":1,"misread":0,"number_errors":0,"relation_errors":null,'
              '"hallucinations":0,"chatter":0,"grade":"오류","ambiguous":false,"evidence":"누락: 버튼 2개"}')
    assert v.evidence == ['누락: 버튼 2개']


def test_같은_문서에서_반복되는_단위는_한_번만_센다():
    units = [cov.Unit(1, 'pdf', '본문', p, '오후 10:24 [인사 | 표준] 급여 표준') for p in (1, 2, 3)]
    assert len(cov.classify(units, [{'text': '무관한 청크', 'heading_path': []}])) == 1


def test_청크_보기는_캡션_고유_줄을_덮는_최소_청크_또는_같은_범위_자리표시_청크():
    from eval.doc_fidelity.run import chunk_view
    common = 'Hansol Time Report\n'
    chunks = [{'text': common + '본문\n실적 관리\n1주차 계획', 'page': 5}, {'text': common + '1주차 계획\n사후특근 등록', 'page': 5},
              {'text': common + '<!-- image -->\n다른 쪽', 'page': 6}, {'text': common + '<!-- image -->', 'page': 8},
              {'text': common + '무관', 'page': 5}, {'text': common + '무관2', 'page': 5}]
    got = chunk_view({'caption': common + '실적 관리\n사후특근 등록', 'page': 5}, chunks)
    assert len(got) == 2                                   # 경계에서 나뉜 캡션 — 공통 줄로 '무관' 청크까지 끌어오지 않는다
    assert [c['page'] for c in chunk_view({'caption': '', 'page': 7}, chunks)] == [6]   # 앞 쪽에서 시작한 청크의 자리표시


def test_모양이_달라진_그림은_캡션으로_짝짓고_합쳐진_그림은_정의_둘에_걸린다(tmp_path):
    from eval.doc_fidelity.picture_specs import PICTURES
    from eval.doc_fidelity.synthetic import match_pictures
    from PIL import Image
    Image.new('RGB', (300, 300), 'gray').save(tmp_path / 'm.png')          # 어느 정의와도 모양이 다르다
    recs = [{'picture_id': 'x-000', 'crop': 'm.png', 'caption': '지급 기준\n유니폼 지급: 연 2벌\n지급 기준\n안전화 지급: 2년마다 1켤레'}]
    got = match_pictures([(k, PICTURES[k]) for k in ('P06_edge05', 'P07_edge15')], recs, tmp_path)
    assert [(sid, (r or {}).get('picture_id')) for sid, r in got] == [('P06_edge05', 'x-000'), ('P07_edge15', 'x-000')]


class TestPromptAB:
    def test_변형은_현행에서_한_줄씩만_다르다(self):
        from eval.doc_fidelity.prompt_ab import VARIANTS
        from rag.chunking import _PICTURE_CAPTION_PROMPT
        assert VARIANTS['V13_V11+계층'] == _PICTURE_CAPTION_PROMPT     # 채택안(#247 결정 15)이 운영 프롬프트와 같다
        assert VARIANTS['V0_현행'].split('\n') == [x for x in _PICTURE_CAPTION_PROMPT.split('\n')       # 옛 운영 = 채택안에서 더한 4줄을 뺀 것
                                                    if not x.startswith(('- 흐름도', '- 조직도', '- 차트', '- 값이 비어'))]
        v0 = VARIANTS['V0_현행'].split('\n')
        for name in ('V1_한국어로_제거', 'V2_위치나열_제거', 'V3_빈칸_추가'):
            v = VARIANTS[name].split('\n')
            assert len(set(v) ^ set(v0)) <= 2, name

    def test_흐름도_화살표와_빈칸_환각_채점(self):
        from eval.doc_fidelity.prompt_ab import blank_hallucinated, edge_hits
        assert edge_hits('고장 신고 → 원격 점검\n원격 해결? → 처리 종료 (예)', [('고장 신고', '원격 점검'), ('원격 해결', '처리 종료'),
                                                                     ('원격 해결', '보증 기간')]) == 2
        assert blank_hallucinated('2026년 4분기 362') and not blank_hallucinated('2026년: 4분기: 값 없음')
        assert not blank_hallucinated('2025년 4분기 362')


def test_판정_호출이_실패해도_그_그림만_실패로_남기고_계속한다(tmp_path, monkeypatch):
    import asyncio
    import eval.claude_client as cc
    from eval.doc_fidelity.judge import judge_all
    calls = []

    class _Fake:
        def __init__(self, model):
            self.model = model

        async def acomplete_image(self, prompt, png, system=None):
            calls.append(1)
            if len(calls) == 1:
                raise RuntimeError('claude -p 실패 (exit 1): 사용량 한도')
            return ('{"type":"차트","text_units":1,"missing":0,"misread":0,"number_errors":0,"relation_errors":null,'
                    '"hallucinations":0,"chatter":0,"grade":"정확","ambiguous":false,"evidence":[]}')
    monkeypatch.setattr(cc, 'ClaudeCliClient', _Fake)
    for n in ('a.png', 'b.png'):
        (tmp_path / n).write_bytes(b'x')
    recs = [{'picture_id': 'x-000', 'crop': 'a.png', 'caption': 'c'}, {'picture_id': 'x-001', 'crop': 'b.png', 'caption': 'c'}]
    rows = asyncio.run(judge_all(recs, tmp_path, concurrency=1, sink=tmp_path / 'p.jsonl'))
    assert rows[0]['verdict']['error'].startswith('호출 실패') and rows[1]['verdict']['grade'] == '정확'
    assert len((tmp_path / 'p.jsonl').read_text().splitlines()) == 2           # 그림마다 바로 이어 쓴다


def test_판정_실패_메시지도_HTML로_이스케이프한다(tmp_path):
    from eval.doc_fidelity.review import write_html
    rec = {'picture_id': '1-000', 'filename': 'a.pdf', 'fmt': 'pdf', 'page': 1, 'area_ratio': 0.1, 'in_table': False,
           'crop': 'a.png', 'caption': 'c', 'in_index': True,
           'verdict': {'grade': None, 'error': "ValueError: JSON 없음: '<script>alert(1)</script>'"}}
    html = write_html([rec], tmp_path / 'r.html').read_text()
    assert '<script>alert(1)</script>' not in html and '&lt;script&gt;alert(1)&lt;/script&gt;' in html


def test_판정_실패는_제외와_따로_센다():
    recs = [{'picture_id': '1-000', 'captioned': True, 'verdict': {'type': '차트', 'grade': '정확'}},
            {'picture_id': '1-001', 'captioned': True, 'verdict': {'grade': None, 'error': '호출 실패: 한도'}},
            {'picture_id': '1-002', 'captioned': False, 'verdict': {'type': '사진·장식', 'grade': None}},
            {'picture_id': '1-003', 'captioned': True, 'verdict': {'grade': None, 'error': '호출 실패'}}]
    s = summarize(recs, {'1-003': {'grade': '오류'}})                  # 사람이 본 실패분은 사람 등급이 최종
    assert (s['judged'], s['excluded'], s['judge_errors']) == (2, 1, 1)
