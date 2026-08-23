"""부재단정 판정기의 **결정적인 부분만** 단위로 고정 (#76).

judge 자체의 정확도는 비결정적이라 여기서 안 본다 — `python -m eval.absence_judge`의
FIXTURES 대조(RUNS=3)가 그 몫이다. 여기서 지키는 것은 세 계약이다:
프롬프트 조립, 스키마 형상, 그리고 **인용이 있으면 judge를 부르지 않는다**는 2단 게이트.
게이트가 깨지면 judge 호출량이 배로 늘고도 아무 신호가 없다.
"""
import pytest
from pydantic import ValidationError

from eval.absence_judge import (FIXTURES, JUDGE_PROMPT_VERSION, AbsenceJudgment,
                                _build_messages)
from rag.prompt_texts import NO_EVIDENCE_ANSWER


class TestPromptAssembly:
    def test_정상형_문구가_상수에서_온다(self):
        """손으로 베끼면 문구가 갈린다 — #62에서 실제로 겪은 실패다."""
        system = _build_messages('q', 'a')[0]['content']
        assert NO_EVIDENCE_ANSWER in system

    def test_질문과_답변이_유저_메시지에_실린다(self):
        msgs = _build_messages('환불 되나요?', '제공되지 않습니다.')
        assert [m['role'] for m in msgs] == ['system', 'user']
        assert '환불 되나요?' in msgs[1]['content']
        assert '제공되지 않습니다.' in msgs[1]['content']


class TestSchema:
    def test_label은_두_갈래뿐(self):
        assert AbsenceJudgment(label='absence_assertion').label == 'absence_assertion'
        assert AbsenceJudgment(label='refusal_ok').reason is None
        with pytest.raises(ValidationError):
            AbsenceJudgment(label='maybe')

    def test_label에_기본값이_없다(self):
        """기본값이 있으면 json 스키마에서 required가 빠져 모델이 생략해도 합법이 된다 —
        강제력이 조용히 약해진다(rag/llm_schemas.py 규약)."""
        assert 'label' in AbsenceJudgment.model_json_schema()['required']
        with pytest.raises(ValidationError):
            AbsenceJudgment()


class TestFixtures:
    def test_결함_6종이_전부_들어있다(self):
        """정규식이 실제로 틀린 케이스가 회귀 자산이다 — 줄이면 그만큼 감시가 준다."""
        assert len(FIXTURES) >= 7
        assert {f['expected'] for f in FIXTURES} == {'absence_assertion', 'refusal_ok'}
        for f in FIXTURES:
            assert f['defect'], f"{f['id']}: 왜 이 케이스가 있는지 적혀 있어야 한다"

    def test_정상형_픽스처는_상수를_쓴다(self):
        canonical = next(f for f in FIXTURES if f['id'] == 'canonical')
        assert NO_EVIDENCE_ANSWER in canonical['answer']

    def test_근거있는_부정은_픽스처에_없다(self):
        """사유는 eval/absence_judge.py 모듈 docstring(판정 범위)."""
        assert not any('반품 불가' in f['answer'] for f in FIXTURES)


class TestVersioning:
    def test_판정_버전이_박혀_있다(self):
        """판정 기준이 바뀌면 올려야 한다 — 시계열에서 '기준 변경'과 '실제 회귀'를 가르는 유일한 단서."""
        assert JUDGE_PROMPT_VERSION.startswith('v')


class TestTwoStageGate:
    @pytest.mark.asyncio
    async def test_인용이_있으면_judge를_부르지_않는다(self, monkeypatch, tmp_path):
        """2단 게이트의 핵심 계약. 깨지면 호출량이 배로 늘고도 조용하다."""
        import eval.refusal as R

        calls = []

        async def _fake_judge(llm, query, answer):
            calls.append(query)
            return AbsenceJudgment(label='refusal_ok')

        monkeypatch.setattr(R, 'judge_absence', _fake_judge)
        monkeypatch.setattr(R, 'judge_llm', lambda: object())

        async def _fake_refused(tenant, query):
            # route까지 3-tuple — OTHER는 판정 대상이 아니다(게이트 2조건 중 하나)
            return (query == 'refused'), 'answer body', 'knowledge'

        monkeypatch.setattr(R, '_refused', _fake_refused)
        monkeypatch.setattr(R, 'row_tenant', lambda g: 't')
        monkeypatch.setattr(R, 'save_audit', lambda rows, stamp=None: 'audit.jsonl')
        gold = tmp_path / 'gold.jsonl'      # PosixPath.read_text는 패치 불가 — 실파일로 대체
        gold.write_text('{"id":"a","type":"no_evidence","query":"refused"}\n'
                        '{"id":"b","type":"trap","query":"answered"}\n')
        monkeypatch.setattr(R, 'GOLD', gold)

        r = await R.compute()
        assert calls == ['refused']          # 인용 0건인 한 건만 판정했다
        assert r['absence_judged_n'] == 1

    @pytest.mark.asyncio
    async def test_OTHER_라우팅은_판정하지_않는다(self, monkeypatch, tmp_path):
        """OTHER는 규칙 3을 안 타고 sources가 항상 빈 목록이라 무조건 근거없음으로 집계된다.
        판정 대상에 넣으면 분모가 라우팅 확률에 좌우되고, 다른 프롬프트의 화법을
        규칙 3 기준으로 재는 오염이 된다(#76 리뷰에서 실측으로 잡힌 문제)."""
        import eval.refusal as R

        calls = []

        async def _fake_judge(llm, query, answer):
            calls.append(query)
            return AbsenceJudgment(label='absence_assertion')

        monkeypatch.setattr(R, 'judge_absence', _fake_judge)
        monkeypatch.setattr(R, 'judge_llm', lambda: object())
        monkeypatch.setattr(R, 'row_tenant', lambda g: 't')
        monkeypatch.setattr(R, 'save_audit', lambda rows, stamp=None: 'audit.jsonl')

        async def _fake_refused(tenant, query):
            return True, 'body', ('other' if query == 'other' else 'knowledge')

        monkeypatch.setattr(R, '_refused', _fake_refused)
        gold = tmp_path / 'gold.jsonl'
        gold.write_text('{"id":"a","type":"no_evidence","query":"other"}\n'
                        '{"id":"b","type":"no_evidence","query":"knowledge"}\n')
        monkeypatch.setattr(R, 'GOLD', gold)

        r = await R.compute()
        assert 'other' not in calls          # OTHER 행은 judge에 안 갔다
        assert r['absence_judged_n'] == 1
        # knowledge 행은 2회 불린다 — 부재단정으로 판정된 행은 flaky 확인차 1회 재판정한다.
        assert calls == ['knowledge', 'knowledge']
        assert r['absence_flaky'] == 0       # 재판정에서도 같은 판정 → 흔들림 아님
