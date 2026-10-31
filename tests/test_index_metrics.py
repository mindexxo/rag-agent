"""인제스션 소요 지표 (#151).

워커에는 "문서 한 건이 얼마나 걸리나"를 볼 다른 방법이 없다 — 로그에 남지 않고, cron drain의
소요는 배치 전체라 건당으로 나눌 수 없다. 그래서 이 지표가 유일한 창이고, 라벨이 어긋나면
(예: 확장자가 그대로 라벨이 되면) 카디널리티가 터지거나 조회가 조용히 빈다.
"""
import pytest

from rag.metrics import INDEX_DURATION_SECONDS, INDEX_TOTAL, ext_label


def _counts(metric):
    return {(s.labels.get('ext'), s.labels.get('result')): s.value
            for m in metric.collect() for s in m.samples if s.name.endswith('_total')}


class TestExtLabel:
    def test_아는_확장자는_소문자로(self):
        assert ext_label('a.PDF') == 'pdf'
        assert ext_label('보고서.docx') == 'docx'
        assert ext_label('표.xlsx') == 'xlsx'

    def test_모르는_확장자와_확장자_없음은_other로_접는다(self):
        # 카디널리티 고정이 목적이다 — 임의 확장자가 그대로 라벨이 되면 시계열이 무한히 늘어난다.
        assert ext_label('문서.hwp') == 'other'
        assert ext_label('scan.tiff') == 'other'
        assert ext_label('README') == 'other'

    def test_점이_여러_개여도_마지막만_본다(self):
        assert ext_label('a.rev2.pdf') == 'pdf'


class TestHistogram:
    def test_네_단계를_같은_라벨셋으로_기록한다(self):
        # stage가 하나라도 빠지면 대시보드의 단계 비교가 조용히 빈다.
        for stage in ('parse', 'embed', 'index', 'total'):
            INDEX_DURATION_SECONDS.labels(stage=stage, ext='pdf').observe(0.5)

        samples = {(s.labels.get('stage'), s.labels.get('ext')): s.value
                   for m in INDEX_DURATION_SECONDS.collect() for s in m.samples
                   if s.name.endswith('_count')}
        for stage in ('parse', 'embed', 'index', 'total'):
            assert samples.get((stage, 'pdf'), 0) >= 1

    def test_버킷_상한이_docling_타임아웃과_같은_눈금이다(self):
        # 변환 마감(docling_document_timeout_seconds) 위는 실패로 끝나 기록되지 않는다 — 눈금 상한이 마감과 같아야
        # "마감 직전 구간"을 읽는다. #223 전에는 '... or 300.0'이 붙어 있어 마감이 900으로 바뀐 뒤에도 통과했다
        # (300 위가 한 칸에 뭉쳤다). 이제 마감 값 자체가 유한 버킷의 최댓값이어야 한다.
        from config import settings
        finite = [float(s.labels['le']) for m in INDEX_DURATION_SECONDS.collect()
                  for s in m.samples if s.name.endswith('_bucket') and s.labels['le'] != '+Inf']
        assert max(finite) == settings.docling_document_timeout_seconds
        assert 600.0 in finite          # 정상 최악(약 810초)까지 300~900 사이에서 갈리게


class TestResultCounter:
    """실패율의 분모가 되는 카운터다 — 셈 단위가 문서여야 한다."""

    def test_세_결과를_구분해_센다(self):
        before = _counts(INDEX_TOTAL)
        for result in ('ok', 'retry', 'failed'):
            INDEX_TOTAL.labels(ext='pdf', result=result).inc()
        after = _counts(INDEX_TOTAL)
        for result in ('ok', 'retry', 'failed'):
            assert after[('pdf', result)] - before.get(('pdf', result), 0) == 1

    def test_최종_실패율은_retry를_분모에_넣지_않는다(self):
        # 한 문서가 5번 재시도되면 재시도 단위 카운터(kms_search_index_sync_total)는 error 5를
        # 세지만, 문서는 1건 실패다. ok+failed만 분모로 쓴다는 계약을 여기서 고정한다.
        before = _counts(INDEX_TOTAL)  # 같은 프로세스의 앞선 통합 테스트가 올린 값은 뺀다
        INDEX_TOTAL.labels(ext='docx', result='ok').inc(8)
        INDEX_TOTAL.labels(ext='docx', result='retry').inc(12)
        INDEX_TOTAL.labels(ext='docx', result='failed').inc(2)
        after = _counts(INDEX_TOTAL)
        ok, retry, failed = (after[('docx', r)] - before.get(('docx', r), 0) for r in ('ok', 'retry', 'failed'))
        assert failed / (ok + failed) == 0.2      # retry 12건이 실패율을 흔들지 않는다
        assert retry == 12
