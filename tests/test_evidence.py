"""테스트 증거 스크립트(tests/evidence.py, #311) — 합성 JUnit 리포트로 집계·사유·오래된 리포트 경고를 검사한다."""

import os
import time

from tests.evidence import summarize

REPORT = """<?xml version="1.0" encoding="utf-8"?>
<testsuites><testsuite name="pytest">
 <testcase classname="suite.test_a" name="test_ok"/>
 <testcase classname="suite.test_a.TestX" name="test_ok2"/>
 <testcase classname="suite.test_b" name="test_bad"><failure message="assert 1 == 2&#10;상세">trace</failure></testcase>
 <testcase classname="suite.test_c" name="test_skip"><skipped type="pytest.skip" message="Skipped: ruff 미설치"/></testcase>
 <testcase classname="suite.test_c" name="test_skip2"><skipped type="pytest.skip" message="Skipped: ruff 미설치"/></testcase>
 <testcase classname="suite.test_d" name="test_xf"><skipped type="pytest.xfail" message="알려진 결함"/></testcase>
</testsuite></testsuites>
"""


def test_통과_실패_스킵_xfail을_나누고_스킵은_사유별로_센다(tmp_path):
    report = tmp_path / 'junit.xml'
    report.write_text(REPORT, encoding='utf-8')
    text, failed = summarize(report)
    assert failed
    assert '| **합계** | 2 | 1 | 2 | 1 |' in text
    assert '`suite/test_b.py::test_bad` — assert 1 == 2' in text
    assert '스킵: ruff 미설치 (2건)' in text and 'xfail: 알려진 결함 (1건)' in text
    assert '`suite/test_a.py`' not in text  # 다 통과한 파일은 표에서 뺀다
    assert '⚠️' not in text


def test_오래된_리포트는_경고한다(tmp_path):
    report = tmp_path / 'junit.xml'
    report.write_text(REPORT.replace('<failure message="assert 1 == 2&#10;상세">trace</failure>', ''), encoding='utf-8')
    old = time.time() - 3 * 3600
    os.utime(report, (old, old))
    text, failed = summarize(report)
    assert not failed and '⚠️' in text


def test_0건_리포트와_깨진_리포트는_실패로_끝난다(tmp_path):
    from tests.evidence import main

    empty = tmp_path / 'empty.xml'
    empty.write_text('<testsuites><testsuite name="pytest"/></testsuites>', encoding='utf-8')
    text, failed = summarize(empty)
    assert failed and '0건' in text
    broken = tmp_path / 'broken.xml'
    broken.write_text('<testsuites><testsuite', encoding='utf-8')
    assert main(['x', str(broken)]) == 1
    assert main(['x']) == 2


def test_error_요소도_실패로_센다(tmp_path):
    report = tmp_path / 'junit.xml'
    report.write_text(
        '<testsuites><testsuite><testcase classname="suite.test_e" name="t">'
        '<error message="fixture 실패">x</error></testcase></testsuite></testsuites>',
        encoding='utf-8',
    )
    text, failed = summarize(report)
    assert failed and 'fixture 실패' in text
