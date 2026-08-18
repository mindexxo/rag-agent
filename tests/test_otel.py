"""OTel 계측(#7) 단위 테스트 — endpoint 미설정(no-op) 경로가 핵심.

운영 기본값이 no-op이므로 "계측 코드가 깔려 있어도 아무 일도 안 일어난다"가
가장 중요한 계약이다. 실제 전송은 로컬 Phoenix E2E로 검증 (자동 테스트 밖).
"""
import pytest

from config import settings
from rag import otel


def test_endpoint_미설정이면_init은_no_op():
    assert settings.otel_endpoint == ''      # 기본값 확인 (바뀌면 운영 영향)
    assert otel.init_tracing() is False


def test_no_op_스팬은_기록되지_않는다():
    with otel.span('테스트', 'CHAIN') as sp:
        assert not sp.is_recording()
        # 기록 안 되는 스팬에도 속성 세팅이 예외 없이 무시돼야 한다
        otel.set_attrs(sp, {'k': 'v', 'none은_걸러짐': None})


def test_clip_상한_적용():
    assert otel.clip('짧다') == '짧다'
    long = '가' * (settings.otel_text_limit + 100)
    clipped = otel.clip(long)
    assert clipped.endswith('…[절단]')
    assert len(clipped) == settings.otel_text_limit + len('…[절단]')
    assert otel.clip(None) == ''
    assert otel.clip('') == ''


def test_start_turn_핸드오프_수명주기_no_op에서_무해():
    # 시작 → detach → (핸드오프 가정) 늦은 end — SSE 생성 경로의 수명주기 그대로
    sp, token = otel.start_turn()
    assert not sp.is_recording()
    otel.set_attrs(sp, {'kms.route': 'knowledge'})
    otel.detach_turn(token)
    otel.mark_error(sp, RuntimeError('x'))   # no-op 스팬엔 기록 안 됨, 예외도 없어야
    sp.end()


def test_set_documents_no_op에서_무해():
    class _C:
        chunk_id, text = 1, '본문'          # set_documents가 읽는 두 속성뿐
    with otel.span('t', 'RETRIEVER') as sp:
        otel.set_documents(sp, [_C()])       # 예외 없이 통과하면 충분 (no-op)


def test_record_turn_result_no_op에서_무해_그리고_clip_비용_회피():
    """턴 결과 기록(#54, #56에서 TurnResult 소비로 전환) — no-op 스팬에서 무해해야 하고,
    is_recording 가드가 먼저라 clip(문자열 절단) 비용도 없어야 한다(otel.py 모듈 docstring 규약)."""
    from unittest.mock import patch
    from rag.streaming import TurnResult, _record_turn_result
    from schemas.kms import SourceCitation

    sp, token = otel.start_turn()
    assert not sp.is_recording()
    try:
        with patch.object(otel, 'clip', side_effect=AssertionError('no-op인데 clip이 호출됨')):
            _record_turn_result(sp, TurnResult(
                answer='답변' * 1000, finish_reason='done', latency_ms=1234,
                citations=[SourceCitation(document_id=1, filename='a.pdf', version=1)]))
            _record_turn_result(sp, TurnResult(                     # None latency도 무해
                answer='', finish_reason='failed', latency_ms=None, citations=[]))
    finally:
        otel.detach_turn(token)
        sp.end()
