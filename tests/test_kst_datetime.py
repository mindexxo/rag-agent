"""응답 시각 직렬화(#181) — KstDatetime의 순수 동작. DB 불필요.

pytest tests/test_kst_datetime.py 만으로 몇 초에 돈다 (AGENTS.md의 순수 로직 묶음에 추가할 것).
"""
from datetime import datetime, timezone

from pydantic import BaseModel

from schemas.common import KST, KstDatetime, to_kst_iso


class _M(BaseModel):
    at: KstDatetime
    opt: KstDatetime | None = None


UTC_13 = datetime(2026, 9, 15, 13, 0, 53, 930031, tzinfo=timezone.utc)


def test_UTC를_KST_오프셋으로_내보낸다():
    out = _M(at=UTC_13).model_dump()['at']
    assert out == '2026-09-15T22:00:53.930031+09:00'
    assert out.endswith('+09:00')                      # Z가 아니라 오프셋 — 시간대 정보를 잃지 않는다


def test_시점은_바뀌지_않는다_변환이지_이동이_아님():
    out = _M(at=UTC_13).model_dump()['at']
    assert datetime.fromisoformat(out) == UTC_13


def test_None은_serializer를_타지_않고_null():
    assert _M(at=UTC_13, opt=None).model_dump()['opt'] is None
    assert _M(at=UTC_13).model_dump_json().endswith('"opt":null}')


def test_naive는_UTC로_본다():
    """DB는 aware로 주지만(rag/models.py Base) 방어 — 로컬 tz로 오해하면 9시간 어긋난다."""
    assert to_kst_iso(UTC_13.replace(tzinfo=None)) == to_kst_iso(UTC_13)


def test_새벽_경계_날짜가_KST로_넘어간다():
    """UTC 15:00 = KST 00:00 — 문자열 앞 10자를 잘라 쓰는 FE가 전날로 보이던 지점."""
    utc_1500 = datetime(2026, 9, 15, 15, 0, tzinfo=timezone.utc)
    assert to_kst_iso(utc_1500)[:10] == '2026-09-16'


def test_시간대_상수는_서울_고정():
    assert KST.key == 'Asia/Seoul'                     # 통계 SQL이 :tz로 바인드하는 값


def test_응답_OpenAPI_스키마에_date_time_형식이_남는다():
    """PlainSerializer(return_type=str)만으로는 FastAPI 응답 스키마가 평범한 string이 된다(실측) —
    WithJsonSchema(mode='serialization')로 되살린 것을 고정한다. Swagger·코드젠이 시각임을 알아야 한다."""
    from main import app
    schemas = app.openapi()['components']['schemas']
    at = schemas['DocumentUploadResponse']['properties']['uploaded_at']
    assert at.get('format') == 'date-time' and at.get('type') == 'string'
    opt = schemas['DocumentExistsResponse']['properties']['uploaded_at']
    assert {'type': 'string', 'format': 'date-time'} in opt['anyOf']      # Optional도 형식 유지
