"""응답 스키마 공용 타입 — 시각 직렬화 (#181).

API 응답의 시각은 **KST 오프셋(+09:00) ISO 문자열**로 나간다. 저장은 그대로 UTC(TIMESTAMPTZ,
#164)고 바뀌는 건 응답 직렬화만이다 — 국내 상담 시스템이라 표시 기준이 KST 하나인데, UTC `Z`로
내려주면 FE가 문자열을 잘라 쓰는 곳(`uploaded_at.slice(0, 10)`)에서 새벽 업로드가 전날 날짜로
보였다(2026-09-19 실측). 오프셋을 붙이므로 시간대 정보를 잃지 않고, `new Date()`로 파싱하는
FE 코드도 그대로 맞다.

시간대는 배포 서버의 tz 설정에 기대지 않고 여기 못 박는다 — DB 세션 tz는 Etc/UTC고(실측),
그 값이 바뀌어도 응답은 흔들리지 않아야 한다. 통계의 날짜 경계(routers/stats.py)도 같은
상수를 바인드해 쓴다 — "오늘"의 기준이 응답 시각과 어긋나지 않게.
"""
from datetime import datetime, timezone
from typing import Annotated
from zoneinfo import ZoneInfo

from pydantic import PlainSerializer, WithJsonSchema

KST = ZoneInfo('Asia/Seoul')


def to_kst_iso(v: datetime) -> str:
    """aware datetime → KST 오프셋 ISO 문자열. naive가 오면 UTC로 본다 — DB는 aware로 주므로
    (rag/models.py의 Base가 TIMESTAMPTZ 매핑) 방어일 뿐이고, 로컬 tz로 오해하는 것보다 안전하다."""
    if v.tzinfo is None:
        v = v.replace(tzinfo=timezone.utc)
    return v.astimezone(KST).isoformat()


# 응답 스키마의 모든 시각 필드가 이 타입을 쓴다 — 필드마다 serializer를 붙이면 형식이 갈린다.
# 입력(파싱)은 보통 datetime과 같고, 출력만 KST 문자열이다. `KstDatetime | None`에서 None은
# serializer를 타지 않고 null로 나간다(실측).
KstDatetime = Annotated[
    datetime,
    PlainSerializer(to_kst_iso, return_type=str),
    # return_type=str만 두면 FastAPI의 **응답(직렬화) 스키마**에서 format: date-time이 빠져
    # 평범한 string으로 보인다(실측) — Swagger에서 시각 필드임이 사라지고, date-time을 Date로
    # 매핑하는 코드젠이 문자열로 받는다. 출력 스키마에만 원래 형식을 되살린다.
    WithJsonSchema({'type': 'string', 'format': 'date-time'}, mode='serialization'),
]
