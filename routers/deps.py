"""라우터 공용 의존성 — 요청 헤더에서 테넌트·사용자를 꺼내고, 목록 페이징 파라미터를 정리한다."""

from typing import NamedTuple

from fastapi import Header


def get_tenant_id(x_tenant_id: str = Header(..., min_length=1, pattern=r'\S')) -> str:
    """X-Tenant-Id 필수 — 없거나 공백뿐이면 422. 빈 테넌트로 행이 쌓이면 격리가 깨진다."""
    return x_tenant_id


def get_user_id(x_user_id: str | None = Header(None, pattern=r'\S')) -> str | None:
    """X-User-Id (선택). 있으면 사용자별 동시 제한·대화 소유 판정에 쓰고, 없으면 테넌트 제한만.
    보냈는데 공백뿐이면 422 — 폴백 사용자로 조용히 합쳐지지 않게."""
    return x_user_id


class Page(NamedTuple):
    limit: int
    offset: int
    q: str | None


def sanitize_page(limit: int, offset: int, q: str | None, max_limit: int) -> Page:
    """목록 API 공용 파라미터 위생 — 범위 밖 limit은 상한으로, 음수 offset은 0으로(거부 대신 sane 결과 — 음수 500·
    무상한 조회 방지). 빈 문자열·공백뿐인 q는 미전송과 같다 — 필터로 쓰면 '%%'가 되어 전건 매칭이다."""
    return Page(
        limit if 1 <= limit <= max_limit else max_limit,
        max(offset, 0),
        (q or '').strip() or None,
    )
