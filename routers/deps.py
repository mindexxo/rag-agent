"""라우터 공용 의존성 — 요청 헤더에서 테넌트·사용자를 꺼낸다."""

from fastapi import Header


def get_tenant_id(x_tenant_id: str = Header(..., min_length=1, pattern=r'\S')) -> str:
    """X-Tenant-Id 필수 — 없거나 공백뿐이면 422. 빈 테넌트로 행이 쌓이면 격리가 깨진다."""
    return x_tenant_id


def get_user_id(x_user_id: str | None = Header(None, pattern=r'\S')) -> str | None:
    """X-User-Id (선택). 있으면 사용자별 동시 제한·대화 소유 판정에 쓰고, 없으면 테넌트 제한만.
    보냈는데 공백뿐이면 422 — 폴백 사용자로 조용히 합쳐지지 않게."""
    return x_user_id
