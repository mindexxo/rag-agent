"""멀티테넌트 격리 강제 진입점.

모든 멀티테넌트 SELECT는 tenant_scoped(model, tenant_id)를 거쳐야 함.
원시 select(Model)을 직접 호출하지 말 것 — 코드리뷰에서 리젝트.

이 프로젝트는 RLS를 사용하지 않고 WHERE 절 방식으로 격리.
누락 검출은 tests/test_tenant_isolation.py 통합 테스트가 담당.
"""
from typing import Any

from sqlalchemy import Select, select


def tenant_scoped(model: Any, tenant_id: str) -> Select:
    """tenant_id 필터가 미리 박힌 SELECT 시작점."""
    return select(model).where(model.tenant_id == tenant_id)
