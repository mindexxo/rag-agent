"""테넌트 헤더를 실은 인메모리 ASGI 클라이언트 — 교차 테넌트·헤더 없음 시나리오용."""

from httpx import ASGITransport, AsyncClient


def client_for(tenant: str | None) -> AsyncClient:
    """tenant가 None이면 X-Tenant-Id 헤더 없이 붙는다."""
    from main import app

    headers = {'X-Tenant-Id': tenant} if tenant is not None else {}
    return AsyncClient(transport=ASGITransport(app=app), base_url='http://testserver', headers=headers)
