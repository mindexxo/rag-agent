"""턴·스트림 테스트 공용 도우미 — 정상 경로 prepare와 재접속 리더 출력 수집·파싱."""

import json

from database import AsyncSessionLocal
from rag.turn.prepared import PreparedRag
from rag.turn.service import RagService


async def prepare_turn(tenant_id: str) -> PreparedRag:
    """정상 경로로 prepare까지 만든다 — 자리표시(generating)는 prepare가 커밋한다. 그 뒤는 테스트가 조작한다."""
    async with AsyncSessionLocal() as session:
        svc = RagService(tenant_id=tenant_id, session=session, user_id='agent-x')
        return await svc.prepare('환불 기간 알려줘')


async def drain(agen, limit: int = 200) -> list[str]:
    """비동기 제너레이터를 끝까지(또는 limit개까지) 모은다."""
    out = []
    async for chunk in agen:
        out.append(chunk)
        if len(out) >= limit:
            break
    return out


def chunk_events(chunks: list[str]) -> list[tuple[str, dict]]:
    """SSE 문자열 조각들 → [(event, payload)] — 재접속 리더 출력 파싱. data 줄이 없는 블록은 건너뛴다."""
    out = []
    for raw in chunks:
        for block in raw.strip().split('\n\n'):
            lines = block.splitlines()
            if len(lines) < 2:
                continue
            out.append((lines[0].removeprefix('event: '), json.loads(lines[1].removeprefix('data: '))))
    return out
