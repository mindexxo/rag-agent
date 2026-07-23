"""운영 리포트 API (지표 MVP).

Alli형 지표 — 사용량·답변률·캐시·지식 갭. 관여율/해결률은 agent-assist 특성상
관찰 불가라 영구 제외 (봇이 상담을 직접 끝내는 제품이 아님 — 2026-07-18 결정).
전부 messages/answer 원천 데이터의 기간 집계 — 별도 수집 인프라 없음.
"""
from fastapi import APIRouter, Depends, Query
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_session
from routers.kms import get_tenant_id
from schemas.stats import DailyCount, StatsSummary, TopDocument, UnansweredItem

router = APIRouter(prefix='/kms')


@router.get('/stats', response_model=StatsSummary)
async def stats_summary(
        days: int = Query(7, ge=1, le=90),
        tenant_id: str = Depends(get_tenant_id),
        session: AsyncSession = Depends(get_session),
):
    params = {'tenant_id': tenant_id, 'days': days}

    row = (await session.execute(text("""
        SELECT
          count(*) FILTER (WHERE role = 'user')                                        AS questions,
          count(DISTINCT user_id) FILTER (WHERE role = 'user' AND user_id IS NOT NULL) AS active_users,
          count(*) FILTER (WHERE role = 'assistant' AND status = 'done')               AS done,
          count(*) FILTER (WHERE role = 'assistant' AND status = 'done'
                             AND is_refusal)                                           AS refusals,
          count(*) FILTER (WHERE role = 'assistant' AND status = 'blocked')            AS blocked,
          count(*) FILTER (WHERE role = 'assistant' AND status = 'failed')             AS failed,
          avg(latency_ms) FILTER (WHERE role = 'assistant' AND latency_ms IS NOT NULL) AS avg_latency,
          percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_ms)
            FILTER (WHERE role = 'assistant' AND latency_ms IS NOT NULL)               AS p95_latency
        FROM messages
        WHERE tenant_id = :tenant_id
          AND created_at >= now() - make_interval(days => :days)
    """), params)).one()

    daily_rows = (await session.execute(text("""
        SELECT to_char(created_at::date, 'YYYY-MM-DD') AS d, count(*) AS n
        FROM messages
        WHERE tenant_id = :tenant_id AND role = 'user'
          AND created_at >= now() - make_interval(days => :days)
        GROUP BY 1 ORDER BY 1
    """), params)).all()

    # 인용 top 문서 — 저장 시 확정된 실인용 목록(cited_docs)만 집계. 본문 재파싱·전송 없음.
    # (저장 시점에 cited_filenames()가 답변 텍스트와 인용 라벨을 대조해 확정 — 단일 정의점)
    top_rows = (await session.execute(text("""
        SELECT d AS filename, count(*) AS cnt
        FROM messages, jsonb_array_elements_text(cited_docs) AS d
        WHERE tenant_id = :tenant_id AND role = 'assistant'
          AND jsonb_typeof(cited_docs) = 'array'
          AND created_at >= now() - make_interval(days => :days)
        GROUP BY 1 ORDER BY cnt DESC LIMIT 10
    """), params)).all()

    done = row.done or 0
    return StatsSummary(
        period_days=days,
        questions=row.questions or 0,
        active_users=row.active_users or 0,
        refusals=row.refusals or 0,
        refusal_rate=round((row.refusals or 0) / done, 3) if done else 0.0,
        blocked=row.blocked or 0,
        failed=row.failed or 0,
        avg_latency_ms=int(row.avg_latency) if row.avg_latency is not None else None,
        p95_latency_ms=int(row.p95_latency) if row.p95_latency is not None else None,
        daily=[DailyCount(date=r.d, questions=r.n) for r in daily_rows],
        top_documents=[TopDocument(filename=r.filename, citations=r.cnt) for r in top_rows],
    )


@router.get('/stats/unanswered', response_model=list[UnansweredItem])
async def stats_unanswered(
        days: int = Query(7, ge=1, le=90),
        limit: int = Query(50, ge=1, le=200),
        tenant_id: str = Depends(get_tenant_id),
        session: AsyncSession = Depends(get_session),
):
    """지식 갭 — 거절당한 질문 원문 (최신순). FAQ/문서 보강의 직접 재료.

    거절 여부·질문 짝 모두 저장 시 확정된 컬럼(is_refusal, question_message_id) —
    본문 재파싱·휴리스틱 없음.
    """
    rows = (await session.execute(text("""
        SELECT u.content AS question, a.created_at AS asked_at
        FROM messages a
        JOIN messages u ON u.id = a.question_message_id
                       AND u.tenant_id = a.tenant_id
        WHERE a.tenant_id = :tenant_id
          AND a.role = 'assistant' AND a.status = 'done' AND a.is_refusal
          AND a.created_at >= now() - make_interval(days => :days)
        ORDER BY a.created_at DESC
        LIMIT :limit
    """), {'tenant_id': tenant_id, 'days': days, 'limit': limit})).all()
    return [UnansweredItem(question=r.question, asked_at=r.asked_at) for r in rows]
