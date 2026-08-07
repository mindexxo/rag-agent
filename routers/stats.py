"""운영 리포트 API (지표 MVP).

Alli형 지표 — 사용량·답변률·지식 갭. 전부 messages 원천 데이터의 기간 집계 (별도 수집 인프라 없음).
"운영자가 보고 행동할 수 있는 것만" 노출한다 (2026-08-07 정리):
- 관여율/해결률: agent-assist 특성상 관찰 불가라 영구 제외 (봇이 상담을 끝내는 제품 아님 — 7/18)
- 활성 사용자: 인증 전엔 X-User-Id 하드코딩이라 항상 1 — 진짜 수치가 될 때까지 미노출
- 응답 지연·캐시 적중·차단/실패: 내부 모니터링(SLO·알람) 소관 — 원천 컬럼은 계속 저장하므로 필요 시 SQL로
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

    # 기간 경계는 모든 쿼리가 '달력일' 기준을 공유한다 — days=7이면 오늘 포함 최근 7일(일 단위 절단).
    # 롤링 윈도(now()-7일)를 쓰면 윈도 첫 부분날의 데이터가 "질문 수엔 있는데 일별 합계엔 없는"
    # 불일치가 난다 (일별 표는 날짜 단위라 부분날을 표현할 수 없음).
    #
    # 답변률: 분모는 KNOWLEDGE 인텐트만 — OTHER(인사·요약 등)는 거절이 나올 수 없는 경로라
    # 분모에 넣으면 답변률이 잡담 비율만큼 부풀어 "지식 커버리지 신호"가 못 된다 (2026-08-07).
    # 분자도 같은 조건 필수 — intent NULL(컬럼 도입 전 행)의 거절이 분자에만 들어가면
    # 분자가 분모의 부분집합이 아니게 돼 거절률이 1을 넘거나 답변률이 음수가 된다.
    row = (await session.execute(text("""
        SELECT
          count(*) FILTER (WHERE role = 'user')                                        AS questions,
          count(*) FILTER (WHERE role = 'assistant' AND status = 'done'
                             AND intent = 'KNOWLEDGE')                                 AS knowledge_done,
          count(*) FILTER (WHERE role = 'assistant' AND status = 'done'
                             AND intent = 'KNOWLEDGE' AND is_refusal)                  AS refusals
        FROM messages
        WHERE tenant_id = :tenant_id
          AND created_at >= (now() - make_interval(days => :days))::date + 1
    """), params)).one()

    # 질문 없는 날도 0으로 채운다 — 빠진 행은 FE에서 "구멍 난 표"가 되고 추이 감이 왜곡된다.
    daily_rows = (await session.execute(text("""
        SELECT to_char(d, 'YYYY-MM-DD') AS d, count(m.id) AS n
        FROM generate_series(
               (now() - make_interval(days => :days))::date + 1, now()::date, '1 day') AS d
        LEFT JOIN messages m
          ON m.created_at::date = d
         AND m.tenant_id = :tenant_id AND m.role = 'user'
        GROUP BY 1 ORDER BY 1
    """), params)).all()

    # 인용 top 문서 — 저장 시 확정된 실인용 목록(cited_docs)만 집계. 본문 재파싱·전송 없음.
    # (저장 시점에 cited_filenames()가 답변 텍스트와 인용 라벨을 대조해 확정 — 단일 정의점)
    top_rows = (await session.execute(text("""
        SELECT d AS filename, count(*) AS cnt
        FROM messages, jsonb_array_elements_text(cited_docs) AS d
        WHERE tenant_id = :tenant_id AND role = 'assistant'
          AND jsonb_typeof(cited_docs) = 'array'
          AND created_at >= (now() - make_interval(days => :days))::date + 1
        GROUP BY 1 ORDER BY cnt DESC LIMIT 10
    """), params)).all()

    knowledge_done = row.knowledge_done or 0
    return StatsSummary(
        period_days=days,
        questions=row.questions or 0,
        knowledge_done=knowledge_done,
        refusals=row.refusals or 0,
        refusal_rate=round((row.refusals or 0) / knowledge_done, 3) if knowledge_done else 0.0,
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
          AND a.created_at >= (now() - make_interval(days => :days))::date + 1
        ORDER BY a.created_at DESC
        LIMIT :limit
    """), {'tenant_id': tenant_id, 'days': days, 'limit': limit})).all()
    return [UnansweredItem(question=r.question, asked_at=r.asked_at) for r in rows]
