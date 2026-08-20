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
    # 답변률: 분모는 KNOWLEDGE 인텐트만 — OTHER(인사·요약 등)는 애초에 근거를 댈 일이 없는
    # 경로라 분모에 넣으면 답변률이 잡담 비율만큼 부풀어 "지식 커버리지 신호"가 못 된다
    # (2026-08-07). #61 이후 이 필터는 **더 중요해졌다** — 판정이 인용 개수로 바뀌면서 OTHER는
    # 꼬리 메커니즘 자체가 없어 항상 인용 0건이 되므로, 필터가 없으면 잡담이 전부 분자에
    # 들어간다(같은 이유로 stats_unanswered에도 이 필터를 새로 넣었다).
    # 분자도 같은 조건 필수 — intent NULL(컬럼 도입 전 행)이 분자에만 들어가면 분자가 분모의
    # 부분집합이 아니게 돼 근거미확인율이 1을 넘거나 답변률이 음수가 된다.
    row = (await session.execute(text("""
        SELECT
          count(*) FILTER (WHERE role = 'user')                                        AS questions,
          count(*) FILTER (WHERE role = 'assistant' AND status = 'done'
                             AND intent = 'KNOWLEDGE')                                 AS knowledge_done,
          -- 근거없음(#61): 거절 문구 부분일치(옛 is_refusal 컬럼)를 실인용 0건으로 대체했다.
          -- 폐기 사유·실측은 rag/citation_tail.py 모듈 docstring(단일 정의점).
          -- coalesce: cited_docs NULL도 0건으로 본다 — routers/conversations.py의
          -- `m.cited_docs or []` 관례와 같은 결론. (assistant·done 행에는 NULL이 없는 것을
          -- 개발계에서 확인했지만, 컬럼이 nullable이라 방어는 남긴다.)
          count(*) FILTER (WHERE role = 'assistant' AND status = 'done'
                             AND intent = 'KNOWLEDGE'
                             AND coalesce(jsonb_array_length(cited_docs), 0) = 0)      AS ungrounded
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
    # (저장 시점에 출처 꼬리 해석(rag/citation_tail, #56)이 확정 — 단일 정의점.
    #  첨부 인용은 '첨부: 파일명' 문자열로 함께 잡힌다 — 첨부 기반 답변도 지표에 노출)
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
        ungrounded=row.ungrounded or 0,
        ungrounded_rate=round((row.ungrounded or 0) / knowledge_done, 3) if knowledge_done else 0.0,
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
    """지식 갭 — 근거를 못 댄 질문 원문 (최신순). FAQ/문서 보강의 직접 재료.

    판정·질문 짝 모두 저장 시 확정된 컬럼(cited_docs, question_message_id) —
    본문 재파싱·휴리스틱 없음.

    #61에서 판정이 "거절 문구"에서 "실인용 0건"으로 바뀌었다. 목록의 성격도 그만큼
    넓어진다 — 거절뿐 아니라 "근거 없이 답한" 답변도 여기 들어온다. KB 보강이라는
    목적에는 그게 맞다(둘 다 근거가 없다는 뜻이므로). 다만 API 응답만으로는 그 둘을
    구별할 수 없다는 것을 알고 볼 것 — 필요하면 messages.content를 직접 감사해야 한다.
    """
    rows = (await session.execute(text("""
        SELECT u.content AS question, a.created_at AS asked_at
        FROM messages a
        JOIN messages u ON u.id = a.question_message_id
                       AND u.tenant_id = a.tenant_id
        WHERE a.tenant_id = :tenant_id
          -- intent 필터는 #61에서 **새로 필수**가 됐다: 문구 판정 시절엔 OTHER(잡담)
          -- 답변이 우연히 거절 문구를 담을 일이 없어 안 걸렸는데, 인용 기반 판정에서는
          -- OTHER가 항상 인용 0건이다(꼬리 메커니즘 자체가 없다) — 이 필터가 없으면
          -- 잡담이 전부 지식 갭 목록을 덮는다(개발계 실측: OTHER·done 53건 전부 빈 배열).
          AND a.role = 'assistant' AND a.status = 'done' AND a.intent = 'KNOWLEDGE'
          AND coalesce(jsonb_array_length(a.cited_docs), 0) = 0
          AND a.created_at >= (now() - make_interval(days => :days))::date + 1
        ORDER BY a.created_at DESC
        LIMIT :limit
    """), {'tenant_id': tenant_id, 'days': days, 'limit': limit})).all()
    return [UnansweredItem(question=r.question, asked_at=r.asked_at) for r in rows]
