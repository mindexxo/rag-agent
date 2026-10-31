"""테넌트 경계 — 헤더 필수·단건 리소스·집계가 다른 테넌트의 행을 보지 않는지 HTTP로 고정한다.

RLS 없이 WHERE 절이 유일한 방어선이라, 목록·검색 격리(test_integration_isolation.py)에 더해
id를 직접 지목하는 단건 경로와 집계 SQL(통계·FAQ 인용 수)까지 확인한다.
"""

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import func, select

from database import AsyncSessionLocal
from rag.models import Conversation, Document, Message
from tests.conftest import sse_meta
from tests.helpers_documents import MD, upload

# 테넌트 헤더 없이 동작하는 것이 현재 동작인 경로 — 검증 기준은 스펙 결정 대기다.
NO_TENANT_HEADER_ROUTES = {('POST', '/kms/attachments/extract')}


def _client_for(tenant: str | None) -> AsyncClient:
    from main import app

    headers = {'X-Tenant-Id': tenant} if tenant is not None else {}
    return AsyncClient(transport=ASGITransport(app=app), base_url='http://testserver', headers=headers)


def _kms_routes() -> list[tuple[str, str, str]]:
    """(메서드, 경로 패턴, 경로 변수를 1로 채운 URL) — 헤더 검증은 경로 변수 값과 무관하게 함께 집계된다."""
    from fastapi.routing import APIRoute

    from main import app

    return sorted(
        (method, route.path, route.path.format(**{name: '1' for name in route.param_convertors}))
        for route in app.routes
        if isinstance(route, APIRoute) and route.path.startswith('/kms')
        for method in route.methods
    )


@pytest.mark.asyncio
async def test_KMS_전_라우트는_테넌트_헤더가_없으면_422(_loop_hygiene):
    """ISO-01 — 라우트를 앱에서 직접 열거한다. 새 라우트가 헤더 의존성을 빠뜨리면 여기서 걸린다."""
    routes = _kms_routes()
    listed = {(method, path) for method, path, _ in routes}
    assert ('POST', '/kms/query') in listed  # 열거가 비어 통과하는 일을 막는다
    assert NO_TENANT_HEADER_ROUTES <= listed  # 제외 목록이 사라진 라우트를 가리키면 알아챈다

    async with _client_for(None) as bare:
        for method, path, url in routes:
            if (method, path) in NO_TENANT_HEADER_ROUTES:
                continue
            res = await bare.request(method, url)
            assert res.status_code == 422, f'{method} {path}: {res.status_code}'
            locs = [tuple(e['loc']) for e in res.json()['detail']]
            assert ('header', 'x-tenant-id') in locs, f'{method} {path}: 422 원인이 헤더가 아니다 {locs}'


@pytest.mark.asyncio
async def test_다른_테넌트_문서는_단건_조회_수정_삭제_다운로드_모두_404(
    client, tenant_id, other_tenant_id, fake_queue, blob_tmp
):
    """ISO-08 — 일괄 경로와 별개로, id를 직접 지목하는 단건 경로도 존재 자체를 알려주지 않는다."""
    doc_id = (await upload(client, 'A사문서.md', MD))['document_id']
    assert (await client.get(f'/kms/documents/{doc_id}')).status_code == 200  # 양성 대조 — 주인에겐 보인다
    assert (await client.get(f'/kms/documents/{doc_id}/download')).status_code == 200

    async with _client_for(other_tenant_id) as other:
        assert (await other.get(f'/kms/documents/{doc_id}')).status_code == 404
        assert (await other.get(f'/kms/documents/{doc_id}/download')).status_code == 404
        assert (await other.patch(f'/kms/documents/{doc_id}', json={'is_searchable': False})).status_code == 404
        assert (await other.delete(f'/kms/documents/{doc_id}')).status_code == 404

    async with AsyncSessionLocal() as s:
        doc = await s.get(Document, doc_id)
    assert doc is not None
    assert doc.status != 'deleted' and doc.is_searchable is True, '남의 요청으로 문서가 바뀌었다'


@pytest.mark.asyncio
async def test_다른_테넌트_대화는_제목_변경과_삭제가_404(client, tenant_id, other_tenant_id, fake_llm):
    """ISO-11 — 사용자 헤더는 양쪽 모두 생략(같은 기본 사용자)이라 테넌트만 다르다."""
    conv_id = sse_meta(await client.post('/kms/query', json={'query': 'A사 질문입니다'}))['conversation_id']

    async with _client_for(other_tenant_id) as other:
        assert (await other.patch(f'/kms/conversations/{conv_id}', json={'title': '탈취'})).status_code == 404
        assert (await other.delete(f'/kms/conversations/{conv_id}')).status_code == 404

    async with AsyncSessionLocal() as s:
        conv = await s.get(Conversation, conv_id)
    assert conv is not None
    assert conv.title != '탈취' and conv.deleted_at is None
    assert (
        await client.patch(f'/kms/conversations/{conv_id}', json={'title': '내 제목'})
    ).status_code == 200  # 양성 대조


@pytest.mark.asyncio
async def test_다른_테넌트_대화_id로_질의하면_404이고_어느_쪽에도_행이_생기지_않는다(
    client, tenant_id, other_tenant_id, fake_llm
):
    """ISO-12 — 대화 소유 확인은 턴을 열기 전이라 질문 행도 남지 않는다."""
    conv_id = sse_meta(await client.post('/kms/query', json={'query': 'A사 질문입니다'}))['conversation_id']
    before = await _message_count(tenant_id)

    async with _client_for(other_tenant_id) as other:
        res = await other.post('/kms/query', json={'query': '남의 대화에 끼어들기', 'conversation_id': conv_id})
    assert res.status_code == 404
    assert res.json() == {'detail': '대화를 찾을 수 없습니다.'}
    assert await _message_count(tenant_id) == before
    assert await _message_count(other_tenant_id) == 0


async def _message_count(tenant: str) -> int:
    async with AsyncSessionLocal() as s:
        return (
            await s.execute(select(func.count()).select_from(Message).where(Message.tenant_id == tenant))
        ).scalar_one()


async def _seed_answer(
    tenant: str, question: str, *, cited_docs: list[str] | None = None, cited_faq_ids: list[int] | None = None
) -> None:
    """집계가 읽는 모양 그대로의 한 턴(user + 완료된 KNOWLEDGE 답변)을 넣는다."""
    async with AsyncSessionLocal() as s:
        conv = Conversation(tenant_id=tenant, created_by='agent-a')
        s.add(conv)
        await s.flush()
        user = Message(tenant_id=tenant, conversation_id=conv.id, role='user', content=question)
        s.add(user)
        await s.flush()
        s.add(
            Message(
                tenant_id=tenant,
                conversation_id=conv.id,
                role='assistant',
                content='답변',
                status='done',
                intent='KNOWLEDGE',
                question_message_id=user.id,
                cited_docs=cited_docs or [],
                cited_faq_ids=cited_faq_ids or [],
            )
        )
        await s.commit()


@pytest.mark.asyncio
async def test_통계는_인용_문서와_미답변까지_다른_테넌트를_세지_않는다(client, tenant_id, other_tenant_id):
    """STAT-27 — 질문 수만이 아니라 인용 순위(문서·FAQ 두 갈래)와 미답변 목록까지 본다.

    B의 답변이 A의 FAQ id를 인용한 행은 FAQ 조인의 테넌트 조건을 겨눈다 — 빠지면 B의 순위에 A의 질문이 뜬다.
    """
    faq_id = (
        await client.post('/kms/faqs', json={'question': 'A사 환불 기간은?', 'variants': [], 'answer': '14일'})
    ).json()['id']
    await _seed_answer(tenant_id, 'A 근거 있음', cited_docs=['A사규정.pdf'], cited_faq_ids=[faq_id])
    await _seed_answer(tenant_id, 'A 미답변')
    await _seed_answer(other_tenant_id, 'B 남의 FAQ 인용', cited_faq_ids=[faq_id])

    mine = (await client.get('/kms/stats', params={'days': 1})).json()
    assert {d['filename'] for d in mine['top_documents']} == {'A사규정.pdf', 'FAQ: A사 환불 기간은?'}
    assert (mine['questions'], mine['knowledge_done'], mine['ungrounded']) == (2, 2, 1)
    mine_unanswered = (await client.get('/kms/stats/unanswered', params={'days': 1})).json()
    assert [u['question'] for u in mine_unanswered] == ['A 미답변']

    async with _client_for(other_tenant_id) as other:
        theirs = (await other.get('/kms/stats', params={'days': 1})).json()
        unanswered = (await other.get('/kms/stats/unanswered', params={'days': 1})).json()
    assert theirs['top_documents'] == [], f'남의 인용이 순위에 섞였다: {theirs["top_documents"]}'
    assert (theirs['questions'], theirs['knowledge_done']) == (1, 1)
    assert [u['question'] for u in unanswered] == [], 'A의 미답변이 B 목록에 떴다'


@pytest.mark.asyncio
async def test_FAQ_인용_수는_다른_테넌트_메시지를_세지_않는다(client, tenant_id, other_tenant_id):
    """FAQ-L14 — 인용 수 집계는 FAQ id로만 좁히므로, 같은 id를 인용한 남의 메시지가 섞이면 안 된다."""
    faq_id = (
        await client.post('/kms/faqs', json={'question': 'A사 환불 기간은?', 'variants': [], 'answer': '14일'})
    ).json()['id']
    await _seed_answer(other_tenant_id, 'B 남의 FAQ 인용', cited_faq_ids=[faq_id])
    await _seed_answer(other_tenant_id, 'B 또 인용', cited_faq_ids=[faq_id])

    items = (await client.get('/kms/faqs')).json()['items']
    assert [(i['id'], i['ref_count']) for i in items] == [(faq_id, 0)]

    await _seed_answer(tenant_id, 'A 자기 FAQ 인용', cited_faq_ids=[faq_id])  # 양성 대조 — 자기 인용은 센다
    items = (await client.get('/kms/faqs')).json()['items']
    assert [(i['id'], i['ref_count']) for i in items] == [(faq_id, 1)]
