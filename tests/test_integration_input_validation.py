"""입력 검증 경계 — 헤더 빈 값, 공백만인 입력, 길이 상한, 첨부 형식·빈 텍스트, FAQ 요청 안 중복.

HTTP로만 확인한다. 거절 응답은 상태코드와 함께 본문 문구(FE가 그대로 보여주는 것)를 단언하고,
통과 응답은 공백 제거 등 저장값까지 본다 — 검증 기준과 저장 기준이 같아야 한다.
"""

import pytest
from httpx import ASGITransport, AsyncClient

from schemas.conversations import CONVERSATION_SEARCH_MAX
from schemas.kms import ATTACHMENT_FILENAME_MAX, FOLDER_NAME_MAX
from tests.conftest import register_faq, sse_meta
from tests.helpers_documents import MD, upload

ISO_CASES = [
    ('tenant-blank', {'X-Tenant-Id': ''}, 'x-tenant-id'),
    ('tenant-spaces', {'X-Tenant-Id': '   '}, 'x-tenant-id'),
    ('user-blank', {'X-Tenant-Id': 't-ok', 'X-User-Id': ''}, 'x-user-id'),
    ('user-spaces', {'X-Tenant-Id': 't-ok', 'X-User-Id': ' '}, 'x-user-id'),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(('_name', 'headers', 'bad_header'), ISO_CASES, ids=[c[0] for c in ISO_CASES])
async def test_빈_테넌트_사용자_헤더는_422(_loop_hygiene, _name, headers, bad_header):
    """ISO-02·ISO-17 — 빈 문자열·공백뿐인 헤더는 없는 것과 같이 422. 빈 테넌트로 행이 생기면 격리가 깨진다."""
    from main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url='http://testserver', headers=headers) as c:
        res = await c.get('/kms/conversations')
    assert res.status_code == 422
    assert ('header', bad_header) in [tuple(e['loc']) for e in res.json()['detail']]


@pytest.mark.asyncio
async def test_공백만인_질의는_422(client, tenant_id, fake_llm):
    """QRY-08 — 분류기에 빈 질문이 가지 않는다."""
    res = await client.post('/kms/query', json={'query': '   '})
    assert res.status_code == 422
    assert [tuple(e['loc']) for e in res.json()['detail']] == [('body', 'query')]


@pytest.mark.asyncio
async def test_질의는_앞뒤_공백을_뗀_값으로_저장된다(client, tenant_id, fake_llm, pass_gate):
    await register_faq(client)
    res = await client.post('/kms/query', json={'query': '  환불 기간 알려줘  '})
    assert res.status_code == 200
    conv = sse_meta(res)['conversation_id']
    msgs = (await client.get(f'/kms/conversations/{conv}/messages')).json()
    assert msgs[0]['content'] == '환불 기간 알려줘'


@pytest.mark.asyncio
async def test_대화_제목은_공백만이면_422이고_저장은_공백을_뗀다(client, tenant_id, fake_llm, pass_gate):
    """CNV-23 — FAQ 카테고리 이름과 같은 규칙."""
    await register_faq(client)
    conv = sse_meta(await client.post('/kms/query', json={'query': '환불 기간 알려줘'}))['conversation_id']

    assert (await client.patch(f'/kms/conversations/{conv}', json={'title': '   '})).status_code == 422
    res = await client.patch(f'/kms/conversations/{conv}', json={'title': '  환불 문의  '})
    assert res.status_code == 200
    assert res.json()['title'] == '환불 문의'


@pytest.mark.asyncio
async def test_대화_검색어는_상한을_넘으면_422(client, tenant_id):
    """CNV-08."""
    assert (await client.get('/kms/conversations', params={'q': 'x' * CONVERSATION_SEARCH_MAX})).status_code == 200
    res = await client.get('/kms/conversations', params={'q': 'x' * (CONVERSATION_SEARCH_MAX + 1)})
    assert res.status_code == 422
    assert [tuple(e['loc']) for e in res.json()['detail']] == [('query', 'q')]


@pytest.mark.asyncio
async def test_폴더_이름은_상한과_공백_규칙을_따른다(client, tenant_id):
    """FOL-04 — FAQ 카테고리 이름과 같은 상한(50)·공백 제거·빈 값 422. 등록과 수정 둘 다."""
    assert (await client.post('/kms/folders', json={'name': '   '})).status_code == 422
    assert (await client.post('/kms/folders', json={'name': 'ㄱ' * (FOLDER_NAME_MAX + 1)})).status_code == 422

    res = await client.post('/kms/folders', json={'name': '  반품 규정  '})
    assert res.status_code == 200
    folder = res.json()
    assert folder['name'] == '반품 규정'

    assert (await client.patch(f'/kms/folders/{folder["id"]}', json={'name': ' '})).status_code == 422
    assert (
        await client.patch(f'/kms/folders/{folder["id"]}', json={'name': 'ㄱ' * FOLDER_NAME_MAX})
    ).status_code == 200


@pytest.mark.asyncio
async def test_FAQ_요청_안의_자기_중복은_조용히_제거된다(client, tenant_id, fake_queue):
    """FAQ-C16 — 질문과 같은 유사 질문, 유사 질문끼리 중복은 거절하지 않고 뺀다(순서 유지). 등록·수정 둘 다."""
    res = await client.post(
        '/kms/faqs',
        json={
            'question': '환불 기간은?',
            'answer': '14일',
            'variants': ['환불 기간은?', '환불 언제', '환불 언제', '반품 기간'],
        },
    )
    assert res.status_code == 200, res.text
    assert res.json()['variants'] == ['환불 언제', '반품 기간']

    res = await client.patch(f'/kms/faqs/{res.json()["id"]}', json={'variants': ['환불 기간은?', '환불 언제']})
    assert res.status_code == 200, res.text
    assert res.json()['variants'] == ['환불 언제']


@pytest.mark.asyncio
async def test_업로드_파일명은_상한을_넘으면_422(client, tenant_id, fake_queue, blob_tmp):
    """DOC-14 — 첨부와 같은 상한. 상한 길이는 통과하고 경로 문자는 정제 없이 그대로 저장된다."""
    exact = 'ㄱ' * (ATTACHMENT_FILENAME_MAX - 3) + '.md'
    assert len(exact) == ATTACHMENT_FILENAME_MAX
    assert (await upload(client, exact, MD))['filename'] == exact

    res = await client.post(
        '/kms/documents', files={'file': ('ㄱ' * (ATTACHMENT_FILENAME_MAX - 2) + '.md', MD, 'text/markdown')}
    )
    assert res.status_code == 422
    assert '파일명이 너무 깁니다' in res.json()['detail']

    assert (await upload(client, '../상위.md', MD))['filename'] == '../상위.md'


@pytest.mark.asyncio
async def test_첨부_추출은_업로드와_같은_형식만_받는다(client, tenant_id):
    """ATT-03 — 허용 목록 밖은 400(업로드와 같은 문구). 확장자 없음도 같다."""
    for name in ('악성.exe', '한글.hwp', '확장자없음'):
        res = await client.post(
            '/kms/attachments/extract', files={'file': (name, b'hello', 'application/octet-stream')}
        )
        assert res.status_code == 400, name
        assert res.json()['detail'].startswith('지원하지 않는 형식입니다')

    res = await client.post(
        '/kms/attachments/extract', files={'file': ('메모.txt', '반품은 14일.'.encode(), 'text/plain')}
    )
    assert res.status_code == 200 and res.json()['text'] == '반품은 14일.'


@pytest.mark.asyncio
async def test_첨부_추출_결과가_비면_422(client, tenant_id):
    """ATT-09 — 빈 파일·공백뿐인 파일은 성공으로 보이지 않는다."""
    for content in (b'', b'   \n\n'):
        res = await client.post('/kms/attachments/extract', files={'file': ('빈파일.txt', content, 'text/plain')})
        assert res.status_code == 422, content
        assert res.json() == {'detail': '텍스트를 추출할 수 없는 파일입니다.'}
