"""문서 목록 — 페이징·검색·필터(#176)와 응답 시각 KST 오프셋(#181).

목록은 문서 관리 화면의 첫 화면이라 total·has_more·ref_count가 어긋나면 바로 보인다.
필터 조건은 count 쿼리와 페이지 쿼리가 **같은 것**을 써야 한다(routers/documents.py _list_where).
공용 헬퍼는 tests/helpers_documents.py.
"""
import json

import pytest

from database import AsyncSessionLocal
from rag.models import Conversation, Message
from tests.conftest import ingest
from tests.helpers_documents import MD, doc_data, get_doc, list_docs, make_folder, upload


# ── #176 목록 페이징 + 검색·필터 ──────────────────────────────

@pytest.mark.asyncio
async def test_목록_페이징_total과_has_more(client, tenant_id, fake_queue, blob_tmp):
    ids = [(await upload(client, f'정책{i}.md', MD))['document_id'] for i in range(5)]

    first = await list_docs(client, limit=2, offset=0)
    assert [d['document_id'] for d in first['items']] == ids[:-3:-1]   # 최신순
    assert first['total'] == 5 and first['has_more'] is True

    last = await list_docs(client, limit=2, offset=4)
    assert len(last['items']) == 1                    # 마지막은 부분 페이지
    assert last['total'] == 5 and last['has_more'] is False

    beyond = await list_docs(client, limit=2, offset=99)
    assert beyond['items'] == [] and beyond['total'] == 5 and beyond['has_more'] is False


@pytest.mark.asyncio
async def test_목록_limit은_범위를_벗어나면_상한으로(client, tenant_id, fake_queue, blob_tmp):
    await upload(client, '정책.md', MD)
    for bad in (0, -1, 9999):
        assert (await list_docs(client, limit=bad))['total'] == 1     # 422가 아니라 상한으로 클램프
    assert (await list_docs(client, offset=-5))['items']              # 음수 offset은 0으로


@pytest.mark.asyncio
async def test_목록_파일명_검색(client, tenant_id, fake_queue, blob_tmp):
    await upload(client, '환불정책.md', MD)
    await upload(client, '배송정책.md', MD)
    await upload(client, 'kms_01_비밀번호.md', MD)

    assert (await list_docs(client, q='환불'))['total'] == 1
    assert (await list_docs(client, q='정책'))['total'] == 2
    assert (await list_docs(client, q='없는이름'))['total'] == 0
    assert (await list_docs(client, q=''))['total'] == 3            # 빈 검색어는 필터 없음

    # LIKE 이스케이프 — _는 '아무 글자 하나'라, 이스케이프하지 않으면 kmsX01도 걸린다
    assert (await list_docs(client, q='kms_01'))['total'] == 1
    assert (await list_docs(client, q='kmsX01'))['total'] == 0


@pytest.mark.asyncio
async def test_목록_검색어가_NFD여도_걸린다(client, tenant_id, fake_queue, blob_tmp):
    """파일명은 NFC로 저장된다(#34). 이름을 복사해 붙여넣으면 NFD일 수 있다 — macOS 경로."""
    import unicodedata
    await upload(client, '환불정책.md', MD)
    nfd = unicodedata.normalize('NFD', '환불')
    assert nfd != '환불'                                  # 실제로 분해형이 맞는지 먼저 확인
    assert (await list_docs(client, q=nfd))['total'] == 1


@pytest.mark.asyncio
async def test_목록_폴더_필터와_미분류(client, tenant_id, fake_queue, blob_tmp):
    fid = await make_folder(client, '규정집')
    in_folder = await upload(client, '환불정책.md', MD, extra_parts=doc_data(folder_id=fid))
    unfiled = await upload(client, '배송정책.md', MD)

    assert [d['document_id'] for d in (await list_docs(client, folder_id=fid))['items']] \
        == [in_folder['document_id']]
    assert [d['document_id'] for d in (await list_docs(client, folder_id=0))['items']] \
        == [unfiled['document_id']]                       # 0 = 미분류
    assert (await list_docs(client))['total'] == 2            # 필터 없으면 전부


@pytest.mark.asyncio
async def test_목록_상태_필터(client, tenant_id, fake_queue, blob_tmp):
    doc = await upload(client, '환불정책.md', MD)         # 업로드 직후는 pending
    assert (await list_docs(client, status='pending'))['total'] == 1
    assert (await list_docs(client, status='ready'))['total'] == 0
    assert (await list_docs(client, status=['ready', 'pending']))['total'] == 1   # 복수 선택

    await ingest(doc['document_id'])                       # ready로 승격
    assert (await list_docs(client, status='ready'))['total'] == 1

    res = await client.get('/kms/documents', params={'status': 'done'})
    assert res.status_code == 422                          # 어휘에 없는 값은 조용히 0건이 아니라 거절


@pytest.mark.asyncio
async def test_목록_답변사용_필터(client, tenant_id, fake_queue, blob_tmp):
    on = await upload(client, '환불정책.md', MD)
    off = await upload(client, '배송정책.md', MD)
    assert (await client.patch(f"/kms/documents/{off['document_id']}",
                               json={'is_searchable': False})).status_code == 200

    assert [d['document_id'] for d in (await list_docs(client, is_searchable=True))['items']] \
        == [on['document_id']]
    assert [d['document_id'] for d in (await list_docs(client, is_searchable=False))['items']] \
        == [off['document_id']]


@pytest.mark.asyncio
async def test_목록_ref_count는_이번_페이지만_집계한다(client, tenant_id, fake_queue, blob_tmp):
    """집계를 페이지의 파일명으로 좁혔다(#176) — 좁히는 조건이 틀리면 0이 되거나 남의 값이 붙는다."""
    cited = await upload(client, '환불정책.md', MD)
    await upload(client, '배송정책.md', MD)

    async with AsyncSessionLocal() as s:
        conv = Conversation(tenant_id=tenant_id, created_by='agent-a')
        s.add(conv)
        await s.flush()
        s.add(Message(tenant_id=tenant_id, conversation_id=conv.id, role='assistant',
                      content='답변', cited_docs=['환불정책.md']))
        await s.commit()

    body = await list_docs(client)
    counts = {d['filename']: d['ref_count'] for d in body['items']}
    assert counts == {'환불정책.md': 1, '배송정책.md': 0}

    # 인용된 문서가 페이지에 없을 때도 다른 문서 값이 오염되지 않는다
    only_other = await list_docs(client, q='배송')
    assert [d['ref_count'] for d in only_other['items']] == [0]
    assert cited['document_id'] not in [d['document_id'] for d in only_other['items']]


@pytest.mark.asyncio
async def test_답변사용_필터는_문서_스위치_기준이다(client, tenant_id, fake_queue, blob_tmp):
    """#176 결정을 고정한다 — 폴더가 참조 off여도 **문서 스위치가 on이면** is_searchable=true에 잡힌다.

    실제로 검색에 쓰이는지는 폴더와의 곱(실효 참조)이지만, 그 판정을 이 필터에 넣는 것은
    제품 결정이지 버그 수정이 아니다(같은 규칙의 네 번째 사본이 된다는 이유로 미뤘다).
    나중에 '고치려는' 변경이 오면 이 테스트가 먼저 걸려 결정을 다시 보게 한다.
    """
    off_folder = await make_folder(client, '대외비')
    assert (await client.patch(f'/kms/folders/{off_folder}',
                               json={'is_searchable': False})).status_code == 200
    doc = await upload(client, '환불정책.md', MD, extra_parts=doc_data(folder_id=off_folder))
    assert doc['is_searchable'] is True                      # 문서 스위치는 그대로 on

    on = (await list_docs(client, is_searchable=True))['items']
    assert [d['document_id'] for d in on] == [doc['document_id']]
    assert (await list_docs(client, is_searchable=False))['items'] == []


@pytest.mark.asyncio
async def test_목록_문서가_없으면_빈_페이지(client, tenant_id):
    """문서가 하나도 없는 테넌트 — items·total·has_more의 기본값이 화면에서 깨지지 않아야 한다."""
    body = await list_docs(client)
    assert body == {'items': [], 'total': 0, 'has_more': False}


@pytest.mark.asyncio
async def test_목록_필터를_건_채로_페이징하면_total도_필터_기준(client, tenant_id, fake_queue, blob_tmp):
    """total이 필터를 안 거치면 '전체 6건'이라 써놓고 3건만 보이는 화면이 된다.

    조건을 한 곳에서 만들어 count와 페이지 쿼리가 함께 쓰는지를 보는 테스트다 —
    필터와 페이징을 따로 검증하면 이 어긋남이 안 잡힌다.
    """
    fid = await make_folder(client, '규정집')
    for i in range(3):                                   # 폴더 안 3건
        await upload(client, f'규정{i}.md', MD, extra_parts=doc_data(folder_id=fid))
    for i in range(3):                                   # 미분류 3건
        await upload(client, f'기타{i}.md', MD)

    first = await list_docs(client, folder_id=fid, limit=2, offset=0)
    assert first['total'] == 3                           # 6이 아니라 3 — 필터 기준
    assert len(first['items']) == 2 and first['has_more'] is True
    assert all(d['folder_id'] == fid for d in first['items'])

    second = await list_docs(client, folder_id=fid, limit=2, offset=2)
    assert second['total'] == 3 and len(second['items']) == 1
    assert second['has_more'] is False                    # 필터 기준으로 마지막 페이지

    # 검색어 + 페이징도 같은 규칙
    q_page = await list_docs(client, q='규정', limit=2, offset=2)
    assert q_page['total'] == 3 and len(q_page['items']) == 1 and q_page['has_more'] is False


# ── #181 응답 시각은 KST 오프셋 ──────────────────────────────

@pytest.mark.asyncio
async def test_uploaded_at은_KST_오프셋으로_나가고_시점은_같다(client, tenant_id, fake_queue, blob_tmp):
    from datetime import datetime
    body = await upload(client, '환불정책.md', MD)
    assert body['uploaded_at'].endswith('+09:00')
    # 응답 문자열을 다시 파싱하면 DB의 UTC 값과 같은 시점 — 변환이지 이동이 아니다
    assert datetime.fromisoformat(body['uploaded_at']) == (await get_doc(body['document_id'])).uploaded_at
    # 목록·exists도 같은 형식 (스키마 세 곳이 같은 타입을 쓴다)
    assert (await list_docs(client))['items'][0]['uploaded_at'].endswith('+09:00')
    ex = (await client.get('/kms/documents/exists', params={'filename': '환불정책.md'})).json()
    assert ex['uploaded_at'].endswith('+09:00')
