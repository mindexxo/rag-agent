"""문서 속성 — 등록일시·등록자(#164), 업로드 시 폴더 지정(#165), 일괄 변경(#166).

"누가 언제 올렸고 어느 폴더에 있고 검색에 쓰이는가"를 다룬다. 폴더 소속과 참조 on/off는
단건 PATCH와 일괄 PATCH가 같은 규칙을 써야 하므로 한 파일에 둔다.
공용 헬퍼는 tests/helpers_documents.py.
"""
import json
from datetime import timedelta

import pytest
from sqlalchemy import delete as sql_delete, func, select

from database import AsyncSessionLocal
from rag import cache, outbox
from rag.models import AnswerCache as AnswerCacheRow, Document, SearchIndexOutbox
from tests.conftest import ingest
from tests.helpers_documents import MD, doc_data, get_doc, make_folder, upload


# ── #164 등록일시·등록자 ──────────────────────────────────────

@pytest.mark.asyncio
async def test_목록은_등록일시_내림차순이고_등록자를_싣는다(client, tenant_id, fake_queue, blob_tmp):
    """목록 API의 기본 정렬 키는 uploaded_at이다 (#164).

    정렬 키가 정말 uploaded_at인지 보려면 id 순서와 어긋나게 만들어야 한다 —
    그냥 두면 id.desc()와 결과가 같아 무엇으로 정렬했는지 구분되지 않는다.
    """
    first = await upload(client, '가정책.md', MD, headers={'X-User-Id': 'agent-a'})
    second = await upload(client, '나정책.md', MD, headers={'X-User-Id': 'agent-b'})

    items = (await client.get('/kms/documents')).json()['items']
    assert [d['document_id'] for d in items] == [second['document_id'], first['document_id']]
    assert items[0]['uploaded_by'] == 'agent-b'
    assert items[0]['uploaded_at']                      # datetime 직렬화 확인

    # 먼저 올린 문서의 등록일시만 미래로 옮긴다 → id 순서와 반대가 되어야 한다.
    # (aware 값을 그대로 대입한다 — 시각 컬럼 매핑이 TIMESTAMPTZ에 맞춰져 있다, rag/models.py의 Base)
    async with AsyncSessionLocal() as s:
        doc = await s.get(Document, first['document_id'])
        doc.uploaded_at = doc.uploaded_at + timedelta(days=1)
        await s.commit()

    items = (await client.get('/kms/documents')).json()['items']
    assert [d['document_id'] for d in items] == [first['document_id'], second['document_id']]


@pytest.mark.asyncio
async def test_등록자는_X_User_Id로_기록되고_미전송이면_null(client, tenant_id, fake_queue, blob_tmp):
    """#164: 헤더가 없으면 NULL로 남긴다 — conversation의 'test-user' 폴백을 쓰지 않는다.
    그 가짜 값이 문서 관리 화면의 등록자 칸에 그대로 뜨기 때문."""
    with_user = await upload(client, '가정책.md', MD, headers={'X-User-Id': 'agent-a'})
    without = await upload(client, '나정책.md', MD)

    assert with_user['uploaded_by'] == 'agent-a'
    assert without['uploaded_by'] is None
    assert (await get_doc(with_user['document_id'])).uploaded_by == 'agent-a'
    assert (await get_doc(without['document_id'])).uploaded_by is None


@pytest.mark.asyncio
async def test_재업로드_등록자는_계승하지_않는다(client, tenant_id, fake_queue, blob_tmp):
    """#164: 문서에 건 설정(is_searchable)은 개정판에 계승되지만, 등록자는 그 버전을 올린
    사람으로 갱신된다 — 계승하면 최신 개정을 누가 했는지 화면에서 알 수 없다."""
    v1 = await upload(client, '환불정책.md', MD, headers={'X-User-Id': 'agent-a'})
    await ingest(v1['document_id'])
    assert (await client.patch(f"/kms/documents/{v1['document_id']}",
                               json={'is_searchable': False})).status_code == 200

    v2 = await upload(client, '환불정책.md', MD, headers={'X-User-Id': 'agent-b'})
    assert v2['version'] == 2
    assert v2['is_searchable'] is False       # 설정은 계승
    assert v2['uploaded_by'] == 'agent-b'     # 등록자는 갱신

    # 헤더 없이 올린 버전이 직전 등록자를 물려받지 않는지 — "계승하지 않는다"가 깨지는
    # 가장 흔한 형태(prev 폴백)를 직접 찍는다.
    v3 = await upload(client, '환불정책.md', MD)
    assert v3['version'] == 3
    assert v3['uploaded_by'] is None


# ── #165 업로드 시 폴더 지정 (document-data 파트) ─────────────

@pytest.mark.asyncio
async def test_업로드_시_폴더_지정(client, tenant_id, fake_queue, blob_tmp):
    fid = await make_folder(client, '규정집')
    body = await upload(client, '환불정책.md', MD, extra_parts=doc_data(folder_id=fid))
    assert body['folder_id'] == fid
    assert (await get_doc(body['document_id'])).folder_id == fid


@pytest.mark.asyncio
async def test_document_data를_문자열_파트로_보내도_동작(client, tenant_id, fake_queue, blob_tmp):
    """브라우저는 Blob(=UploadFile)로, 다른 클라이언트는 문자열 필드로 보낼 수 있다 — 둘 다 받는다."""
    fid = await make_folder(client, '규정집')
    body = await upload(client, '환불정책.md', MD,
                         extra_parts={'document-data': (None, json.dumps({'folder_id': fid}))})
    assert body['folder_id'] == fid


@pytest.mark.asyncio
async def test_folder_id_null이면_미분류로_뗀다(client, tenant_id, fake_queue, blob_tmp):
    """null 전송 = 미분류. 값만 보면 '안 보냄'과 같아서, 보냈는지 여부로 갈린다."""
    fid = await make_folder(client, '규정집')
    v1 = await upload(client, '환불정책.md', MD, extra_parts=doc_data(folder_id=fid))
    await ingest(v1['document_id'])

    v2 = await upload(client, '환불정책.md', MD, extra_parts=doc_data(folder_id=None))
    assert v2['version'] == 2
    assert v2['folder_id'] is None


@pytest.mark.asyncio
async def test_folder_id_미전송이면_직전_버전에서_계승(client, tenant_id, fake_queue, blob_tmp):
    fid = await make_folder(client, '규정집')
    v1 = await upload(client, '환불정책.md', MD, extra_parts=doc_data(folder_id=fid))
    await ingest(v1['document_id'])

    # document-data는 보내되 folder_id 키만 없다 → 계승
    v2 = await upload(client, '환불정책.md', MD, extra_parts=doc_data(description='표 설명'))
    assert v2['folder_id'] == fid
    assert (await get_doc(v2['document_id'])).description == '표 설명'   # JSON 경로도 값이 산다
    # 파트 자체를 안 보내도 계승 (옛 방식 호출)
    v3 = await upload(client, '환불정책.md', MD)
    assert v3['folder_id'] == fid


@pytest.mark.asyncio
async def test_없는_폴더는_404_문서도_blob도_안_남는다(client, tenant_id, fake_queue, blob_tmp):
    res = await client.post('/kms/documents',
                            files={'file': ('환불정책.md', MD, 'text/markdown'),
                                   **doc_data(folder_id=999999)})
    assert res.status_code == 404
    async with AsyncSessionLocal() as s:
        cnt = (await s.execute(select(func.count()).select_from(Document)
                               .where(Document.tenant_id == tenant_id))).scalar()
    assert cnt == 0
    assert not (blob_tmp / tenant_id).exists() or not list((blob_tmp / tenant_id).iterdir())


@pytest.mark.asyncio
async def test_다른_테넌트_폴더는_404(client, tenant_id, other_tenant_id, fake_queue, blob_tmp):
    """격리 — 남의 폴더 id를 넣어도 통과하면 안 된다."""
    res = await client.post('/kms/folders', json={'name': '남의폴더'},
                            headers={'X-Tenant-Id': other_tenant_id})
    other_fid = res.json()['id']

    res = await client.post('/kms/documents',
                            files={'file': ('환불정책.md', MD, 'text/markdown'),
                                   **doc_data(folder_id=other_fid)})
    assert res.status_code == 404


@pytest.mark.asyncio
async def test_document_data와_평면필드_동시_전송은_400(client, tenant_id, fake_queue, blob_tmp):
    """한쪽을 조용히 무시하면 '보냈는데 안 먹는' 버그가 된다 — 명시 거절."""
    res = await client.post('/kms/documents',
                            files={'file': ('환불정책.md', MD, 'text/markdown'),
                                   **doc_data(description='새 방식')},
                            data={'description': '옛 방식'})
    assert res.status_code == 400


@pytest.mark.asyncio
async def test_잘못된_document_data는_422(client, tenant_id, fake_queue, blob_tmp):
    for bad in ('{깨진 json', json.dumps({'folder_id': 'abc'}), json.dumps({'오타필드': 1})):
        res = await client.post('/kms/documents',
                                files={'file': ('환불정책.md', MD, 'text/markdown'),
                                       'document-data': ('blob', bad, 'application/json')})
        assert res.status_code == 422, f'{bad} → {res.status_code}'


@pytest.mark.asyncio
async def test_옛_평면필드_방식은_그대로_동작(client, tenant_id, fake_queue, blob_tmp):
    """FE 전환 전 호출 호환 — description·expect_version 평면 필드."""
    res = await client.post('/kms/documents',
                            files={'file': ('환불정책.md', MD, 'text/markdown')},
                            data={'description': '표 설명', 'expect_version': '0'})
    assert res.status_code == 200, res.text
    assert (await get_doc(res.json()['document_id'])).description == '표 설명'


@pytest.mark.asyncio
async def test_신규_문서에_폴더를_안_주면_미분류(client, tenant_id, fake_queue, blob_tmp):
    """계승할 직전 버전이 없을 때의 기본값 — 파트를 안 보내도, 보내되 folder_id만 빼도 미분류."""
    a = await upload(client, '환불정책.md', MD)
    b = await upload(client, '배송정책.md', MD, extra_parts=doc_data(description='표 설명'))
    assert a['folder_id'] is None and b['folder_id'] is None


# ── #166 문서 일괄 변경 (PATCH /kms/documents) ────────────────

async def _bulk(client, **body):
    return await client.patch('/kms/documents', json=body)


@pytest.mark.asyncio
async def test_일괄_폴더이동과_참조토글(client, tenant_id, fake_queue, blob_tmp):
    fid = await make_folder(client, '규정집')
    a = await upload(client, '환불정책.md', MD)
    b = await upload(client, '배송정책.md', MD)

    res = await _bulk(client, document_ids=[a['document_id'], b['document_id']],
                      folder_id=fid, is_searchable=False)
    assert res.status_code == 200, res.text
    body = res.json()
    assert {d['document_id'] for d in body} == {a['document_id'], b['document_id']}
    assert all(d['folder_id'] == fid and d['is_searchable'] is False for d in body)
    assert all(d['ref_count'] is None for d in body)      # 목록 API 전용 집계
    for d in (a, b):
        doc = await get_doc(d['document_id'])
        assert doc.folder_id == fid and doc.is_searchable is False


@pytest.mark.asyncio
async def test_일괄_folder_id_null이면_미분류(client, tenant_id, fake_queue, blob_tmp):
    fid = await make_folder(client, '규정집')
    a = await upload(client, '환불정책.md', MD, extra_parts=doc_data(folder_id=fid))
    res = await _bulk(client, document_ids=[a['document_id']], folder_id=None)
    assert res.status_code == 200, res.text
    assert res.json()[0]['folder_id'] is None


@pytest.mark.asyncio
async def test_일괄_대상이_하나라도_어긋나면_아무것도_안_바뀐다(
        client, tenant_id, other_tenant_id, fake_queue, blob_tmp):
    """없는 id·다른 테넌트 id·삭제된 문서 — 셋 다 전체 거절이고, 성공분도 반영되지 않는다."""
    fid = await make_folder(client, '규정집')
    mine = await upload(client, '환불정책.md', MD)

    # 다른 테넌트의 문서
    other = await client.post('/kms/documents', files={'file': ('남의문서.md', MD, 'text/markdown')},
                              headers={'X-Tenant-Id': other_tenant_id})
    # 삭제된 문서
    gone = await upload(client, '지울문서.md', MD)
    assert (await client.delete(f"/kms/documents/{gone['document_id']}")).status_code == 204

    for label, bad in (('없는 id', 999999),
                       ('다른 테넌트', other.json()['document_id']),
                       ('삭제된 문서', gone['document_id'])):
        res = await _bulk(client, document_ids=[mine['document_id'], bad], folder_id=fid)
        assert res.status_code == 404, f'{label} → {res.status_code}'
        assert (await get_doc(mine['document_id'])).folder_id is None, f'{label}: 성공분이 반영됐다'


@pytest.mark.asyncio
async def test_일괄_실효참조가_꺼지는_문서만_캐시_무효화(client, tenant_id, fake_queue, blob_tmp):
    """문서마다 현재 폴더·스위치가 달라 on→off 전이도 제각각이다 — 일괄 판정하면 안 된다.

    off 폴더에 있던 문서는 이미 실효 off라 지울 캐시가 없고, on이던 문서만 무효화 대상이다.
    """
    off_folder = await make_folder(client, '대외비')
    assert (await client.patch(f'/kms/folders/{off_folder}',
                               json={'is_searchable': False})).status_code == 200

    on_doc = await upload(client, '환불정책.md', MD)                                   # 미분류 = 실효 on
    off_doc = await upload(client, '배송정책.md', MD, extra_parts=doc_data(folder_id=off_folder))

    async with AsyncSessionLocal() as s:
        await cache.save_answer(s, tenant_id, '질의 하나', '답1', [], [on_doc['document_id']])
        await cache.save_answer(s, tenant_id, '질의 둘', '답2', [], [off_doc['document_id']])
        await s.commit()

    # 둘 다 문서 스위치를 끈다 — 실효 참조가 실제로 on→off로 바뀌는 건 on_doc뿐이다
    res = await _bulk(client, document_ids=[on_doc['document_id'], off_doc['document_id']],
                      is_searchable=False)
    assert res.status_code == 200, res.text

    async with AsyncSessionLocal() as s:
        remain = (await s.execute(select(AnswerCacheRow.answer)
                                  .where(AnswerCacheRow.tenant_id == tenant_id))).scalars().all()
    assert remain == ['답2'], '이미 off였던 문서의 캐시까지 지웠거나, on이던 문서 캐시가 남았다'


@pytest.mark.asyncio
async def test_일괄_색인_갱신은_한_행에_담긴다(client, tenant_id, fake_queue, blob_tmp):
    a = await upload(client, '환불정책.md', MD)
    b = await upload(client, '배송정책.md', MD)
    ids = [a['document_id'], b['document_id']]
    async with AsyncSessionLocal() as s:      # 업로드가 만든 INDEX_DOCUMENT 행을 비우고 본다
        await s.execute(sql_delete(SearchIndexOutbox).where(SearchIndexOutbox.tenant_id == tenant_id))
        await s.commit()

    assert (await _bulk(client, document_ids=ids, is_searchable=False)).status_code == 200

    async with AsyncSessionLocal() as s:
        rows = (await s.execute(select(SearchIndexOutbox)
                                .where(SearchIndexOutbox.tenant_id == tenant_id))).scalars().all()
    assert len(rows) == 1 and rows[0].op == outbox.META_DOCUMENTS
    assert sorted(rows[0].payload['document_ids']) == sorted(ids)


@pytest.mark.asyncio
async def test_일괄_상한_초과는_422(client, tenant_id, fake_queue, blob_tmp):
    res = await _bulk(client, document_ids=list(range(1, 202)), is_searchable=False)
    assert res.status_code == 422


@pytest.mark.asyncio
async def test_일괄_바꿀_것이_없으면_무동작(client, tenant_id, fake_queue, blob_tmp):
    """빈 배열이든 바꿀 필드가 없든 빈 목록 200 — 이 경로에선 id 유효성도 따지지 않는다."""
    assert (await _bulk(client, document_ids=[], folder_id=None)).json() == []
    assert (await _bulk(client, document_ids=[999999])).json() == []


@pytest.mark.asyncio
async def test_삭제된_문서는_단건_PATCH도_404(client, tenant_id, fake_queue, blob_tmp):
    """#166: 목록에 뜨지도 않는 죽은 행의 컬럼이 조용히 바뀌던 것을 막는다."""
    doc = await upload(client, '환불정책.md', MD)
    assert (await client.delete(f"/kms/documents/{doc['document_id']}")).status_code == 204
    res = await client.patch(f"/kms/documents/{doc['document_id']}", json={'is_searchable': False})
    assert res.status_code == 404


@pytest.mark.asyncio
async def test_일괄_폴더가_없거나_남의_것이면_404(client, tenant_id, other_tenant_id,
                                                fake_queue, blob_tmp):
    """대상 문서는 멀쩡해도 옮길 폴더가 어긋나면 전체 거절 — 문서도 안 바뀐다."""
    doc = await upload(client, '환불정책.md', MD)
    other_folder = (await client.post('/kms/folders', json={'name': '남의폴더'},
                                      headers={'X-Tenant-Id': other_tenant_id})).json()['id']

    for label, fid in (('없는 폴더', 999999), ('다른 테넌트 폴더', other_folder)):
        res = await _bulk(client, document_ids=[doc['document_id']], folder_id=fid)
        assert res.status_code == 404, f'{label} → {res.status_code}'
    assert (await get_doc(doc['document_id'])).folder_id is None


@pytest.mark.asyncio
async def test_일괄_상한_경계(client, tenant_id, fake_queue, blob_tmp):
    """200건은 스키마를 통과하고(대상이 없어 404), 201건은 스키마에서 422.

    유효한 문서 200건을 실제로 만들지 않고 경계만 본다 — 통과 여부는 대상 조회 이전에
    갈리므로 404가 나오면 스키마를 지난 것이다.
    """
    assert (await _bulk(client, document_ids=list(range(1, 201)),
                        is_searchable=False)).status_code == 404
    assert (await _bulk(client, document_ids=list(range(1, 202)),
                        is_searchable=False)).status_code == 422


@pytest.mark.asyncio
async def test_일괄_404는_어긋난_id를_알려준다(client, tenant_id, fake_queue, blob_tmp):
    """화면이 '무엇이 문제인지' 보여줄 수 있어야 한다 — 어느 id가 걸렸는지 응답에 싣는다."""
    doc = await upload(client, '환불정책.md', MD)
    res = await _bulk(client, document_ids=[doc['document_id'], 999999], is_searchable=False)
    assert res.status_code == 404
    assert '999999' in res.json()['detail']
    assert str(doc['document_id']) not in res.json()['detail']   # 멀쩡한 id는 안 싣는다
