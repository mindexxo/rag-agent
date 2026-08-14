"""한글 파일명 유니코드 정규화 회귀 테스트 (#34).

이 버그는 **에러 없이 조용히 지표만 틀리는** 종류였다 — 답변에 인용 라벨이 멀쩡히 있고
검색도 정상인데 `cited_docs`만 빈 배열이 되어 FE 각주와 인용 통계가 0이 됐다.

eval의 `citation_accuracy`는 이걸 원리적으로 못 잡는다 — gold(NFC)와 LLM 출력(NFC)만
비교하고 **DB filename이 개입하지 않기** 때문이다. 그래서 Cite 지표가 만점인데 운영은
깨져 있었다. 이 파일이 그 구멍을 메운다: **NFD로 업로드해 DB를 거친 뒤** 인용이 잡히는지 본다.
"""
import unicodedata

import pytest
from sqlalchemy import select

from database import AsyncSessionLocal
from rag.documents import index_pending_document
from rag.models import Document, Message
from tests.conftest import sse_meta

# 분해형(NFD) 한글 파일명 — macOS 브라우저 업로드가 주는 형태.
# 소스 파일 자체는 NFC로 저장되므로 리터럴을 쓰지 않고 런타임에 분해한다.
FILENAME_NFC = '환불반품정책.md'
FILENAME_NFD = unicodedata.normalize('NFD', FILENAME_NFC)

MD = '# 환불 정책\n\n## 1. 기간\n\n단순변심 반품은 14일 이내 신청한다.\n'.encode()


def test_전제_두_정규형은_다른_문자열이다():
    """이 테스트 파일의 의미가 성립하는 전제 — 같아 보이지만 다른 값이다."""
    assert FILENAME_NFD != FILENAME_NFC
    assert len(FILENAME_NFD) > len(FILENAME_NFC)
    assert unicodedata.normalize('NFC', FILENAME_NFD) == FILENAME_NFC


async def _upload(client, filename: str) -> dict:
    res = await client.post('/kms/documents', files={'file': (filename, MD, 'text/markdown')})
    assert res.status_code == 200, res.text
    return res.json()


@pytest.mark.asyncio
async def test_NFD_파일명_업로드는_NFC로_저장된다(client, tenant_id, fake_queue, blob_tmp):
    """경계 정규화 — 분해형으로 올려도 DB에는 조합형만 들어간다."""
    body = await _upload(client, FILENAME_NFD)
    assert body['filename'] == FILENAME_NFC          # 응답도 정규화된 이름

    async with AsyncSessionLocal() as session:
        doc = await session.get(Document, body['document_id'])
        assert doc.filename == FILENAME_NFC
        assert unicodedata.is_normalized('NFC', doc.filename)


@pytest.mark.asyncio
async def test_NFD로_올린_문서도_인용이_cited_docs에_잡힌다(
        client, tenant_id, fake_llm, pass_gate, fake_queue, blob_tmp):
    """#34의 본질 — 업로드부터 DB를 거쳐 인용 매칭까지 관통 검증.

    LLM은 프롬프트의 라벨을 NFC로 출력한다(실측). 저장값이 NFD로 남아 있으면
    cited_filenames의 부분문자열 검색이 실패해 cited_docs가 빈 배열이 된다.
    """
    body = await _upload(client, FILENAME_NFD)
    await index_pending_document(body['document_id'])

    # 모델이 NFC 라벨로 인용하는 상황을 재현 (저장값이 NFD였다면 여기서 매칭이 깨진다)
    fake_llm.answer = f'단순변심 반품은 14일 이내입니다. [{FILENAME_NFC} v1]'

    res = await client.post('/kms/query', json={'query': '반품 기간 알려줘'})
    assert res.status_code == 200
    assistant_id = sse_meta(res)['assistant_message_id']

    async with AsyncSessionLocal() as session:
        msg = await session.get(Message, assistant_id)
        assert msg.status == 'done'
        assert msg.cited_docs == [FILENAME_NFC], (
            f'인용이 잡히지 않았다 — cited_docs={msg.cited_docs!r}. '
            'FE 각주와 stats 인용률이 조용히 0이 되는 회귀다 (#34).'
        )


@pytest.mark.asyncio
async def test_모델이_NFD로_인용해도_잡힌다(
        client, tenant_id, fake_llm, pass_gate, fake_queue, blob_tmp):
    """방어층 — LLM 출력은 통제 불가 입력이라 어느 정규형으로 오든 견뎌야 한다."""
    body = await _upload(client, FILENAME_NFC)
    await index_pending_document(body['document_id'])
    fake_llm.answer = f'14일 이내입니다. [{FILENAME_NFD} v1]'   # 모델이 분해형으로 냈다고 가정

    res = await client.post('/kms/query', json={'query': '반품 기간 알려줘'})
    assistant_id = sse_meta(res)['assistant_message_id']

    async with AsyncSessionLocal() as session:
        msg = await session.get(Message, assistant_id)
        assert msg.cited_docs == [FILENAME_NFC]


@pytest.mark.asyncio
async def test_NFD와_NFC_동명_재업로드는_새_버전이_된다(
        client, tenant_id, fake_queue, blob_tmp):
    """UNIQUE(tenant_id, filename, version)가 코드포인트 단위라, 정규화가 없으면 두 정규형이
    별개 문서로 통과해 시각적으로 같은 이름의 문서가 조용히 2건 생긴다."""
    first = await _upload(client, FILENAME_NFD)
    second = await _upload(client, FILENAME_NFC)     # 같은 이름의 재업로드로 취급돼야 한다

    assert second['version'] == first['version'] + 1

    async with AsyncSessionLocal() as session:
        names = (await session.execute(
            select(Document.filename).where(Document.tenant_id == tenant_id)
        )).scalars().all()
    assert set(names) == {FILENAME_NFC}, f'정규형이 섞여 문서가 갈렸다: {names!r}'


@pytest.mark.asyncio
async def test_exists_조회는_정규형과_무관하게_같은_문서를_찾는다(
        client, tenant_id, fake_queue, blob_tmp):
    """exists는 '업로드 판정과 정확히 같아야' 하는 API — 여기가 어긋나면 FE가 중복 확인창을
    띄우지 않고 expect_version=0으로 올려 기존 문서를 별개로 만든다."""
    await _upload(client, FILENAME_NFC)

    for probe in (FILENAME_NFC, FILENAME_NFD):
        res = await client.get('/kms/documents/exists', params={'filename': probe})
        assert res.status_code == 200
        assert res.json()['exists'] is True, f'{probe!r}로 조회했을 때 못 찾았다'


@pytest.mark.asyncio
async def test_채팅_첨부_파일명도_NFC로_정규화된다(client, tenant_id):
    """`[첨부: 파일명]` 라벨이 프롬프트에 들어가므로 문서 파일명과 같은 위험을 갖는다.
    스키마 validator라 extract 응답과 /kms/query 바디 양쪽이 함께 덮인다."""
    res = await client.post(
        '/kms/attachments/extract',
        files={'file': (FILENAME_NFD, MD, 'text/markdown')},
    )
    assert res.status_code == 200, res.text
    assert res.json()['filename'] == FILENAME_NFC


@pytest.mark.asyncio
async def test_query_바디의_첨부_파일명도_정규화된다(client, tenant_id, fake_llm):
    """스키마 validator를 두는 근거가 이 경로다 — FE는 extract 결과를 매 턴 /kms/query에
    재전송하므로, 실사용에서 반복 통과하는 건 extract가 아니라 여기다.
    라벨(`[첨부: 파일명]`)이 프롬프트에 들어가는 지점이라 문서 파일명과 같은 위험을 갖는다."""
    res = await client.post('/kms/query', json={
        'query': '첨부 요약해줘',
        'attachments': [{'filename': FILENAME_NFD, 'text': '단순변심 반품은 14일 이내.'}],
    })
    assert res.status_code == 200
    assistant_id = sse_meta(res)['assistant_message_id']

    # 프롬프트에 주입된 첨부 라벨이 정규화된 이름인지 (LLM이 실제로 받은 것)
    generate_prompts = [s for kind, s in fake_llm.system_prompts if kind == 'generate']
    assert generate_prompts, '생성 프롬프트가 없다 — 경로가 바뀌었다'

    async with AsyncSessionLocal() as session:
        user_msg = (await session.execute(
            select(Message).where(Message.tenant_id == tenant_id)
            .where(Message.role == 'user').order_by(Message.id.desc())
        )).scalars().first()
        assert user_msg.attachments[0]['filename'] == FILENAME_NFC
    assert assistant_id is not None


@pytest.mark.asyncio
async def test_CLI_인제스트도_NFC로_저장된다(tenant_id, fake_embed, tmp_path):
    """CLI(`ingest_file`)는 파일시스템 이름을 직접 쓰는 4번째 경계다. 웹 업로드와 형태가
    갈리면 같은 문서가 CLI/웹에서 별개로 취급돼 supersede·인용 매칭이 어긋난다."""
    from rag.ingestion import ingest_file

    src = tmp_path / FILENAME_NFD
    src.write_bytes(MD)
    doc_id = await ingest_file(src, tenant_id)

    async with AsyncSessionLocal() as session:
        doc = await session.get(Document, doc_id)
        assert doc.filename == FILENAME_NFC
        assert unicodedata.is_normalized('NFC', doc.filename)
