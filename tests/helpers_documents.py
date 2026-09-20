"""문서 관리 통합 테스트가 함께 쓰는 헬퍼 (#188 D).

test_integration_documents*.py 다섯 파일이 이 모듈을 가져다 쓴다. 전엔 한 파일(1,319줄) 안에
섹션별로 흩어져 있었는데, 정의된 섹션 밖에서도 불리는 것들이라(예: 폴더 생성은 #165에서 정의됐지만
#166·#176도 쓴다) 파일을 쪼개는 순간 어디에 둘지가 문제가 됐다.

**conftest에 넣지 않은 이유**: 문서 축 전용이라 전 테스트에 노출할 이유가 없다. 공용 픽스처
(client·tenant_id·ingest·indexed_chunk_texts)는 그대로 conftest가 정의점이다.
"""
import json

from database import AsyncSessionLocal
from rag.models import Document
from tests.conftest import indexed_chunk_texts

# 여러 파일이 쓰는 표준 본문. '14일'이 들어 있어 재업로드 테스트가 '30일' 본문과 대조할 수 있다.
MD = '# 환불 정책\n\n## 1. 기간\n\n단순변심 반품은 14일 이내 신청한다.\n'.encode()


async def upload(client, filename: str, content: bytes, mime='text/markdown', headers=None,
                 extra_parts=None) -> dict:
    files = {'file': (filename, content, mime), **(extra_parts or {})}
    res = await client.post('/kms/documents', files=files, headers=headers)
    assert res.status_code == 200, res.text
    return res.json()


async def post_upload(client, filename: str, content: bytes, expect_version=None):
    """상태 코드를 직접 보는 업로드 — 409·400을 기대하는 테스트용(upload는 200을 단언한다)."""
    data = {} if expect_version is None else {'expect_version': str(expect_version)}
    return await client.post('/kms/documents',
                             files={'file': (filename, content, 'text/markdown')},
                             data=data)


def doc_data(**fields) -> dict:
    """document-data 파트 (#165) — 브라우저가 Blob으로 실을 때와 같은 모양.
    filename이 붙어 서버에는 UploadFile로 도착한다."""
    return {'document-data': ('blob', json.dumps(fields), 'application/json')}


async def make_folder(client, name: str) -> int:
    res = await client.post('/kms/folders', json={'name': name})
    assert res.status_code == 200, res.text
    return res.json()['id']


async def list_docs(client, **params):
    res = await client.get('/kms/documents', params=params or None)
    assert res.status_code == 200, res.text
    return res.json()


async def get_doc(doc_id: int) -> Document:
    async with AsyncSessionLocal() as session:
        return await session.get(Document, doc_id)


async def chunk_texts(doc_id: int) -> list[str]:
    return await indexed_chunk_texts(doc_id)        # 청크는 색인에만 있다(#139)
