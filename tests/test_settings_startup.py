"""기동 시 설정 검증 — 상용 공급자인데 키·모델이 비면 Settings 생성이 실패한다(DB 없이 Pydantic만).

형식만 본다 — 벤더에 붙어 보지 않는다. 사내 vLLM·TEI·로컬 VLM은 키 없이 기동된다.
"""

import pytest

from config import Settings

LOCAL_VLM = 'http://localhost:18892/v1/chat/completions'
OPENAI_VLM = 'https://api.openai.com/v1/chat/completions'

REJECTED = [
    ('llm-empty', {'llm_provider': 'openai', 'llm_api_key': ''}, 'LLM_API_KEY'),
    ('llm-placeholder', {'llm_provider': 'anthropic', 'llm_api_key': 'EMPTY'}, 'LLM_API_KEY'),
    (
        'embed-key',
        {'embed_provider': 'openai', 'embed_model': 'text-embedding-3-large', 'embed_api_key': ''},
        'EMBED_API_KEY',
    ),
    ('embed-model', {'embed_provider': 'openai', 'embed_model': '', 'embed_api_key': 'k'}, 'EMBED_MODEL'),
    (
        'rerank-key',
        {'rerank_provider': 'cohere', 'rerank_model': 'rerank-v3.5', 'rerank_api_key': ''},
        'RERANK_API_KEY',
    ),
    ('vlm-commercial', {'vlm_caption_url': OPENAI_VLM, 'vlm_caption_api_key': ''}, 'VLM_CAPTION_API_KEY'),
]

ACCEPTED = [
    ('vllm-no-key', {'llm_provider': 'vllm', 'llm_api_key': 'EMPTY'}),
    ('commercial-with-key', {'llm_provider': 'openai', 'llm_api_key': 'k'}),
    (
        'tei-no-key',
        {
            'embed_provider': 'tei',
            'embed_model': '',
            'embed_api_key': '',
            'rerank_provider': 'tei',
            'rerank_api_key': '',
        },
    ),
    ('vendor-with-key', {'rerank_provider': 'cohere', 'rerank_model': 'rerank-v3.5', 'rerank_api_key': 'k'}),
    ('local-vlm', {'vlm_caption_url': LOCAL_VLM, 'vlm_caption_api_key': ''}),
    ('commercial-vlm-with-key', {'vlm_caption_url': OPENAI_VLM, 'vlm_caption_api_key': 'k'}),
]


@pytest.mark.parametrize(('_name', 'fields', 'hint'), REJECTED, ids=[c[0] for c in REJECTED])
def test_상용_공급자인데_키나_모델이_비면_기동을_거부한다(_name, fields, hint):
    with pytest.raises(ValueError, match=hint):
        Settings(**fields)


@pytest.mark.parametrize(('_name', 'fields'), ACCEPTED, ids=[c[0] for c in ACCEPTED])
def test_사내_공급자이거나_키가_있으면_기동된다(_name, fields):
    assert Settings(**fields) is not None
