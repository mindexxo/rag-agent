"""PG가 검색용 파생 컬럼을 안 드는 구성 (#139 전환 2단계) — 순수 테스트.

`config.pg_vector_columns=False`면 PG는 본문만 들고 벡터·어휘는 OpenSearch에만 있다.
그 구성에서 코드가 실제로 성립하는지를 값으로 고정한다. 스키마 마이그레이션(dense의
NOT NULL 해제·컬럼 DROP)은 실행하지 않았으므로 DB를 태우지 않고 계약만 검증한다.
"""
import pytest
from config import Settings
from rag import opensearch as O


class _Chunk:
    """색인 대상 청크의 최소 형태 — PG 행 대신. dense가 아예 없는 상태를 만들 수 있다."""

    def __init__(self, cid=1, text='반품은 7일 이내', dense=None, has_dense=True):
        self.id, self.text = cid, text
        self.tenant_id, self.document_id, self.faq_id = 't1', 10, None
        self.page, self.heading_path, self.meta = 3, ['2. 반품'], {}
        if has_dense:
            self.dense = dense


def _settings(**kw):
    return Settings(database_url='postgresql+asyncpg://x/y', **kw)


def test_전환은_두_단계다():
    """백엔드만 바꾸는 1단계에서는 컬럼이 그대로여야 한다 — 마이그레이션 강제 금지."""
    assert _settings(search_backend='opensearch').pg_vector_columns is True
    assert _settings(search_backend='opensearch', pg_vector_columns=False) is not None


def test_pg_백엔드에_컬럼없음_조합은_기동에서_막힌다():
    """조용히 빈 결과를 내는 설정이라 런타임이 아니라 기동에서 끊는다."""
    with pytest.raises(Exception) as e:
        _settings(search_backend='pg', pg_vector_columns=False)
    assert 'pg_vector_columns' in str(e.value)


def test_명시_벡터가_PG값을_이긴다():
    """인제스션이 방금 계산한 임베딩을 넘기면 PG를 경유하지 않는다."""
    doc = O.build_doc(_Chunk(dense=[0.1] * 1024), '정책.pdf', 2, dense=[0.9] * 1024)
    assert doc['dense'][0] == pytest.approx(0.9)


def test_PG에_벡터가_없어도_명시로_색인된다():
    """dense 컬럼이 DROP된 세계 — 속성 자체가 없다."""
    doc = O.build_doc(_Chunk(has_dense=False), '정책.pdf', 2, dense=[0.5] * 1024)
    assert len(doc['dense']) == 1024
    assert doc['chunk_id'] == 1 and doc['filename'] == '정책.pdf'


def test_벡터가_아예_없으면_크게_실패한다():
    """조용히 벡터 없는 문서를 색인하면 kNN에 안 잡히는 유령이 된다."""
    with pytest.raises(ValueError, match='벡터가 없다'):
        O.build_doc(_Chunk(has_dense=False), '정책.pdf', 2)


def test_어휘_필드는_PG_컬럼과_무관하다():
    """lex_tsv를 PG에서 버려도 OpenSearch 어휘 필드는 본문에서 다시 만들어진다 —
    그래서 어휘 컬럼도 함께 DROP할 수 있다."""
    from rag.lexical import bigrams
    ch = _Chunk(has_dense=False)
    doc = O.build_doc(ch, '정책.pdf', 2, dense=[0.5] * 1024)
    expected = O.lex_text(ch.text, '정책.pdf', ch.heading_path)
    assert doc[O.NORI_FIELD] == expected
    assert doc[O.LEX_BIGRAM_FIELD] == ' '.join(bigrams(expected))
