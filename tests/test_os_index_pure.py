"""OpenSearch 색인 문서 조립 계약 (#139, #146에서 모듈 분리에 맞춰 3파일로 나눔)
— DB·OpenSearch 없이 도는 순수 테스트.

  1. 어휘 입력 텍스트 조립이 임베딩 입력과 같다       (문서=프리픽스 / FAQ=원문)
  2. searchable 판정이 조건표대로다                   (문서 활성·ready·토글·폴더 / FAQ 활성)
  3. build_doc는 청크에 실린 벡터로 색인하고, 없으면 크게 실패한다
  4. chunk_os_id는 결정적이고 문서·청크 간 충돌이 없으며 20비트 상한을 강제한다
"""
import pytest

from rag import os_client as C
from rag import os_index as I
from rag import lexical
from rag.index_text import build_index_text


def test_어휘_입력_조립이_운영과_같다():
    """문서는 '파일명>헤딩' 프리픽스, FAQ는 원문 — 인제스션·운영 어휘 채널과 동일 비대칭."""
    text, fn, hp = "반품은 7일 이내", "환불반품정책.pdf", ["2. 반품", "2.1 기한"]
    assert I.lex_text(text, fn, hp) == build_index_text(text, fn, hp)
    assert I.lex_text(text, None, None) == text          # FAQ — 프리픽스 없음
    # 폴더 설명은 들어가지 않는다 (리랭커 전용, rag/index_text.py docstring)
    assert "폴더" not in I.lex_text(text, fn, hp)


def test_searchable_판정이_조건표대로다():
    """검색 가능 판정의 정의점(effective_searchable) — 문서: 활성+ready+검색토글+(미분류 or 폴더 on),
    FAQ: 활성. 색인 시 계산해 넣는 플래그라 조건이 바뀌면 전량 META 갱신이 따라야 한다."""
    ok = dict(is_faq=False, doc_is_active=True, doc_status='ready', doc_is_searchable=True)
    assert I.effective_searchable(**ok, folder_is_searchable=None) is True   # 미분류 문서
    assert I.effective_searchable(**ok, folder_is_searchable=True) is True
    assert I.effective_searchable(**ok, folder_is_searchable=False) is False  # 폴더 off
    assert I.effective_searchable(**{**ok, 'doc_is_active': False}) is False
    assert I.effective_searchable(**{**ok, 'doc_status': 'deleted'}) is False
    assert I.effective_searchable(**{**ok, 'doc_is_searchable': False}) is False
    assert I.effective_searchable(is_faq=True, faq_is_active=True) is True
    assert I.effective_searchable(is_faq=True, faq_is_active=False) is False


class _Chunk:
    """색인 대상 청크의 최소 형태(_ParsedChunk 모양). dense 속성이 없는 상태도 만들 수 있다."""

    def __init__(self, cid=1, text='반품은 7일 이내', dense=None, has_dense=True):
        self.id, self.text = cid, text
        self.tenant_id, self.document_id, self.faq_id = 't1', 10, None
        self.page, self.heading_path, self.meta = 3, ['2. 반품'], {}
        if has_dense:
            self.dense = dense


def test_벡터는_청크에_실려_온_값으로_색인된다():
    doc = I.build_doc(_Chunk(dense=[0.5] * 1024), '정책.pdf', 2)
    assert len(doc['dense']) == 1024 and doc['dense'][0] == pytest.approx(0.5)
    assert doc['chunk_id'] == 1 and doc['filename'] == '정책.pdf' and doc['version'] == 2


def test_벡터가_없으면_크게_실패한다():
    """조용히 벡터 없는 문서를 색인하면 kNN에 안 잡히는 유령이 된다."""
    with pytest.raises(ValueError, match='벡터가 없다'):
        I.build_doc(_Chunk(has_dense=False), '정책.pdf', 2)
    with pytest.raises(ValueError, match='벡터가 없다'):
        I.build_doc(_Chunk(dense=None), '정책.pdf', 2)


def test_어휘_필드는_본문에서_만든다():
    """Nori·bigram 필드는 lex_text(본문+프리픽스)에서 색인 시점에 조립된다."""
    ch = _Chunk(dense=[0.5] * 1024)
    doc = I.build_doc(ch, '정책.pdf', 2)
    expected = I.lex_text(ch.text, '정책.pdf', ch.heading_path)
    assert doc[C.NORI_FIELD] == expected
    assert doc[C.LEX_BIGRAM_FIELD] == ' '.join(lexical.bigrams(expected))


def test_chunk_os_id는_결정적이고_충돌이_없다():
    """PG 시퀀스 없이 (부모 id, chunk_index)로 만드는 id — 재색인이 같은 _id로 덮여 멱등하려면
    결정적이어야 하고, 문서·청크 조합마다 달라야 한다."""
    f = I.chunk_os_id
    assert f(document_id=1008, chunk_index=3) == f(document_id=1008, chunk_index=3)
    ids = {f(document_id=d, chunk_index=i) for d in (1, 2, 1008, 99999) for i in (0, 1, 7, 500)}
    assert len(ids) == 16
    # 문서 id 1의 첫 청크조차 2^20 이상 — 소규모 PG BIGSERIAL 청크 id와 겹치지 않는다
    assert f(document_id=1, chunk_index=0) == 1 << 20
    # FAQ는 음수 네임스페이스 — 문서 id 공간과 절대 겹치지 않는다(캐시의 -faq_id 관례와 같은 방식)
    assert f(faq_id=42) < 0 and f(faq_id=42) == -(42 << 20)
    assert {f(faq_id=q) for q in (1, 2, 42)}.isdisjoint(ids)


def test_chunk_os_id는_20비트_상한을_강제한다():
    """상한을 넘으면 상위 비트(문서 id 몫)를 침범해 다른 문서의 청크와 같은 id가 된다 —
    조용히 덮어쓰지 않고 크게 실패해야 한다."""
    assert I.chunk_os_id(document_id=5, chunk_index=(1 << 20) - 1) > 0
    with pytest.raises(ValueError, match='20비트'):
        I.chunk_os_id(document_id=5, chunk_index=1 << 20)
