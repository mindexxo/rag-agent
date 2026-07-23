"""텍스트 인코딩 감지 단위 테스트 — chunking._read_text.

리팩토링에서 고친 P2(CP949 txt가 조용히 �로 깨지던 문제)를 고정한다.
"""
from rag.chunking import _read_text


def test_utf8_그대로_읽힘(tmp_path):
    p = tmp_path / 'a.txt'
    p.write_text('환불 규정 안내', encoding='utf-8')
    assert _read_text(p) == '환불 규정 안내'


def test_cp949_폴백(tmp_path):
    p = tmp_path / 'b.txt'
    p.write_bytes('배송비는 삼천원입니다'.encode('cp949'))
    assert _read_text(p) == '배송비는 삼천원입니다'


def test_utf8_우선순위_고정(tmp_path):
    # '주문 취소'의 utf-8 바이트는 cp949로도 디코드 가능(→ mojibake) — utf-8을 먼저 시도해야 함
    p = tmp_path / 'p.txt'
    p.write_bytes('주문 취소'.encode('utf-8'))
    assert _read_text(p) == '주문 취소'


def test_둘_다_아니면_replace로_예외없이(tmp_path):
    p = tmp_path / 'c.txt'
    p.write_bytes(b'\xff\xfe\x00 broken \x81\x40')  # utf-8도 cp949도 아닌 바이트
    result = _read_text(p)
    assert 'broken' in result        # 읽을 수 있는 부분은 보존
    assert '�' in result        # 깨진 바이트는 �로 표시 (replace — ignore로 바뀌면 조용한 소실)


def test_빈_파일(tmp_path):
    p = tmp_path / 'd.txt'
    p.write_bytes(b'')
    assert _read_text(p) == ''
