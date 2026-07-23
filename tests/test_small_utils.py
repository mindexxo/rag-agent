"""소형 유틸 단위 테스트 — estimate_tokens / _detect_mime / _sha256."""
import hashlib
from pathlib import Path

from rag.ingestion import _detect_mime, _sha256
from rag.tokens import estimate_tokens


class TestEstimateTokens:
    def test_빈_문자열도_최소_1(self):
        assert estimate_tokens('') == 1

    def test_한국어_과대추정_방향(self):
        # 1.4로 나눔 — 실제(~1.5자/토큰)보다 크게 잡아 예산 초과를 막는 방향
        assert estimate_tokens('가' * 14) == 11

    def test_길이에_단조증가(self):
        assert estimate_tokens('가' * 100) > estimate_tokens('가' * 50)


class TestDetectMime:
    def test_지원_확장자_전부_판별됨(self):
        # SUPPORTED_SUFFIXES 전체가 이 환경에서 mime 판별 가능해야 업로드가 500 없이 동작
        # (리뷰 지적: 환경 따라 guess_type 실패 가능 — 실패하면 이 테스트가 그 환경임을 알려줌)
        for name, expected_part in [
            ('a.pdf', 'pdf'), ('a.docx', 'wordprocessingml'), ('a.xlsx', 'sheet'),
            ('a.txt', 'text'), ('a.md', 'markdown'),
        ]:
            mime = _detect_mime(Path(name))
            assert expected_part in mime, f'{name} -> {mime}'

    def test_override_경로_대문자_확장자(self, monkeypatch):
        # 이 환경 guess_type은 docx를 알아서 override 분기가 가려짐 — guess_type을 죽여 분기를 직접 핀
        import mimetypes
        monkeypatch.setattr(mimetypes, 'guess_type', lambda *_: (None, None))
        assert 'wordprocessingml' in _detect_mime(Path('a.DOCX'))  # override + suffix.lower() 둘 다 통과해야 성공

    def test_미지원_확장자는_ValueError(self):
        # 조용한 None 반환으로 바뀌면 업로드가 500으로 새는 계약 위반
        import pytest
        with pytest.raises(ValueError):
            _detect_mime(Path('a.zzz'))


class TestSha256:
    def test_같은_내용_같은_해시(self, tmp_path):
        a, b = tmp_path / 'a.txt', tmp_path / 'b.txt'
        a.write_bytes(b'content')
        b.write_bytes(b'content')
        assert _sha256(a) == _sha256(b) == hashlib.sha256(b'content').hexdigest()

    def test_다른_내용_다른_해시(self, tmp_path):
        a, b = tmp_path / 'a.txt', tmp_path / 'b.txt'
        a.write_bytes(b'content1')
        b.write_bytes(b'content2')
        assert _sha256(a) != _sha256(b)

    def test_블록_크기_초과_파일(self, tmp_path):
        # 64KB 블록 루프가 첫 블록만 읽도록 붕괴하면 dedupe 오판 — 실제 PDF는 대부분 64KB 초과
        content = b'x' * (64 * 1024) + b'tail'
        p = tmp_path / 'big.bin'
        p.write_bytes(content)
        assert _sha256(p) == hashlib.sha256(content).hexdigest()
