"""출처 꼬리 분리·해석 단위 테스트 (#56) — rag/citation_tail.

TailSplitter는 vLLM 토큰 경계가 임의라는 전제에서 동작해야 한다 — 마커가 청크 경계에서
쪼개져 도착하는 경우를 1글자 단위 feed로 재현한다. 핵심 불변식: **feed·finish 반환값의
누적 == prose == 화면에 보인 것 == 저장되는 것**, 그리고 꼬리·잘린 버퍼는 그 어디에도 없다.
"""
import pytest

from rag.citation_labels import TAIL_END, TAIL_START, attachment_label, source_label
from rag.citation_tail import MAX_TAIL_CHARS, TailSplitter, resolve_citations
from schemas.kms import SourceCitation

SRC = [SourceCitation(document_id=5, filename='환불규정.docx', version=2),
       SourceCitation(document_id=None, filename='FAQ', version=1)]


def _run(chunks: list[str]) -> tuple[str, TailSplitter]:
    """feed 반환값 누적(=화면에 보인 것)과 splitter를 돌려준다.

    prose == 방출 누적 불변식을 모든 케이스에서 검증한다 — 오탐 복귀 경로의 이중 누적
    버그(리뷰 발견)가 정확히 이 단언이 빠진 케이스에서만 살아남았었다.
    """
    sp = TailSplitter()
    emitted = ''.join(sp.feed(c) for c in chunks)
    emitted += sp.finish()
    assert sp.prose == emitted, f'prose 불변식 위반: {sp.prose!r} != {emitted!r}'
    return emitted, sp


class TestTailSplitter:
    def test_정상_꼬리_분리(self):
        emitted, sp = _run(['답변입니다. ', f'{TAIL_START}[환불규정.docx v2]{TAIL_END}'])
        assert emitted == '답변입니다. '
        assert sp.prose == emitted                       # 화면 == 저장 불변식
        assert sp.tail_raw == '[환불규정.docx v2]'
        assert not sp.truncated

    def test_마커가_청크_경계에서_쪼개져도_새지_않는다(self):
        full = f'답변. {TAIL_START}[FAQ]{TAIL_END}'
        emitted, sp = _run(list(full))                   # 1글자씩 — 최악의 토큰 경계
        assert emitted == '답변. '
        assert TAIL_START[0] not in emitted or '§' not in emitted
        assert sp.tail_raw == '[FAQ]'

    def test_빈_꼬리(self):
        emitted, sp = _run([f'거절 답변. {TAIL_START}{TAIL_END}'])
        assert emitted == '거절 답변. '
        assert sp.tail_raw == ''

    def test_꼬리가_없으면_본문_전부_방출(self):
        emitted, sp = _run(['그냥 ', '답변만 ', '있다'])
        assert emitted == '그냥 답변만 있다'
        assert sp.tail_raw is None and not sp.truncated

    def test_본문_속_우연한_마커는_END_없으면_본문으로_복귀(self):
        # END 없이 MAX_TAIL_CHARS 초과 — 오탐 판정, 버퍼 전체가 본문으로 되돌아온다
        long_body = 'x' * (MAX_TAIL_CHARS + 10)
        emitted, sp = _run([f'본문 {TAIL_START}', long_body])
        assert f'본문 {TAIL_START}' in emitted + '…' or emitted.startswith('본문 ')
        assert TAIL_START in emitted                     # 마커째 화면에 보인다 (본문이었으니까)
        assert sp.tail_raw is None

    def test_END_뒤에_텍스트가_더_오면_오탐(self):
        # 진짜 꼬리는 스트림 마지막 — 마지막 마커만 인정
        emitted, sp = _run([f'본문 {TAIL_START}[FAQ]{TAIL_END} 그리고 계속되는 본문. ',
                            f'{TAIL_START}[환불규정.docx v2]{TAIL_END}'])
        assert f'{TAIL_START}[FAQ]{TAIL_END} 그리고 계속되는 본문. ' in emitted
        assert sp.tail_raw == '[환불규정.docx v2]'       # 마지막 것만 꼬리

    def test_취소로_꼬리가_잘리면_truncated_버퍼는_어디에도_없다(self):
        emitted, sp = _run(['답변. ', f'{TAIL_START}[환불규'])   # END 전에 스트림 종료
        assert emitted == '답변. '
        assert sp.truncated and sp.tail_raw is None
        assert '[환불규' not in sp.prose                 # 잘린 버퍼 폐기 — 저장에도 없다

    def test_END_뒤_후행_공백류는_꼬리로_인정하고_폐기(self):
        # FakeLlm·토크나이저가 끝에 공백/개행을 붙이는 관행 — 오탐이 아니다
        emitted, sp = _run(['답변. ', f'{TAIL_START}[FAQ]{TAIL_END} ', '\n'])
        assert emitted == '답변. '
        assert sp.tail_raw == '[FAQ]'
        assert not sp.truncated

    def test_마커_접두로_끝나는_본문은_finish가_되살린다(self):
        emitted, sp = _run(['각주는 §', '§ 두 개로 표기'])  # §§가 마커 접두 — 보류됐다 복귀
        assert emitted == '각주는 §§ 두 개로 표기'
        assert sp.tail_raw is None


class TestResolveCitations:
    def test_후보_교집합만_통과(self):
        tail = '[환불규정.docx v2][없는문서.pdf v9][FAQ]'
        cited = resolve_citations(tail, SRC, [])
        assert [c.filename for c in cited] == ['환불규정.docx', 'FAQ']   # 후보 밖은 버림

    def test_빈꼬리_None꼬리는_빈_목록(self):
        assert resolve_citations('', SRC, []) == []
        assert resolve_citations(None, SRC, []) == []

    def test_중복_라벨은_한_번만(self):
        assert len(resolve_citations('[FAQ][FAQ]', SRC, [])) == 1

    def test_첨부_인용은_FAQ_선례의_가짜_인용_객체(self):
        cited = resolve_citations(attachment_label('계약서.pdf'), SRC, ['계약서.pdf'])
        assert cited == [SourceCitation(document_id=None, filename='첨부: 계약서.pdf', version=1)]

    def test_NFD_라벨도_잡힌다(self):
        # fail-open(문법 미적용) 경로의 자유 생성 방어 — #34와 같은 원리
        import unicodedata
        nfd = unicodedata.normalize('NFD', source_label(SRC[0]))
        cited = resolve_citations(nfd, SRC, [])
        assert [c.filename for c in cited] == ['환불규정.docx']   # 반환은 원본 NFC 객체

    def test_malformed는_부분_복구_없이_빈_목록(self):
        # 라벨 형식이 아예 아니면(대괄호 없음) 아무것도 안 잡힌다 — 그럴듯한 복구 금지
        assert resolve_citations('환불규정.docx v2, FAQ', SRC, []) == []
