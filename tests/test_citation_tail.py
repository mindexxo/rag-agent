r"""출처 꼬리 분리·해석 단위 테스트 (#56) — rag/citation_tail.

TailSplitter는 vLLM 토큰 경계가 임의라는 전제에서 동작해야 한다 — 마커가 청크 경계에서
쪼개져 도착하는 경우를 1글자 단위 feed로 재현한다. 핵심 불변식: **feed·finish 반환값의
누적 == prose == 화면에 보인 것 == 저장되는 것**, 그리고 꼬리·잘린 버퍼는 그 어디에도 없다.
꼬리 내용물은 JSON 정수 배열(««[1,3]»», #65 — 그전엔 ««1,3»») — 번호↔문서 순서는
citation_labels가 단일 정의점. 이 모듈은 마커 위치만 보므로 안쪽 형식과 무관하다:
resolve_citations의 \d+ 추출이 대괄호·쉼표·여백을 무시해 두 형식을 같게 읽는다
(fail-open으로 제약이 떨어진 경로에서 옛 형식이 나와도 흡수된다는 게 그 관용성의 값이다).
"""
import pytest

from rag.citation_labels import TAIL_END, TAIL_START
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
        emitted, sp = _run(['답변입니다. ', f'{TAIL_START}1,2{TAIL_END}'])
        assert emitted == '답변입니다. '
        assert sp.prose == emitted                       # 화면 == 저장 불변식
        assert sp.tail_raw == '1,2'
        assert not sp.truncated

    def test_마커가_청크_경계에서_쪼개져도_새지_않는다(self):
        full = f'답변. {TAIL_START}1,2{TAIL_END}'
        emitted, sp = _run(list(full))                   # 1글자씩 — 최악의 토큰 경계
        assert emitted == '답변. '
        assert TAIL_START[0] not in emitted              # 마커 문자가 한 자도 안 샜다
        assert sp.tail_raw == '1,2'

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
        assert emitted.startswith('본문 ')
        assert TAIL_START in emitted                     # 마커째 화면에 보인다 (본문이었으니까)
        assert sp.tail_raw is None

    def test_END_뒤에_텍스트가_더_오면_오탐(self):
        # 진짜 꼬리는 스트림 마지막 — 마지막 마커만 인정
        emitted, sp = _run([f'본문 {TAIL_START}1{TAIL_END} 그리고 계속되는 본문. ',
                            f'{TAIL_START}2{TAIL_END}'])
        assert f'{TAIL_START}1{TAIL_END} 그리고 계속되는 본문. ' in emitted
        assert sp.tail_raw == '2'                        # 마지막 것만 꼬리

    def test_취소로_꼬리가_잘리면_truncated_버퍼는_어디에도_없다(self):
        emitted, sp = _run(['답변. ', f'{TAIL_START}1'])   # END 전에 스트림 종료
        assert emitted == '답변. '
        assert sp.truncated and sp.tail_raw is None
        assert '1' not in sp.prose                       # 잘린 버퍼 폐기 — 저장에도 없다

    def test_END_뒤_후행_공백류는_꼬리로_인정하고_폐기(self):
        # FakeLlm·토크나이저가 끝에 공백/개행을 붙이는 관행 — 오탐이 아니다
        emitted, sp = _run(['답변. ', f'{TAIL_START}1{TAIL_END} ', '\n'])
        assert emitted == '답변. '
        assert sp.tail_raw == '1'
        assert not sp.truncated

    def test_마커_접두로_끝나는_본문은_finish가_되살린다(self):
        # 시작 마커의 절반(« 한 자)으로 끝나는 본문 — 보류됐다 finish에서 복귀해야 한다
        half = TAIL_START[0]
        emitted, sp = _run([f'인용부호는 {half}'])
        assert emitted == f'인용부호는 {half}'
        assert sp.tail_raw is None

    def test_본문_속_외따로_한_글자_마커는_그대로_방출(self):
        # « 하나만으로는 마커가 아니다(«« 두 겹이 마커) — 뒤에 본문이 이어지면 복귀 방출
        half = TAIL_START[0]
        emitted, sp = _run([f'{half}인용{TAIL_END[0]} 표기 본문. ', f'{TAIL_START}1{TAIL_END}'])
        assert emitted == f'{half}인용{TAIL_END[0]} 표기 본문. '
        assert sp.tail_raw == '1'


class TestResolveCitations:
    def test_번호를_후보_순서로_매핑(self):
        cited = resolve_citations('1,2', SRC, [])
        assert [c.filename for c in cited] == ['환불규정.docx', 'FAQ']

    def test_범위_밖_번호는_버림(self):
        cited = resolve_citations('1,9', SRC, [])
        assert [c.filename for c in cited] == ['환불규정.docx']   # 9번 후보 없음 — 조용히 제외
        assert resolve_citations('0', SRC, []) == []              # 0은 유효 번호가 아니다(1-based)

    def test_빈꼬리_None꼬리는_빈_목록(self):
        assert resolve_citations('', SRC, []) == []
        assert resolve_citations(None, SRC, []) == []

    def test_중복_번호는_한_번만(self):
        assert len(resolve_citations('2,2', SRC, [])) == 1

    def test_첨부_인용은_FAQ_선례의_가짜_인용_객체(self):
        # 첨부 번호는 sources 다음부터 — SRC 2건이므로 3번이 첫 첨부
        cited = resolve_citations('3', SRC, ['계약서.pdf'])
        assert cited == [SourceCitation(document_id=None, filename='첨부: 계약서.pdf', version=1)]

    def test_구분자는_관용(self):
        # fail-open(제약 미적용) 경로의 자유 생성 대비 — 숫자 뭉치만 뽑는다
        assert [c.filename for c in resolve_citations('1 2', SRC, [])] == ['환불규정.docx', 'FAQ']

    def test_숫자가_없으면_빈_목록(self):
        # malformed 꼬리는 부분 복구 없이 — 그럴듯한 복구는 실패를 지표에서 숨긴다
        assert resolve_citations('환불규정.docx, FAQ', SRC, []) == []

    def test_여러_자리_번호는_한_뭉치로_읽는다(self):
        # '12'가 1,2로 쪼개지면 오귀속 — \d+ 뭉치 추출이라 12 단일 번호(범위 밖)로 버려져야 한다
        assert resolve_citations('12', SRC, []) == []

    def test_반환_순서는_꼬리_등장_순서(self):
        cited = resolve_citations('2,1', SRC, [])
        assert [c.filename for c in cited] == ['FAQ', '환불규정.docx']
