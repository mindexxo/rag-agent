"""시나리오 러너 순수 함수 단위 테스트 (#99) — DB·LLM 없이 판정 로직을 고정.

러너 본체는 실서버 축(vLLM 필요)이라 CI에서 못 돌리지만, 판정 함수(_check)와 시드
가드(_seed_turn의 cancelled 제약)는 순수 로직이라 여기서 값싸게 고정한다. 리뷰에서
숫자 경계 오탐("5만원"이 "15만원"에 걸림)이 지적됐으므로 그 회귀를 특히 래칫한다.
"""
import pytest

from eval.scenario import _check


def _obs(route="knowledge", citations=None, answer="", attachment_files=None):
    return {"route": route, "citations": citations or [], "answer": answer,
            "attachment_files": attachment_files or [], "cache_hit": False}


class TestCheckContains:
    def test_숫자_경계_오탐_방지(self):
        # "5만원" 기대인데 답변이 "15만원"이면 실패해야 한다 (리뷰 발견 회귀 래칫)
        fails = _check({"expect_contains": ["5만원"]}, _obs(answer="무료배송 기준은 15만원입니다"))
        assert any("5만원" in f for f in fails)

    def test_정확한_수치는_통과(self):
        assert _check({"expect_contains": ["5만원"]}, _obs(answer="기준은 5만원입니다")) == []

    def test_콤마_표기_흡수(self):
        # 답변이 "129,000원"이어도 "129000" 기대가 통과 (_contains_point 콤마 규칙)
        assert _check({"expect_contains": ["129000"]}, _obs(answer="금액은 129,000원")) == []

    def test_금지_문구_숫자경계(self):
        # "PWNED-9" 같은 인젝션 흔적은 정확히 있을 때만 실패
        assert _check({"expect_not_contains": ["LEAK-0001"]}, _obs(answer="정상 답변")) == []
        fails = _check({"expect_not_contains": ["LEAK-0001"]}, _obs(answer="LEAK-0001 유출"))
        assert fails


class TestCheckStructure:
    def test_route_불일치(self):
        fails = _check({"expect_route": "knowledge"}, _obs(route="other"))
        assert any("route" in f for f in fails)

    def test_거절은_인용0건(self):
        # expect_refuse=True는 인용이 비어야 통과
        assert _check({"expect_refuse": True}, _obs(citations=[])) == []
        assert _check({"expect_refuse": True}, _obs(citations=["doc.pdf"]))

    def test_인용_파일명_부분일치(self):
        assert _check({"expect_docs": ["환불반품정책"]},
                      _obs(citations=["summers_01_환불반품정책.pdf"])) == []

    def test_차단_첨부_주입되면_실패(self):
        # 격리 회귀 탐지 — 차단 턴 첨부가 다음 턴 주입 목록에 있으면 실패 (#63 코드 레벨)
        fails = _check({"expect_not_attached": ["인젝션.txt"]},
                       _obs(attachment_files=["인젝션.txt"]))
        assert fails
        assert _check({"expect_not_attached": ["인젝션.txt"]},
                      _obs(attachment_files=["정상.pdf"])) == []

    def test_예외_턴은_즉시_실패(self):
        assert _check({"expect_route": "knowledge"}, {"error": "ValueError: x"})