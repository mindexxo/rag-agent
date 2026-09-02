"""기계 가드(rag/cache.guard_blocks, #113) 단위 계약 — DB·임베딩 불요, 순수 문자열 판정.

케이스 출처: eval/cache_set_v1.jsonl에서 실측된 오답 재생 쌍(임계·doc집합을 모두 뚫은 것)과
히트가 정당한 paraphrase 쌍. 가드의 오판 비용은 비대칭(잘못 차단=미스 1회, 잘못 통과=오답
재생)이므로 차단 케이스가 계약의 본체다.
"""
import pytest

from rag.cache import guard_blocks


class TestGuardBlocks_차단:
    """답이 달라질 신호가 있는 쌍 — 반드시 차단."""

    def test_수치가_다르면_차단(self):
        # 실측 0.9717 누수 쌍 — 10% 경계로 부분환불 vs 전량환불
        assert guard_blocks("30구 계란 중 3개 깨졌으면 얼마 환불되나요?",
                            "30구 계란 중 5개 깨졌으면 얼마 환불되나요?") == "numeric"

    def test_날짜_숫자가_다르면_차단(self):
        # 실측 0.9618 누수 쌍 — 경과조치 종료일 앞뒤
        assert guard_blocks("2026년 4월에 구 RF 카드로 리필하면 15% 할인 받을 수 있어요?",
                            "2026년 6월에 구 RF 카드로 리필하면 15% 할인 받을 수 있어요?") == "numeric"

    def test_부정_극성이_다르면_차단(self):
        # 실측 0.9627 누수 쌍 — 서로 다른 판정 목록
        assert guard_blocks("세탁 후 프린트 크랙이 불량으로 인정되는 경우는?",
                            "세탁 후 프린트 크랙이 불량으로 인정 안 되는 경우는?") == "negation"

    def test_상대_시간_어휘가_다르면_차단(self):
        # 실측 0.9927 누수 쌍 — 7일 보상 경계
        assert guard_blocks("닷새 전에 산 옷이 세일에 들어갔어요. 차액 보상되나요?",
                            "보름 전에 산 옷이 세일에 들어갔어요. 차액 보상되나요?") == "time"

    def test_오늘_나흘_시간_어휘_차단(self):
        assert guard_blocks("오늘 아침에 받은 고기가 상했어요. 보상되나요?",
                            "나흘 전에 받은 고기가 상했어요. 보상되나요?") == "time"

    def test_수치_유무_자체가_다르면_차단(self):
        # 한쪽에만 수치가 있어도 조건이 달라진 것
        assert guard_blocks("반품 배송비 얼마예요?",
                            "반품 배송비가 6,000원인가요?") == "numeric"

    def test_대칭이다(self):
        a, b = "환불 되는 경우 알려줘", "환불 안 되는 경우 알려줘"
        assert guard_blocks(a, b) == guard_blocks(b, a) == "negation"


class TestGuardBlocks_통과:
    """같은 답으로 수렴하는 쌍 — 차단하면 안 됨 (paraphrase 히트 보존)."""

    def test_어미_변형은_통과(self):
        assert guard_blocks("단순변심 반품 기간은 며칠인가요?",
                            "단순변심 반품 기간은 며칠이죠?") is None

    def test_구어체_paraphrase는_통과(self):
        assert guard_blocks("단순변심 반품 며칠까지 돼요?",
                            "단순변심 반품 기간 알려주세요") is None

    def test_천단위_콤마는_같은_수치다(self):
        assert guard_blocks("반품 배송비가 3,000원인가요?",
                            "반품 배송비가 3000원인가요?") is None

    def test_같은_수치_같은_극성은_통과(self):
        assert guard_blocks("30구 계란 중 3개 깨졌으면 얼마 환불되나요?",
                            "계란 30구에서 3개가 파손됐는데 환불이 얼마나 되나요?") is None

    def test_양쪽_다_부정이면_통과(self):
        # 극성 '집합'이 같으면 차단 사유가 아니다
        assert guard_blocks("보온이 안 되면 교환되나요?",
                            "보온이 안 되는데 교환 가능한가요?") is None

    def test_한자어_시간_표현은_고유어와_같은_시점이다(self):
        # 40쌍 실측의 오차단 사례 — '이번 달'↔'당월'은 같은 시점
        assert guard_blocks("홈플 프라임 해지하면 이번 달 구독료 돌려받아요?",
                            "프라임 구독 해지 시 당월 구독료 환불 기준이 어떻게 되나요?") is None

    def test_한자어끼리도_다른_시점이면_차단(self):
        # 정규화는 표기 통일일 뿐 — 당월 vs 익월은 여전히 다른 시점
        assert guard_blocks("당월 출금 되나요?", "익월 출금 되나요?") == "time"


class TestGuardBlocks_경계:
    def test_빈_문자열은_통과(self):
        assert guard_blocks("", "") is None

    def test_소수점_수치_구분(self):
        assert guard_blocks("할인율이 2.5%인가요?", "할인율이 2%인가요?") == "numeric"
