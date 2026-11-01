"""가짜 LLM — 시스템 프롬프트로 용도(인텐트·재작성·캐시 판정·생성)를 가려 고정 응답을 준다. fake_llm 픽스처가 쓴다."""

import asyncio
import json


def _current_question(user_content: str) -> str:
    """condense 유저 메시지에서 '현재 질문:' 다음 줄을 집는다 — FakeLlm 반향용."""
    lines = [ln.strip() for ln in user_content.splitlines() if ln.strip()]
    return lines[lines.index('현재 질문:') + 1] if '현재 질문:' in lines else lines[-1]


class FakeLlm:
    """시스템 프롬프트로 용도를 판별해 고정 응답을 주는 가짜 LLM.

    - 인텐트 가드 → safe/KNOWLEDGE JSON (테스트에서 intent_json으로 변경 가능)
    - condense → 마지막 user 내용에서 질문만 반향
    - 그 외(답변 생성) → answer를 어절 단위로 스트리밍
    """

    def __init__(self, answer: str | None = None):
        # 기본 답변은 **1번 문서를 인용하는** 출처 꼬리를 단다. 인용 0건이 "근거없음"의 정의라
        # (캐시 제외·지표 분자) 빈 꼬리를 기본값으로 두면 대다수 테스트의 "정상 답변"이 근거없음으로
        # 분류된다. 후보가 없는 테스트(차단·근거없음·OTHER)에서는 1번이 범위 밖이라 resolve_citations가
        # 걸러내 citations=[]가 그대로 유지된다 — 그 경로의 기대값은 안 바뀐다.
        # 인용 검증 테스트는 여전히 answer를 직접 교체해서 쓴다.
        # 어절 3개 유지 주의: pause_after_tokens=2(셋째 토큰 정지)를 쓰는 취소 테스트들의 전제.
        from rag.turn.citation_labels import citation_tail

        self.answer = answer if answer is not None else f'테스트 답변입니다. {citation_tail([1])}'
        self.intent_json = '{"safe": true, "intent": "KNOWLEDGE"}'
        self.reuse_judge_json = '{"same_answer": true, "reason": "테스트 고정 승인"}'  # 캐시 판정
        self.calls: list[str] = []  # 어떤 용도로 호출됐는지 기록 (검증용)
        self.system_prompts: list[tuple[str, str]] = []  # (용도, 시스템 프롬프트) — 주입 내용 검증용
        self.citation_tails: list = []  # astream별 citations 의도(CitationTail|None) — 꼬리 제약 주입 검증용
        self.schemas: list = []  # acomplete별 schema 의도 — 스키마 주입 증거
        # 취소 테스트용 정지 지점. None이면 멈추지 않는다.
        # 왜 필요한가: astream에 진짜 await가 없으면 이벤트 루프에 제어가 넘어가지 않아
        # task.cancel()이 전달될 지점 자체가 없다 — 생성이 그대로 완주한다.
        # 시간(sleep) 대신 Event로 멈추는 이유: 테스트가 "정확히 N토큰 뒤"를 잡을 수 있어
        # CI에서 타이밍 플레이크가 나지 않는다.
        self.pause_after_tokens: int | None = None
        self.finish_reason = 'stop'  # astream 완주 시 콜백에 전달 — 'length'로 바꿔 잘림 재현
        # 생성 실패 테스트용 — N번째 토큰에서 스트림이 끊긴 상황(LLM·네트워크 장애) 주입.
        # pause와 달리 재개가 없다: 실패 경로의 저장 규칙을 보려는 것이라 중단 자체가 목적.
        self.raise_after_tokens: int | None = None
        self.paused = asyncio.Event()  # 테스트: 정지 지점 도달을 기다린다
        self.resume = asyncio.Event()  # 테스트: set()하면 스트림이 계속된다

    def _kind(self, messages: list[dict]) -> str:
        system = messages[0]['content'] if messages else ''
        if '분류기' in system or '입력 검사' in system:
            return 'intent'
        if '질문 여러 개' in system:  # CONDENSE_MULTI — '재작성'보다 먼저 (문자열 겹침)
            return 'condense_multi'
        if '재작성' in system:
            return 'condense'
        if '재사용 판정기' in system:  # 캐시 재사용 판정
            return 'reuse_judge'
        return 'generate'

    def _record(self, kind: str, messages: list[dict]) -> None:
        self.system_prompts.append((kind, messages[0]['content'] if messages else ''))

    async def acomplete(self, messages: list[dict], *, schema=None, citations=None, on_usage=None) -> str:
        kind = self._kind(messages)
        self.calls.append(kind)
        self._record(kind, messages)
        self.schemas.append(schema)
        if kind == 'intent':
            return self.intent_json
        if kind == 'reuse_judge':
            # 판정 승인 없이는 캐시 히트가 없으므로 기본 승인.
            # 거절 시나리오는 reuse_judge_json을 교체해서 쓴다.
            return self.reuse_judge_json
        if kind == 'condense':
            # 마지막 줄이 아니라 '현재 질문:' 라벨 다음 줄(실제 질문)을 JSON으로 반향한다.
            return json.dumps({'standalone': _current_question(messages[-1]['content'])}, ensure_ascii=False)
        if kind == 'condense_multi':
            # 멀티쿼리 규격 반향 — variants를 채워야 플래그 on 통합 테스트가
            # off와 동일 경로로 축소되지 않는다.
            q = _current_question(messages[-1]['content'])
            return json.dumps({'standalone': q, 'variants': [f'{q} 변형A', f'{q} 변형B']}, ensure_ascii=False)
        return self.answer

    async def astream(self, messages: list[dict], *, schema=None, citations=None, on_finish_reason=None, on_usage=None):
        kind = self._kind(messages)
        self.calls.append('stream:' + kind)
        self._record(kind, messages)
        self.citation_tails.append(citations)
        for i, token in enumerate(self.answer.split(' ')):
            if self.raise_after_tokens is not None and i == self.raise_after_tokens:
                raise RuntimeError('LLM 스트림 중단 (테스트 주입)')
            if self.pause_after_tokens is not None and i == self.pause_after_tokens:
                self.paused.set()
                await self.resume.wait()  # 취소 테스트가 여기서 task.cancel()을 건다
            yield token + ' '
        # 실제 구현(rag/llm/openai_compat.py)처럼 완주 시 종료 사유를 콜백으로 알린다.
        # finish_reason 속성을 'length'로 바꾸면 잘림 시나리오를 흉내 낼 수 있다.
        if on_finish_reason:
            on_finish_reason(self.finish_reason)
