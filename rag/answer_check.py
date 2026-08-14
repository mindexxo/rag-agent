"""답변 텍스트 판정 — 거절 여부와 실인용 문서 (#36).

프롬프트 조립이 아니라 **LLM이 낸 답변을 해석하는** 일이라 rag/prompts.py에서 떼어냈다.
소비처도 프롬프트와 무관하다 — rag/service.py(저장·캐시 제외 판정), rag/streaming.py(인용 정정),
eval/refusal.py(거절율 측정).

의존은 text_norm뿐이다(단방향). 프롬프트 텍스트·조립 어느 쪽도 참조하지 않는다.
"""
from text_norm import nfc


def cited_filenames(answer: str, sources) -> list[str]:
    """답변이 실제로 인용한 문서 파일명 목록 (라벨 [파일명 vN] / [FAQ] 매칭).

    저장 시점에 확정해 messages.cited_docs로 보존 — 지표·FE가 재파싱 없이 사용.
    sources: SourceCitation 리스트 (filename/version 속성).

    비교 직전에 양쪽을 NFC로 맞춘다 (#34). 경계 정규화로 저장값은 NFC지만 **LLM 출력은
    우리가 통제하는 입력이 아니고**, 마이그레이션 전 남아 있는 NFD 행도 여기서 구제된다.
    반환값은 원본 s.filename을 유지해야 cited_docs가 DB filename과 조인된다.
    """
    nfc_answer = nfc(answer)
    return [
        s.filename for s in sources
        if (s.filename == 'FAQ' and '[FAQ]' in nfc_answer)
        or f'[{nfc(s.filename)} v{s.version}]' in nfc_answer
    ]


# 거절 문구에서 주어를 뗀 핵심부. 모델이 주어를 바꿔 쓰는 변형까지 잡는다 —
# 실측(2026-08-05, 거절축 108문항): "죄송하지만, 새벽배송 서비스는 제공된 문서에서 확인할 수
# 없습니다"처럼 뒷부분은 그대로인 변형이 나와 완전일치로는 미검출(56/57)이었다.
# 좁게만 넓힌 이유: "명시되어 있지 않습니다"류까지 포함하면, 전제를 부정한 뒤 실제로 답하는
# 정정 답변(trap 유형: "아니요, …명시되어 있지 않습니다. - 닭가슴살은 …6,900원입니다")을
# 거절로 오판한다. 그 케이스가 실제로 있어 확인했다.
# 표현이 더 벌어지면 임베딩 유사도 판정으로 교체 — 같은 실측에서 거절/답변이 완전 분리
# (거절 0.746~1.000 vs 답변 0.303~0.509, 임계값 0.6~0.7)되는 것까지 확인해뒀다.
_REFUSAL_CORE = '제공된 문서에서 확인할 수 없'


def is_refusal(answer: str) -> bool:
    """LLM 거절 답변 판정 — 캐시 제외·인용 제거·지표가 공유하는 단일 판정점."""
    return _REFUSAL_CORE in answer
