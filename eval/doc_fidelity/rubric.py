"""그림 캡션 판정 기준 — **정의점**(#247 결정 6, 2026-10-18 사용자 확정). Claude 1차 판정과 사람 재판정이 같은 기준을 쓴다.

유형: 흐름도 · 차트 · 화면 캡처 · 표형 그림 · 수식 · 사진·장식. 사진·장식(로고·아이콘·사진)은 글자 정보가 없어 **판정에서 빼고
개수만** 센다.

그림마다 센다:
  - 글자 누락(missing): 그림 속 글자 덩어리(라벨·칸·문장)가 캡션에 없다. 띄어쓰기·구두점 차이는 무시한다.
  - 오독(misread): 글자는 있지만 다르게 옮겼다.
  - 숫자 오류(number_errors): 숫자·금액·날짜·비율이 틀렸다 — 오독 중 가장 위험해 따로 센다(오독과 중복 집계하지 않는다).
  - 관계 오류(relation_errors): 흐름도 화살표 방향·순서, 표형 그림의 행·열 귀속, 차트 수치↔범례가 틀렸다. 관계가 없는 그림은 None.
  - 환각(hallucinations): 그림에 없는 내용을 지어냈다 — 가장 위험해 따로 센다.
  - 잡담(chatter): 머리말·맺음말·"관계 없음" 같은 군더더기. 정보 손상이 아니라 개수만 센다.

등급(grade) — **코드가 개수에서 다시 계산한다**(판정자가 매긴 등급과 다르면 코드 쪽이 최종, 불일치는 기록):
  - 위험: 숫자 오류나 환각이 1개 이상(틀린 사실이 답변에 들어갈 수 있다)
  - 오류: 누락+오독이 글자 덩어리의 10% 초과, 또는 관계 오류 1개 이상
  - 경미: 누락+오독이 10% 이하(0 초과)
  - 정확: 모두 0(잡담은 허용)
  캡션이 없는 그림(면적 하한 미만 등)은 글자 덩어리가 있으면 전부 누락 → 오류, 없으면 정확(잃은 것이 없다).
애매한 판정은 ambiguous=True로 표시하고 사람이 다시 본다(이슈의 '2인 교차'를 Claude 1차 + 사람 재판정으로 대신한다).
"""

from typing import Literal

from pydantic import BaseModel, Field, field_validator

TYPES = ('흐름도', '차트', '화면 캡처', '표형 그림', '수식', '사진·장식')
GRADES = ('정확', '경미', '오류', '위험')
EXCLUDED_TYPE = '사진·장식'
MINOR_RATIO = 0.10  # 경미/오류 경계 — 누락+오독 ÷ 글자 덩어리 (결정 6)


class Verdict(BaseModel):
    """판정자(Claude) 출력 계약. 키 이름은 프롬프트에 그대로 적는다(#253 교훈: 키를 안 적으면 다른 이름으로 답한다)."""

    type: Literal[TYPES]  # type: ignore[valid-type]
    text_units: int = Field(ge=0)  # 그림 속 글자 덩어리 수(라벨·칸·문장 단위)
    missing: int = Field(ge=0)
    misread: int = Field(ge=0)
    number_errors: int = Field(ge=0)
    relation_errors: int | None = Field(default=None, ge=0)
    hallucinations: int = Field(ge=0)
    chatter: int = Field(ge=0)
    grade: Literal[GRADES]  # type: ignore[valid-type]
    ambiguous: bool = False
    evidence: list[str] = Field(default_factory=list)  # "누락: …" / "숫자: 원본 3,500 → 캡션 3,800" / "환각: …"

    @field_validator('evidence', mode='before')
    @classmethod
    def _one_string_is_a_list(cls, v):
        # 판정자가 근거가 하나면 문자열로 답한다(10/18 실측 — '누락: 버튼 2개(-, +)'). 내용은 맞으니 목록으로 감싼다
        return [v] if isinstance(v, str) else v


def grade_of(v: Verdict) -> str | None:
    """개수 → 등급(결정 6). 사진·장식은 None(판정 제외)."""
    if v.type == EXCLUDED_TYPE:
        return None
    if v.number_errors or v.hallucinations:
        return '위험'
    lost = v.missing + v.misread
    if (v.relation_errors or 0) > 0 or lost > MINOR_RATIO * max(v.text_units, 1):
        return '오류'
    return '경미' if lost else '정확'


PROMPT = f"""첫 번째 입력은 문서에서 잘라낸 그림이고, 아래 '캡션'은 다른 모델이 이 그림을 글로 옮긴 것이다. 캡션이 그림을 얼마나
정확히 옮겼는지 아래 기준으로 판정하라. 그림을 직접 보고, 캡션을 믿지 말 것.

유형(type): {' · '.join(TYPES)} 중 하나. 로고·아이콘·사진처럼 글자 정보가 없으면 '사진·장식'.
그림마다 센다:
- text_units: 그림 속 글자 덩어리 수(라벨 하나·표 칸 하나·문장 하나가 1)
- missing: 그림에 있는데 캡션에 없는 글자 덩어리 수(띄어쓰기·구두점 차이는 무시)
- misread: 캡션에 있지만 다르게 옮긴 덩어리 수(숫자 오류는 여기서 빼고 아래에)
- number_errors: 숫자·금액·날짜·비율이 틀린 수
- relation_errors: 흐름도 화살표 방향·순서, 표형 그림의 행·열 귀속, 차트 수치↔범례가 틀린 수. 관계가 없는 그림이면 null
- hallucinations: 그림에 없는 내용을 지어낸 수
- chatter: 머리말·맺음말·"관계 없음" 같은 군더더기 문장 수
등급(grade): 숫자 오류·환각이 있으면 '위험', 누락+오독이 덩어리의 10% 초과이거나 관계 오류가 있으면 '오류',
그 밖에 누락·오독이 있으면 '경미', 모두 0이면 '정확'(잡담은 허용). 캡션이 비어 있으면 글자 덩어리는 전부 누락이다.
판정이 애매하면 ambiguous를 true로. evidence에는 틀린 곳을 짧게 적는다(예: "숫자: 그림 3,500원 → 캡션 3,800원").

JSON 하나만 출력한다. 키: type, text_units, missing, misread, number_errors, relation_errors, hallucinations, chatter,
grade, ambiguous, evidence

캡션:
"""
PROMPT_VERSION = 'v1'
