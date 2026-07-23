# KMS Phase 1.5 — RAGAS 평가 (faithfulness / answer relevancy, eval v1)

결정론 지표(EPCov/Cite)로 못 잡는 **환각(faithfulness)·답변 관련성(answer relevancy)**을
RAGAS 프레임워크로 측정. 검색·생성 베이스라인은 `report_generation_v1.md` 참고.

## 조건 (config)

- **프레임워크**: RAGAS 0.4.3 (`ragas.evaluate` + `Faithfulness`, `ResponseRelevancy`)
- **평가 대상 답변**: `eval/results/generation_retrieved.jsonl` (hybrid no-rerank, K=5, Qwen3-4B temp 0.6)
- **judge LLM**: **GPT-5 mini** (OpenAI API) — 외부 강모델로 self-judge 편향 회피
- **임베딩**: 로컬 BGE-M3 재활용 (answer_relevancy용, 외부 콜 절약)
- **입력**: gold v1 + 저장된 답변/문맥 → RAGAS `EvaluationDataset` (120샘플)
- **metric**:
  - `faithfulness` — 답변 주장이 retrieved_contexts에 근거하는가 (환각 여부). reference 불필요.
  - `answer_relevancy` — 답변이 질문에 맞는가. reference 불필요.

## ⚠ 측정 아티팩트 발견 → 수정 (1차 → 2차 측정)

**1차 측정(faith ALL 0.637)은 어댑터 버그성 아티팩트로 저평가된 값이었다.**

- 원인: 어댑터가 chunk의 `text`만 저장 → judge가 보는 문맥에 파일명이 없음.
  생성 프롬프트는 `[파일명 vN] 섹션:...` 헤더를 붙여주므로(`rag/prompts.py`),
  답변의 인용 문장("[kms_01...pdf v1] 문서에서 명시...")이 전부 "근거 없는 주장" 판정 → 전 샘플 일괄 감점.
- 수정: contexts를 **생성기가 본 것과 동일한 헤더 포함 형식**으로 저장(`eval/generation.py`).
  기존 답변은 유지한 채 contexts만 백필(검색은 결정론, 재검색 텍스트 불일치 0건)하여 아티팩트 효과만 분리 측정.
- 1차 결과 보존: `eval/results/ragas_retrieved_noheader.csv`

## 결과 (retrieved, 120샘플) — 헤더 수정 전/후

| type | n | faith(1차) | **faith(2차)** | Δ | ansrel(1차) | **ansrel(2차)** |
|---|---|---|---|---|---|---|
| single_fact | 30 | 0.617 | **0.931** | +0.31 | 0.893 | 0.898 |
| paraphrase | 35 | 0.621 | 0.835 | +0.21 | **0.734** | **0.731** |
| rare_lexical | 35 | 0.702 | 0.918 | +0.22 | 0.892 | 0.885 |
| multi_doc | 20 | 0.579 | **0.801** | +0.22 | 0.837 | 0.791 |
| **ALL** | 120 | 0.637 | **0.878** | +0.24 | 0.837 | 0.828 |

- answer_relevancy는 무변동(문맥 헤더와 무관한 지표) → 수정이 다른 축을 오염시키지 않았다는 일관성 검증.
- per-sample: `eval/results/ragas_retrieved.csv` (2차)

## 관찰 (2차 기준)

- **진짜 faithfulness는 0.88.** 남은 갭 ~0.12의 구성:
  - **실오류 (pp029 대표)**: "받자마자 바꾸려면?"(교환)에 반품 요금 3,000원 답변 — 문맥 표엔 교환 6,000원.
    0.0 → 0.67로 회복됐지만 틀린 요금 주장은 여전히 감점 = judge가 실오류를 정확히 잡음.
  - **멀티홉 엄격성 (md016 대표)**: 두 chunk를 결합해야 나오는 답("DL-03=지연보상" + "3~5일 지연=3,000원 쿠폰")은
    부분 인정(0.50). **multi_doc 0.801 최저 유지** — 다문서 종합 시 근거 이탈이 실제로 더 잦음.
- **순수 아티팩트였던 샘플들(sf001·sf002·sf023·sf030)은 1.00 완전 회복.**
- **남은 실질 환각 의심 후보**: sf010(0.20), pp018(0.33), rl015(0.33) — 후속 정성 리뷰 대상.
- **answer_relevancy: paraphrase 최저(0.73) 그대로.** 기존 결론(paraphrase 유일 약점)과 일치.
- **EPCov의 맹점 실증**: pp029는 EPCov=1.0으로 통과했었음 — 임베딩 매칭은 문장이 유사하면 **숫자가 틀려도 커버로 셈**.
  faithfulness(RAGAS)와 EPCov(결정론)는 상호보완 관계.
- **judge 크로스 체크**(1차 시점, 3샘플): Gemini 2.5 flash 0.53 vs GPT-5 mini 0.55 근접.

## 한계 / 주의

- **유형 매핑이 위치 기반.** CSV 행 순서 = `generation_retrieved.jsonl` 순서 가정. id 기반 조인이 안전 → 어댑터 개선 여지.
- **평가 대상 답변 temp 0.6** (비결정) — 재실행 시 소폭 변동.
- **context_precision / recall 미측정** — reference(정답 문장) 부재(gap2). gold v1은 expected_points만 보유.

## 3차 측정 — 역질문 규칙(SYSTEM_PROMPT 규칙 4) 추가 후 재생성분

| | 1차(아티팩트) | 2차(헤더 수정) | 3차(규칙 4 답변) |
|---|---|---|---|
| faithfulness ALL | 0.637 | 0.878 | **0.862** |
| answer_relevancy ALL | 0.837 | 0.828 | 0.825 |

- **규칙 4 영향 = 중립**: 2차→3차 델타는 temp 0.6 재샘플링 노이즈 범위. EPCov/Cite도 회귀 없음, gold 120문항에서 과잉 되묻기 0건 (모호 질문 스모크에선 조건별 병렬 안내로 반응).
- **pp029 완치**: 규칙 4 생성분에서 "교환 6,000원 고객 부담" 정답, faith 0.0(1차)→1.00(3차).
- **저점 꼬리는 확률적**: faith<0.5가 2차(pp018·rl015 등)와 3차(pp006·pp023·md001·md008)에서 물갈이됨 — temp 0.6 재샘플링마다 삐끗하는 샘플이 바뀜. **sf010만 2회 연속 저점(0.20→0.25)** = 유일한 지속 문제 후보.
- 결과 파일: 3차 `ragas_retrieved.csv` / 2차 `ragas_retrieved_prompt7.csv` / 1차 `ragas_retrieved_noheader.csv`. 답변: 3차 `generation_retrieved.jsonl` / 2차 이전 `generation_retrieved_prompt7.jsonl`.

## sf010 정성 리뷰 (지속 저점 1건)

- 정체: **"정답 + 근거 없는 부연"** 패턴. 핵심 답("분실·오배송 시 재발송")은 문맥에 그대로 있어 정답이나,
  "이외의 경우는 교환·반품으로 처리"라는 부연이 어느 문맥에도 없음 — 옆 chunk(교환/반품 문서)를 보고 이어붙인 도출성 사족.
- 구조적 긴장: 프롬프트 규칙 2는 "규정 비교·연결 도출"을 허용, faithfulness judge는 미명시 주장을 감점 — 서로 당김.
- **최종 결론: 잔여 저점의 정체는 ① temp 재샘플링 꼬리(확률적) ② 도출성 부연 감점 — 심각한 실환각 없음.**

## 다음

1. 잔여 갭(~0.14)은 temp(확률적 꼬리)·리랭커(문맥 질, multi_doc) 축 — 서비스화 단계에서 결정.
2. context precision/recall용 reference 표현 결정 (expected_points 조합 vs oracle 답 활용).
