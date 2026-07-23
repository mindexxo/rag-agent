# KMS Phase 1.5 — 생성 평가 베이스라인 (top-5, eval v1)

RAGAS 도입 전, **리랭커 없는 hybrid 검색 + K=5** 기준의 검색·생성 베이스라인 스냅샷.

## 조건 (config)

- **검색**: hybrid RRF (BGE-M3 dense + sparse), **리랭커 off**, top-5
- **생성 모델**: Qwen/Qwen3-4B (vLLM, dtype half, max_model_len 16384), **temperature 0.6** (모델 generation_config 기본값)
- **gold**: `gold_set_v1.jsonl` (pp035 라벨 수정 반영), corpus fingerprint v1 (12문서 / 161청크)
- **지표**:
  - `R@k / Hit@1 / MRR` — 검색 (LLM 없음)
  - `EPCov` — 기대포인트 커버리지. **임베딩 코사인 매칭(BGE-M3, 임계 0.55)**. 구 부분문자열 방식은 서술형 포인트를 과소평가해 교체.
  - `Cite` — 답변의 `[파일명 vN]` 인용이 정답 문서와 일치하는 비율 (결정론, LLM 아님)

## 1. 검색 (retrieval, LLM 없음)

| type | R@5 | R@10 | R@20 | Hit@1 | MRR |
|---|---|---|---|---|---|
| single_fact | 0.90 | 0.97 | 1.00 | 0.63 | 0.76 |
| paraphrase | **0.71** | 0.83 | 0.94 | 0.37 | 0.53 |
| rare_lexical | 0.91 | 1.00 | 1.00 | 0.60 | 0.73 |
| multi_doc | 0.75 | 0.95 | 1.00 | 0.15 | 0.45 |

- R@20이 전 유형 0.94+ → **정답은 후보 풀에 거의 항상 존재.** 병목은 coverage가 아니라 **순위(top-5로 못 올림)**.
- 최약점 **paraphrase R@5 0.71**.

## 2. 생성 (generation, K=5)

`oracle` = 정답 청크 직접 주입(Qwen 순수 생성력) / `retrieved` = 실제 검색 top-5 주입(시스템 전체).

| type | oracle EPCov | retrieved EPCov | oracle Cite | retrieved Cite |
|---|---|---|---|---|
| single_fact | 0.92 | 0.92 | 1.00 | 0.93 |
| paraphrase | 0.97 | **0.74** | 1.00 | **0.77** |
| rare_lexical | 0.94 | 0.94 | 1.00 | 0.86 |
| multi_doc | 0.97 | 0.90 | 1.00 | 0.95 |

- **oracle 전 유형 0.92+** → 문맥만 제대로 주면 Qwen은 잘 답함.
- **paraphrase만 oracle 0.97 → retrieved 0.74** (−0.23): 검색 약점이 생성까지 번짐. 나머지 3유형은 검색 손실 거의 없음.
- rare_lexical Cite 0.86은 다출처 모호성(용어집 vs 도메인 문서) 영향 — 과소평가 소지. gold 보강 후보.

## 3. 핵심 결론

- **유일한 실질 약점은 paraphrase.** 나머지 3유형은 검색·생성 모두 운영 수준.
- **시도한 무(無)추가모델 레버 둘 다 paraphrase엔 실패:**
  - K 5→10: 검색 R@5 0.71→0.83 올랐으나 생성 EPCov 0.80→0.80 무동 (distractor 상쇄).
  - 쿼리 rewrite: 생성 Cite 0.80→0.60 악화 (rewrite 드리프트 → 오문서 인용).
- 데이터는 paraphrase 해결책으로 **리랭커**(1.5.C: paraphrase R@5 0.69→0.77, 전체 Hit@1 0.47→0.70)를 가리킴. 초기 서비스 부담이면 "상담원 보조라 0.80 수용"도 선택지.

## 4. 한계 / 주의

- **temp 0.6** = 확률적. 재현 시 ±1문항 노이즈 → 작은 델타는 신뢰 주의.
- EPCov 임계 0.55는 oracle 상한 보정으로 정함.
- `faithfulness`(환각) / `answer_relevancy`는 미측정 (LLM-judge 필요) → **다음 단계 RAGAS로 진행.**

## 5. 다음 — RAGAS

결정론 지표(EPCov/Cite)로 못 잡는 **환각(faithfulness)·답변 관련성(answer relevancy)·문맥 정밀/재현(context precision/recall)**을 RAGAS로 측정한다. gold·저장된 답변(`eval/results/generation_*.jsonl`)을 입력으로 활용.
