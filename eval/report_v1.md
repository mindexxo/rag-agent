# KMS Phase 1.5 — 검색 평가 결과 (eval v1)

- **gold**: `eval/gold_set_v1.jsonl` (v1, 210문항). 본 리포트는 검색 subset 120문항(`single_fact`/`paraphrase`/`rare_lexical`/`multi_doc`) 대상.
- **corpus**: fingerprint v1 (`eval/corpus_fingerprint_v1.json`), 문서 12종 / 청크 161.
- **범위**: 1.5.B(검색) + Q2(임베딩 비교)까지. **생성(1.5.D)·게이트/안전성(1.5.E)·운영지표(1.5.F)는 미평가.**
- **결과 파일**: `eval/results/retrieval_bge.jsonl`, `retrieval_kure.jsonl`

---

## 1. 평가 방법

- 검색/생성 분리 평가(§3). 본 단계는 **LLM 없이 검색기만** 측정.
- B0에서 `retrieve_candidates()`(후보 top-20) / `apply_gate()`(게이트) 분리 → Recall@k·threshold sweep 가능.
- gold 정답은 안정 키(filename/heading_path/snippet) → 평가 시작 시 현재 DB id로 resolve(§4.1).
- 지표: Recall@5/@20, Hit@1, MRR (chunk 우선, 없으면 doc 폴백).

## 2. 베이스라인 — hybrid RRF (BGE-M3)

| type | R@5 | R@20 | Hit@1 | MRR |
|---|---|---|---|---|
| single_fact | 0.90 | 1.00 | 0.63 | 0.76 |
| paraphrase | 0.69 | 0.91 | 0.37 | 0.52 |
| rare_lexical | 0.91 | 1.00 | 0.60 | 0.73 |
| multi_doc | 0.75 | 1.00 | 0.15 | 0.45 |

게이트 오거부(근거 있는데 거부) sweep: th=0.4 → 42/120, 0.5 → 17, **0.6 → 2**, 0.7 → 0.

## 3. 핵심 진단 — 병목은 "순서"

- **R@20 ≈ 1.00**(paraphrase만 0.91) → 정답은 후보 풀에 거의 항상 존재. **검색 coverage는 충분.**
- **Hit@1 전반 낮음(0.15~0.63) + R@5 아슬아슬** → 정답을 top-1/top-5로 못 올림 = **ranking(순서)이 병목.**
- §9.1 리랭커 착수 조건(recall@20 高 / precision@5 低) 충족.
- **paraphrase 최약**(R@5 0.69): 표현 다른 질문에 검색이 약함.

### 3.1 지표 해석 주의 — R@5는 하한선 (2026-07-03 정성 리뷰)

paraphrase 실패 11건을 눈으로 확인한 결과, R@5 절대값은 실제보다 비관적이다:

- **chunk 단위 엄격 채점의 편향**: gold가 특정 chunk 하나를 정답으로 지정하는데, **동등 정보가 다른 chunk에도 존재**하는 경우
  그 chunk가 top-5에 들어 생성이 정답을 내도 R@5=0으로 집계됨.
  예: pp029 — gold chunk(4.1 단순변심 교환)는 7위였지만 같은 답(6,000원)이 담긴 기준표 chunk가 2위 → 생성 정답, 지표는 실패.
- **실패의 주 패턴은 자매 문서 혼동**: "감감무소식"(배송조회↔배송지연대응), "입어본 건데"(환불↔교환반품) —
  구어체 질의에서 sparse가 죽고 dense가 주제 유사 문서를 구분 못 함. 정답 문서 근처까지는 감.
- **용도 정리**: R@5는 설정 비교용(편향이 A/B 양쪽에 동일하게 걸려 상쇄) 진단 지표.
  **절대 품질 보고에는 생성 지표(EPCov·faithfulness·answer_relevancy)를 인용할 것.**
- gold v2(실문서 전환 시 재구축) 교훈: **expected_chunks를 처음부터 복수 허용으로 라벨링** (스키마는 이미 리스트).
  현 gold v1은 수정하지 않음 — 실문서 전환 시 폐기 예정이며, 상대 비교엔 지장 없음.

## 4. Q2 — 임베딩 비교 (BGE-M3 vs KURE-v1)

KURE-v1(nlpai-lab, MTEB-ko 검색 1위, bge-m3 한국어 파인튜닝, 1024d/MIT). sparse는 BGE-M3 고정, **dense만 교체**(§7 한 번에 하나).

**하이브리드 (R@5 / Hit@1):**

| type | BGE | KURE |
|---|---|---|
| single_fact | 0.90 / 0.63 | 0.90 / 0.67 |
| paraphrase | 0.69 / 0.37 | 0.69 / 0.37 |
| rare_lexical | 0.91 / 0.60 | 0.94 / 0.60 |
| multi_doc | 0.75 / 0.15 | 0.75 / 0.20 |

**dense-only 진단 (sparse·RRF 제외, R@5 / Hit@1):**

| type | BGE | KURE |
|---|---|---|
| single_fact | 0.93 / 0.60 | 0.93 / 0.67 |
| paraphrase | 0.74 / 0.37 | 0.74 / 0.34 |
| rare_lexical | 0.89 / 0.60 | 0.89 / 0.60 |
| multi_doc | 0.80 / 0.30 | 0.65 / 0.25 |

**판정: 동률.** 하이브리드·dense-only 둘 다 KURE가 BGE-M3를 못 이김(차이 ≤2문항, §8.4 동률). 최약점 paraphrase는 KURE로도 안 오름(오히려 Hit@1 약간 하락). MTEB-ko 성적이 우리 작은 gold엔 전이 안 됨. → **임베딩 교체는 병목의 답 아님. BGE-M3 유지.**

**부수 발견:** paraphrase는 dense-only(R@5 0.74) > 하이브리드(0.69). sparse+RRF가 어휘 유사 오답을 끌어와 정답을 top-5 밖으로 밀어냄 → RRF 가중치 조정 여지(추후).

## 5. 평가 중 해결한 이슈

- **PDF 표 공백 붕괴** (`편도3,000원`): Docling `do_cell_matching=True`(기본)가 한글 표 셀 공백 소실. `chunking.py`에서 **`do_cell_matching=False`**로 해결. §9.3 운영 블로커 동시 해소. PDF 7종 재인제스천.
- **resolve stale 27→1**: 위 파서 수정 후 gold 안정 키가 정상 매칭. gold·corpus 미변경(v1 유효).

## 6. §8 기준 대조 (검색 한정)

| 기준 | 목표 | 현재(BGE) | 판정 |
|---|---|---|---|
| Recall@5 | ≥ 0.90 | single 0.90 / rare 0.91 통과, **paraphrase 0.69 / multi_doc 0.75 미달** | △ |
| Hit@1 | ≥ 0.70 | 전 subset 미달(0.15~0.63) | ✗ |

→ 현 상태로는 검색 기준 미달. 특히 Hit@1(순서)·paraphrase recall.

## 7. 1.5.C — 리랭커 A/B (bge vs bge+rerank)

`BAAI/bge-reranker-v2-m3`(cross-encoder)로 top-20 재정렬 후 top-5. sparse·후보집합 고정.

| type | R@5 | Hit@1 | MRR |
|---|---|---|---|
| single_fact | 0.90 → 0.97 | 0.63 → 0.67 | 0.76 → 0.80 |
| paraphrase | 0.69 → 0.77 | 0.37 → 0.54 | 0.52 → 0.66 |
| rare_lexical | 0.91 → 1.00 | 0.60 → 0.97 | 0.73 → 0.99 |
| multi_doc | 0.75 → 0.90 | 0.15 → 0.55 | 0.45 → 0.69 |

**판정: 유의미한 승리** (KURE 동률과 대조). 4 subset 전부 큰 상승, R@20 불변(=순서만 정확해짐). 특히 rare_lexical 거의 완성(Hit@1 0.97).

**역할 분담 (핵심):** **top-20 후보 recall(R@20≈0.91)은 검색기 영역이고 충분히 이상적. 그 0.91을 100% top-5로 끌어올리는 게 리랭커 영역.** 현재 paraphrase는 0.91 중 0.77까지 끌어올림 → 0.77~0.91 구간이 리랭커 추가 최적화 여지, 0.91 위는 검색기(후보) 영역.

**비용:** latency 176ms → 8121ms (CPU 인프로세스, cross-encoder 20쌍). **GPU/TEI 재측정 필요** (추정 수백 ms, p95 4s 내 예상).

**결정:** GPU/TEI 배포 전제로 **리랭커 채택**. latency만 운영 연결(F.2) 때 실측 확정.

### 리랭커 모델 -ko 비교 (기각)

한국어 파인튜닝 `dragonkue/bge-reranker-v2-m3-ko`를 base와 비교 → **전 subset에서 base 이하**(rare_lexical Hit@1 0.97→0.86, multi_doc 0.55→0.35, paraphrase R@5 동일 0.77). KURE(임베딩)와 같은 패턴 — 한국어 파인튜닝이 우리 gold엔 불리. → **base bge-reranker-v2-m3 확정.** 리랭커 *모델 교체*로 paraphrase 0.77 갭은 안 메워짐(잔여는 top_k 확대·후보 recall 영역).

## 7c. 1.5.E — 근거 게이트 (거리 게이트 비활성 결정)

`apply_gate`(top-1 dense 거리 임계값)가 무근거 질의를 거르는지 평가 (LLM 불필요, `eval/gate.py`).

| th | 오거부(근거O 거부) | 거절정확도(근거X 거부) |
|---|---|---|
| 0.5 | 14% | 45% |
| **0.6 (구 기본)** | 2% | **5%** |
| 0.7 | 0% | 0% |

**결과:** dense 거리 분포가 근거 유/무에서 겹쳐 **깨끗한 임계값이 없음.** 0.6은 무근거를 5%만 거르고(사실상 off), 0.5는 정상 질의 14% 오거부(특히 rare_lexical 40% — sparse로 찾는데 게이트가 dense 거리만 봐서 오판). trap은 어느 값이든 못 거름(유사 청크를 실제로 가져옴 → 게이트 영역 아님).

**결정:** 대상이 **상담원 어시스턴트**라 잡담·무관·악의 질의가 드물어 거리 게이트 실익이 낮고 오거부 리스크가 큼 → **거리 게이트 비활성**(`retrieve()` 기본 `max_dense_distance=inf`, `no_results`만 유지). `apply_gate` 코드는 보존(가역). 무근거 환각 방어는 **LLM 프롬프트("확인 불가")·groundedness**가 담당 → 1.5.E 출력 평가에서 검증. 외부 공개/잡담 환경 시 0.5~0.6으로 복귀.

## 8. 미결 / 주의

- 생성(1.5.D)·출력 안전성(1.5.E)·운영지표(1.5.F) 미평가 → 모델 채택(§8) 결정은 이후.
- **무근거 방어가 LLM 단독 의존**(거리 게이트 비활성) → 1.5.E에서 "LLM이 trap/no_evidence에 환각 없이 확인불가 하나" 반드시 검증.
- multi_doc 지표는 "정답 여럿 중 하나만 1등" 구조라 참고값(결론 아님).
- gold 라벨 미세 보정 잔존: mt001 등 prose 군더더기 공백 snippet(검색엔 무관, 생성 평가 전 보정).
- KURE/-ko 비교 산출물은 `eval/_kure.py` + `eval_kure_dense` 테이블에 격리(운영 무영향). 재실험 불필요.
