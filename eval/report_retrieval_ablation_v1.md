# 검색 어블레이션 리포트 — sparse 기여도 & DB-BM25 대체 검증 (v1)

> 2026-07-10. 목적: F99(임베딩 분리) 결정을 위해 **"우리 데이터에서 sparse가 얼마나 기여하나 /
> model-sparse를 DB 어휘(BM25)로 대체 가능한가"**를 순수 검색 지표로 측정.
> LLM 판정 없음(RAGAS 아님) · 신규 인프라 없음 · gold_set_v1 검색 subset(n=120).
> 스크립트: scratchpad `ablation_sparse.py`(2-way), `ablation_3way.py`(3-way).

## 방법

- gold: `eval/gold_set_v1.jsonl`의 검색 subset (single_fact·paraphrase·rare_lexical·multi_doc)
- 코퍼스: demo 테넌트 active+ready 문서 + FAQ 청크 (175 청크)
- 채점: `eval/retrieval.py` `run_config`와 동일 — chunk 정답 우선, 없으면 doc 폴백. R@5/R@20/Hit@1/MRR
- 융합: 세 하이브리드 경로 모두 동일 RRF(dense top-30, X top-30) → top-20. **sparse 자리에 X만 교체 = 변수 격리**
- **DB-BM25**: 순수 Python BM25Okapi(k1=1.5, b=0.75), **문자 bigram 토크나이저** = pg_bigm 방식 프리뷰(설치 0)

## 결과 1 — 2-way (hybrid vs dense-only vs sparse-only)

전체 평균(n=120):

| 경로 | R@5 | R@20 | Hit@1 | MRR |
|---|---|---|---|---|
| **dense-only** | **0.82** | **0.99** | 0.42 | 0.58 |
| hybrid (dense+sparse, 현행) | 0.80 | 0.97 | 0.42 | 0.58 |
| sparse-only | 0.73 | 0.91 | 0.42 | 0.57 |

## 결과 2 — 3-way (dense-only vs dense+bm25 vs dense+sparse)

전체 평균(n=120):

| 경로 | R@5 | R@20 | Hit@1 | MRR |
|---|---|---|---|---|
| **dense-only** | **0.82** | 0.99 | 0.42 | 0.58 |
| dense+sparse (현행) | 0.80 | 0.97 | 0.42 | 0.58 |
| dense+bm25 | 0.78 | 0.97 | 0.42 | 0.57 |

타입별 R@5:

| 타입 | dense-only | dense+bm25 | dense+sparse |
|---|---|---|---|
| single_fact | **0.90** | 0.80 | 0.87 |
| paraphrase | **0.77** | 0.60 | 0.63 |
| **rare_lexical** | 0.89 | **0.97** ⭐ | 0.94 |
| multi_doc | 0.65 | 0.70 | **0.75** |

타입별 Hit@1:

| 타입 | dense-only | dense+bm25 | dense+sparse |
|---|---|---|---|
| single_fact | 0.50 | 0.47 | **0.53** |
| paraphrase | 0.31 | **0.34** | 0.31 |
| rare_lexical | **0.57** | 0.51 | **0.57** |
| multi_doc | 0.25 | **0.30** | 0.15 |

## 해석

1. **rare_lexical(정확 어휘)에서 BM25가 셋 중 1등(0.97), model-sparse(0.94)마저 능가.** sparse가 존재하는 유일한 이유(정확 어휘 매칭)를 고전 bigram BM25가 더 잘함.
2. **model-sparse에 고유 가치 없음.** BM25가 그 강점을 재현·초과하고, 나머지(paraphrase·single_fact)는 sparse·bm25 둘 다 dense를 깎음. paraphrase는 sparse가 −0.14, bm25가 −0.17.
3. **dense-only가 전체 최강(R@5 0.82)이자 최단순.** 어휘 융합은 rare_lexical/multi_doc만 살리고 나머지는 희석.
4. 기존 인지("paraphrase는 sparse가 깎는다")를 정량 재현 → 실행 신뢰도 ↑.

## 결정 함의 (F99)

**learned-sparse 모델 락인 해제 확정.** 두 목표 아키텍처가 모두 실측 검증됨:

| 안 | 검증 | 성격 |
|---|---|---|
| **dense-only** | 전체 R@5 최고(0.82) | 최단순. 표준 임베딩 서버(TEI/vLLM) 바로 사용 → **F99 서빙 분리까지 단순화**. rare_lexical/multi_doc 소폭 포기 |
| **dense + DB-BM25** | rare_lexical 최고(0.97) | 정확 어휘 강점 유지, 모델은 dense로 교체자유 |

→ 어느 쪽이든 model-sparse 불필요. **pg_bigm 착수는 "그 후 고민"** — Python bigram-BM25가 이미 sparse를 넘었으니 검색 품질은 재현 예상, 실제 세울지는 답변 품질 검증 후 결정.

## 정직한 한계 & 남은 관문

- **검색 지표지 답변 품질 아님.** R@5는 정답 청크 "존재"만 보고 순위·distractor 효과(LLM이 민감)를 못 잡음. config가 다르면 컨텍스트가 달라져 답이 갈릴 수 있음.
- **마지막 관문 = rare_lexical 생성 답변 비교.** 정확 어휘는 검색↔답변 결합이 타이트하고 상담에서 오답 대가가 큼 → 실제로 어휘 채널을 뺄(운영 flip) 때 이 subset의 faithfulness+정확성만 타깃 확인. (지금 flip 안 하면 미뤄도 무방)
- 타입당 n~30, ±0.03~0.05는 노이즈. 신뢰할 큰 신호는 paraphrase(−0.14~0.17)·rare_lexical BM25 우위·multi_doc 어휘 우위.
- BM25 토크나이저를 bigram으로 고정 — mecab 형태소면 tradeoff가 달라짐(미탐색).

## 결과 3 — rare_lexical 생성 관문 (마지막 관문, 2026-07-10 실행)

검색 지표가 아니라 **실제 답변 품질**로 config 비교. rare_lexical 35문항(전부 has_evidence),
top-5 컨텍스트 주입 → Qwen3-14B 생성 → **EPCov(기대 포인트 반영)+Cite(정답 문서 인용)**, 둘 다 결정론(쿼터 0).
스크립트: scratchpad `gen_gate_rare.py`.

| config | EPCov | Cite |
|---|---|---|
| dense-only | 0.943 | 0.914 |
| dense+bm25 | **1.000** | 0.914 |
| dense+sparse (현행) | 0.971 | **1.000** |

**정확 코드/용어 인용 5건 상세:**

| id | dense-only | dense+bm25 | dense+sparse | 질의 |
|---|---|---|---|---|
| rl001 | ✅ | ❌ | ✅ | RF-01 코드 |
| rl004 | ❌ | ❌ | ✅ | SC-01 |
| rl023 | ✅ | ❌ | ✅ | 구매 확정 |
| rl024 | ❌ | ✅ | ✅ | KMS-SEC-001 |
| rl032 | ❌ | ✅ | ✅ | KMS-PNT-009 |

**→ model-sparse만 5/5 완벽. dense-only·bm25 각각 3개 놓침(서로 다른 것들).**

### 관문 초기 해석 → 실제 데이터 육안 검증으로 정정 (2026-07-10)

Cite 표(sparse 5/5, dense·bm25 각 3 miss)만 보고 처음엔 "model-sparse만 정확 코드에 완벽,
bigram-bm25는 대체 불가"로 결론냈으나, **실제 top-5 청크·답변을 눈으로 대조하니 대부분 아티팩트였음**
(scratchpad `peek_retrieval.py`). gold 라벨 대조 결과 SC-01·RF-01은 여러 문서(용어집+상담_코드표.xlsx)에 걸침.

- **rl024(KMS-SEC-001) = 유일한 진짜 신호**: gold 청크(비밀번호재설정.pdf)가 **dense-only top-5에 없음**(검색 실패). dense+bm25·dense+sparse는 **둘 다 rank2에 정확히** 잡음. → 정확 코드엔 어휘 채널 필요. **단 bm25·sparse 동일 성공.**
- **rl004(SC-01) = 순수 아티팩트**: 세 config 모두 rank1=상담_코드표.xlsx, rank2=gold(용어집)로 **검색 동일**. 차이는 LLM이 어느 걸 인용했나뿐 — 검색 품질 아님. rl001도 동형(RF-01 다문서).
- **pp002(paraphrase)**: 셋 다 rank1=gold, 차이 없음.

**정정된 견고한 결론:**
1. **유일하게 견고한 신호: dense-only는 정확 코드를 under-retrieve** (rl024 gold를 top-5에도 못 넣음) → **어휘 채널은 있어야 함.**
2. **sparse vs bm25는 이 eval로 구분 불가** — 유일 하드케이스에서 둘 다 성공. 앞선 "bm25 코드 실패"는 아티팩트(검색 동일, LLM 인용 선택 차이)였음.
3. **model-sparse "5/5"는 부분적으로 LLM 인용운** (다문서 코드에서 gold-라벨 문서를 우연히 인용).
4. EPCov는 세 config 사실상 동률 — 팩트는 다 맞힘.

## 문항별 육안 재검증 (2026-07-10, `verify_all.py`)

집계 숫자가 실제 문항별 1/0에서 정확히 재구성됨(리포트 표 = 진짜). 패턴:
- **dense-only가 지는 곳(dense=0, 어휘=1) 총 9건**: rare_lexical 4(rl014·024·030·032, 정확코드), multi_doc 3(md007·017·020), paraphrase 2(pp009·017).
- **dense-only가 이기는 곳(dense=1, 어휘=0)은 더 많음**: paraphrase 대량(pp001·004·006·020·029·032…), single_fact(sf012·022·025). 어휘가 자연어 질문을 깎음.
- 대칭·상쇄 → 전체는 dense-only 최고(0.82). 강점 재분배지 품질 손해 아님.

**하이브리드가 dense를 못 이긴 것 = 버그 아님**: RRF는 순위 평균이라 약한 채널(sparse)이 강한 dense를 끌어내림. **hybrid 0.80이 dense 0.82·sparse 0.73 사이에 보간**된 게 정상 작동의 지문. RRF 코드(k=60 표준) 정상. 정설 "하이브리드>dense"는 크고 지저분한 코퍼스+약한 dense+**리랭커 얹은** 경우 — 우리(작고 깨끗+BGE-M3 SOTA+paraphrase 다수+생 RRF)엔 불성립.

## 최종 결정 (2026-07-10): dense-only 전환

- **dense-only 채택.** 전체 최고 + 자연어 질문 최강. 약점(정확코드·multi_doc, 9/120)은 좁고, 필요 시 후속 보완.
- **sparse 제거**: 코드 주석(청사진 보존), **DB `chunks.sparse` 컬럼 실제 DROP**(인덱스 동반). 인제스트 3경로 sparse write 주석. 원복=주석복원+컬럼재생성+전체 재인제스트.
- **미래 하이브리드 = BM25(pg_bigm, 텍스트 기반)**, model-sparse 아님. 재도입 시 **가중 RRF/리랭커 필수**(생 RRF는 dense 못 이김).
- **서빙: dense-only 서버 지금 구축** — 표준 TEI/vLLM 임베딩 서버 사용(자체 sparse 서비스 불필요 = dense-only의 이득). ⚠️ **호환성**: 저장된 chunks.dense ↔ 서버 query dense가 동일 모델·서빙에서 나와야 함. 소코퍼스라 **재인제스트로 일치 보장** 권장.
- 한계: 코드질의 n=5, Cite가 다문서에서 노이즈 — 세부 우열 확대해석 금지.

## 결과 4 — 리랭커 4-way (2026-07-10, `verify_rerank.py` + `gen_gate_rerank.py`)

후보 풀 top-20 → 크로스인코더(bge-reranker-v2-m3, 인프로세스) 리랭크 → top-5. 검색지표 전 타입 + rare_lexical 생성 관문.

**검색지표 전체:**
| 지표 | dense-only | dense+rrk | d+bm25+rrk | d+sparse+rrk |
|---|---|---|---|---|
| Recall@5 | 0.82 | **0.89** | 0.88 | 0.88 |
| Hit@1 | 0.42 | **0.64** | 0.66 | 0.64 |
| MRR | 0.56 | **0.74** | 0.75 | 0.74 |

rare_lexical은 리랭크로 사실상 해결: R@5 0.89→**1.00**, Hit@1 0.57→**0.97**, MRR 0.68→**0.99**.

**두 발견:**
1. **리랭커는 sparse/bm25보다 압도적 지렛대** (Hit@1 +0.22). 며칠 논쟁한 어휘 채널의 ±0.02~0.05과 차원이 다름.
2. **리랭커를 붙이면 검색 채널 차이 소멸** — dense+rrk ≈ d+bm25+rrk ≈ d+sparse+rrk. 즉 **리랭커가 있으면 sparse·bm25 불필요**, dense-only+rerank가 하이브리드+rerank 값을 다 담음.

**rare_lexical 생성 관문 (n=35, EPCov+Cite):**
| config | EPCov | Cite |
|---|---|---|
| dense_only | 0.943 | 0.886 |
| dense+rerank | **0.971** | **0.914** |

**육안 검증(갈린 6문항) — 리랭커 순 +3 (4승 1패 1아티팩트):**
- 승 4: rl024(오답→정답), rl027(인용실패→정답), rl030(**환각→정답**), rl032(**거절→정답**)
- 패 1: rl033(정답→**거절** — 리랭커가 정답 청크를 top-5 밖으로 밀어냄, 역회귀 실재)
- 아티팩트 1: rl023(둘 다 정답, 다문서 인용 선택 차이)

**리랭커 결론:**
- **dense-only의 유일 약점(정확 코드)을 리랭커가 대부분 해결** — 환각·거절을 정답으로. 순위개선이 답변개선으로 이어짐 확인.
- 단 **순수 승리 아님**: 재정렬이 정답을 밀어내는 역회귀(rl033) + **매 쿼리 N회 크로스인코더 forward·GPU·지연** 비용. → 자동 채택 아니라 파일럿에서 "품질이득 vs 지연" 측정 후 결정.
- **권장 조합: dense-only(깔끔한 서빙) + 리랭커(품질 지렛대, TEI 2번째 컨테이너).** sparse/bm25는 리랭커가 대체하므로 불필요.
