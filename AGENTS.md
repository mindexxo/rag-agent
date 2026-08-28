# AGENTS.md — 상담도우미(KMS) 백엔드 지도

에이전트용 벤더 중립 지도. **백과사전이 아니라 목차**다 — 상세 규약은 각 파일의 docstring이
정의점이고, 이 파일은 "어디를 보라"만 가리킨다.

이 파일은 에이전트 컨텍스트에 상시 주입된다. 그래서 두 가지 규율이 있다:
문장은 전부 검증된 사실이어야 하고(근거 없는 문장을 넣지 마라), 낡은 문장은 고치기 전에 지워라
— 틀린 채로 두는 것이 없는 것보다 나쁘다. 여기 적힌 경로·심볼·숫자는
`tests/test_docs_freshness.py`가 기계로 검증한다.

## 아키텍처 — leaf → 조립점 → 진입점

의존은 아래에서 위로만 흐른다. 계층은 `rag/` 톱레벨 import로 실측한 것이다.

- **0층(leaf, 다른 `rag/` 모듈을 안 씀)**: `config.py`, `rag/models.py`, `rag/tokens.py`,
  `rag/prompt_texts.py`, `rag/llm.py`, `rag/embeddings.py`, `rag/chunking.py`,
  `rag/index_text.py`, `rag/limiter.py`, `rag/otel.py`
- **1~3층(조합)**: `rag/citation_labels.py`, `rag/clients.py`, `rag/llm_schemas.py`,
  `rag/turn_state.py`, `rag/cache.py`, `rag/reranker.py`, `rag/retriever.py`,
  `rag/stream_resume.py`, `rag/cancellation.py`, `rag/citation_tail.py`, `rag/documents.py`,
  `rag/prompts.py`, `rag/conversation.py`, `rag/guardrail.py`, `rag/worker.py`
- **4층 조립점**: `rag/service.py` — 한 턴의 수명(prepare → generate → finalize)을 조율한다.
- **5층 진입점**: `rag/streaming.py`(SSE). 그 밖의 진입점은 `routers/*.py`(HTTP),
  `rag/worker.py`(arq 백그라운드), `main.py`(FastAPI 부트스트랩 전용).

절대 규칙 둘:
- `rag/`는 `routers/`·`eval/`을 import하지 않는다(역방향 금지).
- `config.py`는 아무것도 import하지 않는다.

함수 안 지연 import은 순환 회피용이며 의도된 것이다 — 톱레벨로 끌어올리지 마라.
현재 6곳: `chunking→xlsx_chunking`, `embeddings/limiter/reranker/stream_resume→clients`,
`retriever→reranker`, `stream_resume→streaming`.

## 실행

- 서버: `uvicorn main:app --reload --port 8000`
- 워커(문서 업로드·정리 작업 시 필요): `arq rag.worker.WorkerSettings`
- 테스트: `pytest` — 전체 528건 3분 47초(DB·Redis 필수, LLM·임베딩은 fake로 대체).
  순수 로직만 빠르게 돌리려면 `pytest tests/test_service_pure.py tests/test_prompts.py
  tests/test_turn_status_contract.py tests/test_eval_condense_trim.py` — 80건 0.6초.
- eval: `python -m eval.run_all --intent --condense --refusal --cache --other --retrieval --ragas`
  축마다 의존이 다르다 — `--retrieval`은 DB+TEI, `--cache`는 임베딩+DB,
  `--refusal`·`--other`는 DB+LLM, `--intent`·`--condense`는 LLM.
  `--ragas`는 **미리 생성된 답변만 채점**한다(생성은 `python -m eval.generation`, 무겁다).

## 환경변수

우선순위: OS 환경변수 > `.env`(gitignore, 로컬 전용) > `.env.dev`(커밋됨, 개발계 공용).
없으면 무엇이 죽는지: `DATABASE_URL`→기동 즉시 실패, Redis→limiter·cancellation·스트림 재개,
vLLM→인텐트·질의재작성·생성, TEI→인덱싱·검색·캐시.
**비밀값은 이 파일에 적지 않는다** — `config.py`의 필드명만 참조하라.

## 이름이 비슷해서 헷갈리는 것 — 절대 동일시하지 말 것

이 목록은 실제로 혼동이 났던 것만 모았다. "왜 다른가"는 각 정의점 파일에 있다.

- `Document.status`(6값: pending|parsing|embedding|ready|failed|deleted)
  ≠ `Message.status`(TurnStatus 5값) — 둘 다 `rag/models.py`.
- SSE 이벤트명 `EVENT_DONE`(`'done'`) ≠ `TurnStatus.DONE`. 어휘는 같지만 다른 축이다.
- `PreparedRag.route`의 `'blocked'`(소문자 문자열) ≠ `TurnStatus.BLOCKED`.
- **인용 5형제 — 전부 다른 것**: `Message.sources`(검색 후보 전체) /
  `Message.cited_docs`(실인용 파일명, 지표는 이 컬럼만 집계) / `PreparedRag.sources`(인메모리 후보)
  / `ConversationMessage.citations`(API 응답 필터 결과) / SSE `done.citations`(생성 시점 재계산).
  정의점: `rag/citation_labels.py`.
- `RetrievalResult.no_evidence`(원시 판정) ≠ `PreparedRag.no_evidence`(첨부 보정 후 실제 판정).
- `Message.intent`(DB 저장값) ≠ `RouteDecision.intent`(LLM 출력, RETRY는 저장 전 소멸)
  ≠ `PreparedRag.intent_label`(파생).
- 질의 4형제: `request.query` ≠ `original_query` ≠ `standalone_query` ≠ `display_query`.
- `schemas/query.py`의 `QueryRequest`·`QueryResponse`는 **죽은 레거시**다(참조처 0). 새 코드에서
  쓰지 마라. 실제로 쓰는 건 `schemas/kms.py`의 `KmsQueryRequest`.

## 정의점·절대규칙은 grep으로 찾아라

표로 박아두면 이 문서가 낡는다. 항상 최신인 방법을 쓴다:

- 정의점: `grep -rn "정의점" rag/ routers/ schemas/ eval/*.py`
- 절대규칙: `grep -rn "금지\|말 것" rag/ eval/*.py`
  — 밀도 최상위는 `rag/prompt_texts.py`(모델에게 거는 규칙이라 성격이 다르다).

## 아직 없는 것 / 확인 불가한 것 — 지어내지 마라

- 도구 호출(파일 읽기·쓰기)은 스콥 밖이다. 이 시스템은 검색+생성만 한다.
- `§`·`1.5.X` 같은 절 번호는 `docs_internal/`(gitignore된 기획 문서)을 가리킨다. 그 디렉터리가
  없는 워크트리에서 이런 참조를 만나면 **"확인 불가"로 보고하라** — 존재나 내용을 추측하지 마라.
- 여러 세션이 같은 워킹트리를 공유할 수 있다. 커밋 전 `git status`로 남의 변경이 섞이지 않았는지
  확인하라. 독립 작업이 필요하면 `git worktree`를 쓴다.

## 데이터·평가

gold 원본은 `eval/gold_set_v2.jsonl`(no_evidence 58 / trap 50). 테넌트별 분할
`eval/gold_v2/*.jsonl`은 수동 동기화라 어긋날 수 있다 — 검증은 `python -m eval.validate_gold_v2`.
측정 결과를 인용할 때는 **어느 커밋·어느 축인지 함께** 적는다.

## 규약

리팩토링 6원칙을 따른다(특히 원칙 2: 동작 보존 — 리팩토링에 기능 변경을 섞지 마라).
판단 근거와 실측치는 주석에 남긴다. 문서·주석도 코드와 같은 리뷰 대상이다. 한국어로 쓴다.
