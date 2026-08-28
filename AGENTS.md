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
- **중간층(조합)**: `rag/citation_labels.py`, `rag/clients.py`, `rag/llm_schemas.py`,
  `rag/turn_state.py`, `rag/cache.py`, `rag/reranker.py`, `rag/retriever.py`,
  `rag/stream_resume.py`, `rag/cancellation.py`, `rag/citation_tail.py`, `rag/documents.py`,
  `rag/prompts.py`, `rag/conversation.py`, `rag/guardrail.py`
- **조립점**: `rag/service.py` — 한 턴의 수명(prepare → generate → finalize)을 조율한다.
- **진입점**(아무도 이들을 import하지 않는다): `rag/streaming.py`(SSE),
  `rag/worker.py`(arq 백그라운드), `routers/*.py`(HTTP), `main.py`(FastAPI 부트스트랩 전용).

절대 규칙 둘:
- `rag/`는 `routers/`·`eval/`을 import하지 않는다(역방향 금지).
- `config.py`는 아무것도 import하지 않는다.

**함수 안 지연 import은 순환 회피용이며 의도된 것이다 — 톱레벨로 끌어올리지 마라.**
이걸 하는 모듈: `chunking`(→xlsx_chunking), `embeddings`·`limiter`(→clients),
`reranker`(→clients, embeddings), `retriever`(→reranker), `stream_resume`(→clients, streaming).
개수는 적지 않는다 — 정확한 목록은 `grep -rn "^\s\+from rag" rag/`로 뽑는다(톱레벨 import는
줄 시작이 들여쓰기 없음이라 이렇게 구분된다).

## 실행

- 서버: `uvicorn main:app --reload --port 8000`
- 워커(문서 업로드·정리 작업 시 필요): `arq rag.worker.WorkerSettings`
- 테스트: `pytest` — 전체 약 4분(DB·Redis 필수, LLM·임베딩은 fake로 대체).
  DB 없이 순수 로직만 몇 초 만에 돌리려면 `pytest tests/test_service_pure.py tests/test_prompts.py
  tests/test_turn_status_contract.py tests/test_docs_freshness.py`.
  (건수는 적지 않는다 — 테스트를 추가할 때마다 어긋나고, 린터가 잡아주지 못하는 종류다.)
- eval: `python -m eval.run_all --intent --condense --refusal --cache --other --retrieval --ragas`
  축마다 의존이 다르다 — `--retrieval`은 DB+TEI, `--cache`는 임베딩+DB,
  `--refusal`은 DB+LLM+TEI(`prepare()`가 검색을 태우므로 TEI가 필요하다 — 실수하기 쉽다),
  `--other`는 DB+LLM(OTHER 라우팅은 검색을 건너뛴다), `--intent`·`--condense`는 LLM.
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
  다섯이 네 파일에 흩어져 있고(`rag/models.py`·`rag/service.py`·`schemas/conversations.py`·
  `rag/streaming.py`) 이 구분 자체를 설명하는 문장은 코드에 없다 — 여기가 유일한 설명이다.
  헷갈리면 각 정의 위치를 직접 열어 확인하라. (별개로 `rag/citation_labels.py`는 "인용 번호
  순서"의 정의점이다 — 이름이 비슷하지만 다른 문제를 다룬다.)
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

**모든 측정 축이 읽는 파일은 `eval/gold_set_v2.jsonl` 하나뿐이다**(no_evidence 58 / trap 50).
테넌트별 분할 `eval/gold_v2/*.jsonl`은 구축용 초안이며 어느 측정 축도 읽지 않는다 —
읽는 것은 `eval/validate_gold_v2.py` 하나다. 그래서 둘은 어긋날 수 있고 실제로 어긋나 있다
(#88 실측 15건, 전부 `expected_docs`·`expected_chunks`). **gold를 고칠 때는 정본을 고쳐라.**

`python -m eval.validate_gold_v2`의 검사 범위를 오해하지 마라 — 분할본의 snippet이 테넌트 원본
문서에 실재하는지를 볼 뿐, 정본과 값이 같은지는 비교하지 않는다. 그래서 정본↔분할본 드리프트는
이 검증기로 잡히지 않는다.

측정 결과를 인용할 때는 **어느 커밋·어느 축인지 함께** 적는다.

## 작업 흐름 — 단계를 건너뛰지 마라

**이슈 → 브랜치(워크트리) → 피처데브 → PR → E2E 테스트케이스 리스트업**

- **이슈 먼저**. 무엇을 왜 하는지 이슈에 적고 시작한다. 분석 결과가 이슈 본문과 다르면
  이슈를 먼저 최신화한다 — 낡은 이슈를 근거로 설계하지 마라.
- **브랜치는 `git worktree`로 딴다**. 다른 세션이 같은 워킹트리에서 동시에 작업 중일 수 있고,
  워크트리를 쓰면 서로의 변경에 영향받지 않는다. 워크트리 안에서 작업할 때 원본 저장소 경로의
  파일은 절대 건드리지 마라.
- **피처데브 단계를 생략하지 않는다**(탐색 → 질문 → 설계 → 구현 → 리뷰). "간단해 보인다"는
  생략 사유가 아니다 — 이 저장소에서 설계 초안의 사실 오류가 실제로 여러 번 나왔다.
- **PR 다음은 E2E 테스트케이스 리스트업이다.** 자동 테스트가 통과했다고 끝이 아니다. 사람이
  실제로 눌러봐야 하는 케이스를 목록으로 만들어 넘긴다. 테스트 코드로 덮을 수 있는 케이스가
  섞여 있으면 먼저 테스트로 옮기고, 남는 것만 수동 목록에 남긴다.

측정·검증 규율: 프롬프트나 설정을 바꿔 측정할 때는 **바뀐 결과물을 값으로 검증한 뒤** 측정을
시작한다(조립이 틀린 채로 측정한 사고가 있었다). 수치만 보고하지 말고 실제 생성물을 파일로
떨궈 사람이 직접 읽을 수 있게 한다.

## 규약

리팩토링 6원칙을 따른다(특히 원칙 2: 동작 보존 — 리팩토링에 기능 변경을 섞지 마라).
판단 근거와 실측치는 주석에 남긴다. 문서·주석도 코드와 같은 리뷰 대상이다. 한국어로 쓴다.
