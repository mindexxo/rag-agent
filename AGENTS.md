# AGENTS.md — 상담도우미(KMS) 백엔드 지도

에이전트용 벤더 중립 지도. **백과사전이 아니라 목차**다 — 반드시·절대 규칙은 `INVARIANTS.md`(강제 테스트와 함께),
동작은 코드에 있고, 이 파일은 "어디를 보라"만 가리킨다.

이 파일은 에이전트 컨텍스트에 상시 주입된다. 그래서 두 가지 규율이 있다:
문장은 전부 검증된 사실이어야 하고(근거 없는 문장을 넣지 마라), 낡은 문장은 고치기 전에 지워라
— 틀린 채로 두는 것이 없는 것보다 나쁘다. 여기 적힌 경로·심볼·숫자는
`tests/test_docs_freshness.py`가 기계로 검증한다.

## 아키텍처 — leaf → 조립점 → 진입점

의존은 아래에서 위로만 흐른다. 계층은 `rag/` 톱레벨 import로 실측한 것이고, `tests/test_architecture.py`가 이 절의 문장을 코드와 대조한다
(모든 모듈이 층에 있는지·0층 예외·진입점·지연 import 목록·순환).

- **0층(leaf, 다른 `rag/` 모듈을 안 씀)**: `config.py`, `rag/models.py`, `rag/tokens.py`,
  `rag/prompt_texts.py`, `rag/llm/`(공급자별 구현 — `base.py`·`openai_compat.py`·`vllm.py`·`litellm.py`, #287·#288), `rag/embeddings.py`, `rag/chunking/`(청킹 패키지 — `types`·`tables`·`errors`·`plain`·`sections`·`split`·`document_converter` 하위 모듈과 본체 `__init__`),
  `rag/index_text.py`, `rag/lexical.py`, `rag/limiter.py`, `rag/otel.py`, `rag/metrics.py`,
  `rag/faq_indexing.py`, `rag/retrieval_types.py`, `rag/identifiers.py`, `rag/os_client.py`(톱레벨은 `config`만 —
  `opensearchpy`는 `client()` 안에서 지연 import한다: 매핑 상수만 보는 테스트가 그 패키지 없이도 import할 수 있게).
  예외 하나: `rag/chunking/`는 0층 둘(`rag/tokens.py`·`rag/identifiers.py`)만 쓴다 — 조 제목 판정을 색인 이름표와
  같은 함수로 하려고(#233)
- **중간층(조합)**: `rag/citation_labels.py`, `rag/clients.py`, `rag/llm_schemas.py`,
  `rag/turn_state.py`, `rag/cache.py`, `rag/reranker.py`, `rag/retriever.py`,
  `rag/stream_resume.py`, `rag/cancellation.py`, `rag/citation_tail.py`, `rag/documents.py`, `rag/ingestion.py`,
  `rag/prepared.py`,
  `rag/prompts.py`, `rag/conversation.py`, `rag/guardrail.py`, `rag/outbox.py`,
  `rag/os_search.py`, `rag/os_index.py`, `rag/os_reconcile.py`, `rag/xlsx_chunking.py`(톱레벨에서 `chunking`을 쓴다 —
  그래서 `chunking`이 함수 안에서 되부른다), `rag/conversation_search.py`(`models`만 쓴다)
- **조립점**: `rag/service.py` — 한 턴의 수명(prepare → generate → finalize)을 조율한다.
- **진입점**(다른 모듈이 톱레벨에서 import하지 않는다 — 예외는 아래 지연 목록의 `stream_resume`→`streaming`): `rag/streaming.py`(SSE),
  `rag/worker.py`(arq 백그라운드), `routers/*.py`(HTTP), `main.py`(FastAPI 부트스트랩 전용).

절대 규칙 둘:
- `rag/`는 `routers/`·`eval/`을 import하지 않는다(역방향 금지).
- `config.py`는 프로젝트 모듈을 import하지 않는다.

**함수 안 지연 import은 순환 회피용이며 의도된 것이다 — 톱레벨로 끌어올리지 마라.**
이걸 하는 모듈: `chunking`(→xlsx_chunking), `embeddings`·`limiter`(→clients),
`reranker`(→clients, embeddings), `retriever`(→reranker), `stream_resume`(→clients, streaming),
`outbox`(→ingestion).
순환 회피가 아닌 지연 import은 둘이다. `opensearchpy` — `rag/` 어느 모듈도 톱레벨에 들이지 않고
`os_client.client()`·`outbox.is_transient()`가 함수 안에서 import한다(매핑 상수만 보는 테스트가 그 패키지
없이 `os_client`를 import할 수 있게 한 규율을 outbox도 따른다). `litellm` — `rag/llm/litellm.py`·`rag/embeddings.py`·`rag/reranker.py`의 상용 분기 함수 안에서만
import한다(118MB 패키지라 사내 vLLM·TEI만 쓰는 프로세스가 비용을 안 내게, #288·#289).

검색 저장소(`os_*` 4모듈)에는 **지연 import가 없다**(#146). 구 rag/opensearch.py는 19~21곳을
함수 안에 두고 있었는데, 그중 진짜 순환은 둘뿐이었다 — `retriever`(자료형)와 `outbox`(재등재).
자료형을 `retrieval_types.py`로 빼고 재동기화를 `os_reconcile.py`로 떼자 둘 다 사라졌고, 나머지는
애초에 "한 파일을 0층으로 유지한다"는 자기 규율이었을 뿐이라 톱레벨로 올라갔다.
개수는 적지 않는다 — 정확한 목록은 `grep -rn "^\s\+from rag" rag/`로 뽑는다(톱레벨 import는
줄 시작이 들여쓰기 없음이라 이렇게 구분된다).

## 실행

- 서버: `uvicorn main:app --reload --port 8000`
- 워커(문서 업로드·정리 작업 시 필요): `arq rag.worker.WorkerSettings`
- OpenSearch(검색 저장소 — 청크는 여기에만 있다 #139): `.env.dev` 기본은 개발계(worker20) 인덱스. 로컬 컨테이너를
  쓰려면 `docker compose -f docker-compose.opensearch.yml up -d` 후 `.env`에서 `OPENSEARCH_URL` 오버라이드.
  못 붙어도 서버·워커는 기동되지만(ERROR 로그) 검색·색인이 안 된다.
- 색인 정합 복구: `python -m rag.os_reconcile`(대조만) / `--apply`(재등재·삭제). `eval/`이 아니라
  `rag/` 아래인 이유는 도커 이미지가 `eval/`을 빼기 때문이다 — 운영 복구 명령은 운영 이미지 안에 있어야 한다(#146).
- 테스트: `pytest` — 전체 십수 분, 절반이 docling 변환 테스트(LLM·임베딩은 fake로 대체).
  DB·Redis·OpenSearch는 **로컬 컨테이너만** 쓴다(`docker-compose.test.yml`, 도커 필요) — 루트 `conftest.py`가 `.env`의
  개발계 주소를 로컬 포트로 덮고, 세션 시작마다 `up -d --wait`로 헬스체크까지 기다린다(떠 있으면 약 1초, 첫 기동은 OpenSearch 이미지 빌드).
  데이터는 tmpfs라 `docker compose -f docker-compose.test.yml down` 후 다시 띄우면 빈 스키마로 돌아온다.
  DB 없이 순수 로직만 몇 초 만에 돌리려면 `pytest tests/test_service_pure.py tests/test_prompts.py
  tests/test_turn_status_contract.py tests/test_docs_freshness.py tests/test_code_quality.py
  tests/test_prompt_bytes.py`. 리팩토링의 동작 보존 증거는 프롬프트 = `tests/test_prompt_bytes.py`(모델에 가는 문자열 해시),
  청킹 = `python -m eval._golden_chunks`, 검색 순서 = `python -m eval._golden_order`(둘 다 전후 diff).
  (건수는 적지 않는다 — 테스트를 추가할 때마다 어긋나고, 린터가 잡아주지 못하는 종류다.)
- eval: `python -m eval.run_all --intent --condense --refusal --cache --other --retrieval --ragas`
  축마다 의존이 다르다 — `--retrieval`은 DB+OpenSearch+TEI, `--cache`는 임베딩·리랭커(TEI)+DB+LLM
  (검색이 rerank를 태우고, 히트 판정이 LLM 재사용 심사를 거친다 #113),
  `--refusal`은 DB+LLM+TEI(`prepare()`가 검색을 태우므로 TEI가 필요하다 — 실수하기 쉽다),
  `--other`는 DB+LLM(OTHER 라우팅은 검색을 건너뛴다), `--intent`·`--condense`는 LLM.
  `--ragas`는 **미리 생성된 답변만 채점**한다(생성은 `python -m eval.generation`, 무겁다).

## 환경변수

우선순위: OS 환경변수 > `.env`(gitignore, 로컬 전용) > `.env.dev`(커밋됨, 개발계 공용).
없으면 무엇이 죽는지: `DATABASE_URL`→기동 즉시 실패, `OPENSEARCH_URL`(연결 불가)→기동은 되지만 검색·색인이
호출 시점에 실패(ERROR 로그), Redis→limiter·cancellation·스트림 재개, vLLM→인텐트·질의재작성·생성·캐시 재사용 판정,
TEI→인덱싱·검색·캐시(`EMBED_PROVIDER`·`RERANK_PROVIDER`가 tei가 아니면 그 자리에 상용 API·키, #289).
**비밀값은 이 파일에 적지 않는다** — `config.py`의 필드명만 참조하라.

## 이름이 비슷해서 헷갈리는 것 — 절대 동일시하지 말 것

이 목록은 실제로 혼동이 났던 것만 모았다. "왜 다른가"는 각 정의점 파일에 있다.

- `Document.status`(6값: pending|parsing|embedding|ready|failed|deleted)
  ≠ `Message.status`(TurnStatus 5값) — 둘 다 `rag/models.py`.
- SSE 이벤트명 `EVENT_DONE`(`'done'`) ≠ `TurnStatus.DONE`. 어휘는 같지만 다른 축이다.
- `PreparedRag.route`의 `'blocked'`(소문자 문자열) ≠ `TurnStatus.BLOCKED`.
- **인용 5형제 — 전부 다른 것**: `Message.sources`(실제 인용된 출처 — 생성은 꼬리에서 확정, 캐시 히트는 복원, 그 외 빈 목록) /
  `Message.cited_docs`(실인용 **문서** 파일명 — FAQ 항목 인용은 `Message.cited_faq_ids`에 id로, #278) /
  `PreparedRag.sources`(인메모리 후보)
  / `ConversationMessage.citations`(API 응답 필터 결과) / SSE `done.citations`(생성 시점 재계산).
  다섯이 네 파일에 흩어져 있고(`rag/models.py`·`rag/prepared.py`·`schemas/conversations.py`·
  `rag/streaming.py`) 이 구분 자체를 설명하는 문장은 코드에 없다 — 여기가 유일한 설명이다.
  헷갈리면 각 정의 위치를 직접 열어 확인하라. (별개로 `rag/citation_labels.py`는 "인용 번호
  순서"의 정의점이다 — 이름이 비슷하지만 다른 문제를 다룬다.)
- `RetrievalResult.no_evidence`(원시 판정) ≠ `PreparedRag.no_evidence`(첨부 보정 후 실제 판정).
- SSE `done`의 finish_reason(TurnStatus 어휘 — `rag/streaming.py`의 `TurnResult`)
  ≠ LLM 레벨 finish_reason(`stop`/`length`/`other` 공통 어휘 — `RagService.last_finish_reason`, SSE로
  나가지 않고 지표 `kms_llm_finish_reason_total`·스팬 속성 전용, #129). 이름이 같지만 다른 축.
- `Message.intent`(DB 저장값) ≠ `RouteDecision.intent`(LLM 출력, RETRY는 저장 전 소멸)
  ≠ `PreparedRag.intent_label`(파생).
- 질의 3형제: `request.query` ≠ `original_query`(RETRY면 직전 실질 질문으로 치환) ≠ `standalone_query`(재작성).
- `schemas/query.py`의 `QueryRequest`·`QueryResponse`는 **죽은 레거시**다(참조처 0). 새 코드에서
  쓰지 마라. 실제로 쓰는 건 `schemas/kms.py`의 `KmsQueryRequest`.

## 정의점·절대규칙은 grep으로 찾아라

표로 박아두면 이 문서가 낡는다. 항상 최신인 방법을 쓴다:

- 정의점: `grep -rn "정의점" rag/ routers/ schemas/ eval/*.py`
- 절대규칙: `grep -rn "금지\|말 것" rag/ eval/*.py`
  — 밀도 최상위는 `rag/prompt_texts.py`(모델에게 거는 규칙이라 성격이 다르다).

불변식 대장 `INVARIANTS.md`는 예외로 표다 — 규칙 한 줄과 그것을 강제하는 테스트의 연결이라, 낡으면
`tests/test_invariants_registry.py`가 깨진다(#312). 강제 칸이 `—`인 것이 할 일이다.

## 아직 없는 것 / 확인 불가한 것 — 지어내지 마라

- 도구 호출(파일 읽기·쓰기)은 스콥 밖이다. 이 시스템은 검색+생성만 한다.
- `§`·`1.5.X` 같은 절 번호는 `docs_internal/`(gitignore된 기획 문서)을 가리킨다. 그 디렉터리가
  없는 워크트리에서 이런 참조를 만나면 **"확인 불가"로 보고하라** — 존재나 내용을 추측하지 마라.
- 여러 세션이 같은 워킹트리를 공유할 수 있다. 커밋 전 `git status`로 남의 변경이 섞이지 않았는지
  확인하라. 독립 작업이 필요하면 `git worktree`를 쓴다.

## 데이터·평가

**Q&A 측정 축이 읽는 골드는 `eval/gold_set_v2.jsonl` 하나뿐이다**(no_evidence 58 / trap 50). 축 전용 질문셋(intent·cache·condense 등)은
골드가 아니라 축마다 따로다. 옛 v1(`eval/gold_set_v1.jsonl`)은 레거시 실행 경로 둘만 연다(정리는 #306, 강제는 INV-82).
예외 하나: 사내 실문서 골드는 원문 발췌라 공개 저장소에 올리지 않고 gitignore된 eval/gold_private 디렉터리에
두며, `eval/generation.py`의 `load_gold`가 정본과 병합한다 — 검색축 `eval/retrieval_v2.py`가 이걸 읽는다.
질문셋 파일이 아예 없는 축도 있다 — `eval/retrieval_article.py`(조 번호, #238)·`eval/retrieval_keyword.py`(상담원식 키워드
질의, #248)는 실행할 때마다 색인에서 질문·정답을 만든다(공용부 `eval/axis_common.py`, 시드 고정).
테넌트별 분할 `eval/gold_v2/*.jsonl`은 구축용 초안이며 어느 측정 축도 읽지 않는다 —
읽는 것은 `eval/validate_gold_v2.py` 하나다. 그래서 둘은 어긋날 수 있고 실제로 어긋나 있다
(#88 실측 15건, 전부 `expected_docs`·`expected_chunks`. 추가로 #95의 고난도 90행과 trap `must_not_contain` 백필은 **정본에만 있다** — 분할본에는 없는 것이 정상이다). **gold를 고칠 때는 정본을 고쳐라.**

`python -m eval.validate_gold_v2`의 검사 범위를 오해하지 마라 — 분할본의 snippet이 테넌트 원본
문서에 실재하는지를 볼 뿐, 정본과 값이 같은지는 비교하지 않는다. 그래서 정본↔분할본 드리프트는
이 검증기로 잡히지 않는다.

측정 결과를 인용할 때는 **어느 커밋·어느 축인지 함께** 적는다.

## 작업 흐름 — 단계를 건너뛰지 마라

**이슈 → 브랜치(워크트리) → 피처데브 → E2E 테스트케이스 리스트업 → 테스트 전환 → PR**

- **이슈 먼저**. 무엇을 왜 하는지 이슈에 적고 시작한다. 분석 결과가 이슈 본문과 다르면
  이슈를 먼저 최신화한다 — 낡은 이슈를 근거로 설계하지 마라.
- **브랜치는 `git worktree`로 딴다**. 다른 세션이 같은 워킹트리에서 동시에 작업 중일 수 있고,
  워크트리를 쓰면 서로의 변경에 영향받지 않는다. 워크트리 안에서 작업할 때 원본 저장소 경로의
  파일은 절대 건드리지 마라.
- **피처데브 단계를 생략하지 않는다**(탐색 → 질문 → 설계 → 구현 → 리뷰). "간단해 보인다"는
  생략 사유가 아니다 — 이 저장소에서 설계 초안의 사실 오류가 실제로 여러 번 나왔다.
- **피처데브 다음은 E2E 테스트케이스 리스트업이다.** 자동 테스트가 통과했다고 끝이 아니다.
  사람이 실제로 눌러봐야 하는 케이스를 전부 목록으로 만든다.
- **그 목록에서 테스트 코드로 옮길 수 있는 것은 PR 전에 옮겨 실제로 돌린다.** 항목마다
  단위/통합 중 무엇으로 가능한지 판정하고, 짜서 돌린 뒤 목록을 갱신한다. 환경이 없어 스킵된 채
  "단위 통과"만 보고 PR로 가지 마라. 수동으로 남기는 항목은 남기는 이유가 있어야 한다
  (외부 장애 주입·실제 모델 응답 품질·부하·UI 육안 확인처럼 프로세스 안에서 재현 안 되는 것).
- **PR은 마지막이다.** 갱신된 E2E 목록(테스트로 옮긴 것과 수동으로 남긴 것을 구분해)을 PR 본문에
  싣는다. 머지는 사용자 지시로만.

측정·검증 규율: 프롬프트나 설정을 바꿔 측정할 때는 **바뀐 결과물을 값으로 검증한 뒤** 측정을
시작한다(조립이 틀린 채로 측정한 사고가 있었다). 수치만 보고하지 말고 실제 생성물을 파일로
떨궈 사람이 직접 읽을 수 있게 한다.

## 규약

리팩토링 6원칙을 따른다(특히 원칙 2: 동작 보존 — 리팩토링에 기능 변경을 섞지 마라).
**주석·docstring은 코드만 보고 빠르게 이해하기 어려운 함수·동작에 한해 간결하게** 쓴다(비자명한 순서·동시성·외부 시스템 제약 등).
히스토리성 — 이슈 번호·날짜·실측 경위·기각한 안·"전엔 이랬다" — 은 쓰지 않는다. 그건 이슈·PR·커밋에 있다. 코드를 그대로
다시 말하는 설명도 쓰지 않는다. 히스토리 표지는 `tests/test_code_quality.py`가 파일별로 래칫한다(새 파일 0, 기존은 늘지 않게).
기존 주석은 리팩토링(#304~#307)에서 이 기준으로 정리한다. 문서·주석도 코드와 같은 리뷰 대상이다. 한국어로 쓴다.

## 코드 품질 기준 — 래칫으로 강제한다(#302)

강제는 `tests/test_code_quality.py`(설정 `pyproject.toml`, 기준선 `tests/quality_baseline.json`)가 한다. 지금의 위반은
기준선에 동결돼 있고 **그보다 나빠지면 실패**한다(줄면 경고 — 정리한 PR이 `QUALITY_BASELINE_UPDATE=1 pytest
tests/test_code_quality.py`로 조인다). ruff는 `requirements.txt`에 버전 고정 — CI가 없으니(#90) 커밋 전에 돌린다.
불가피한 증가·이름 변경·기준선 충돌 처리는 그 파일 docstring.

1. **자동 강제(위반 0이 목표)**: Ruff 포매터, import 정렬, 미사용 코드, 버그 패턴. 새 파일은 처음부터 지킨다
   (`ruff format <파일>`, `ruff check --fix <파일>`). 기존분 일괄 정리는 #303.
2. **리뷰 신호(기존분은 동결, 새로 넘으면 막힌다 — 책임을 나눌 신호)**: 복잡도 10, 분기 12, 문장 50, 인자 5, 중첩 4, 함수 코드줄 50, 모듈 코드줄 300
   (코드줄 = 주석·docstring·빈 줄 제외). 쪼개는 것이 목적이 아니라 책임이 하나인지가 목적이다 — 잘게 흩어진 조각도 나쁜 구조다.
3. **데이터**: 고정 키 dict가 함수 경계를 넘으면 `dataclass`·`NamedTuple`. 외부 JSON 경계(골드 파일·OpenSearch 응답)는 `TypedDict`.
4. **클래스**: 상태와 불변식이 있을 때만. 상속 대신 조합, 인터페이스는 `Protocol`(#52 — 클래스를 모듈 함수로 되돌린 선례).
5. **타입**: 함수 시그니처에 힌트 — 기존 파일까지 ruff ANN 래칫이 막는다(인자·가변 인자·공개/특수/static/class 메서드 반환). **새 파일은 mypy 오류 0**(`tests/test_typecheck.py`,
   기존 파일은 `tests/typecheck_baseline.json`에 동결 — 리팩토링으로 깨끗해지면 갱신 명령이 목록에서 뺀다, #308).

6. **모듈 경계**(#310) — 공식 표준은 없고 실무 기준 넷이다. ① 같이 바뀌는 것끼리 묶는다(기능 단위 — 형식별·계층별 서랍이 아니라)
   ② 의존은 한 방향(위 "아키텍처" 절 — `tests/test_architecture.py`가 강제) ③ 밖에서 쓰는 이름만 공개, 다른 모듈의 `_이름`을
   가져다 쓰면 경계가 새는 것(래칫) ④ 모듈은 하는 일을 한 문장으로 말할 수 있어야 한다(모듈 docstring — 래칫). 크기는 결과다.
   하위 패키지는 같은 접두어 모듈이 여럿이고 그 밖으로 나가는 import가 적을 때만 만든다(#286 실측: 지금은 아니다).

3·4·6①은 기계 검사가 없다(오탐이 많아 #90이 경고한 실패 모드) — 리뷰에서 본다.

7. **에이전트 가드레일**(#311): `.claude/settings.json`(커밋됨) deny가 경로 없는 전체 스테이징·강제 push·main 직접 push의
   **흔한 형태**를 막는다(보안 경계 아님 — 못 막는 형태는 `tests/test_guardrails.py` docstring). 그 테스트가 deny가 빠지지 않았는지·
   평소 명령을 잘못 막지 않는지, 커밋된 설정(`.env*`·`docker-compose*`·`config.py`)에 평문 비밀값이 없는지 본다(공개 저장소).
   개인 allow는 미추적 `.claude/settings.local.json`. PR 테스트 수치는 `tests/evidence.py` 출력을 붙인다.
