# 불변식 대장 (Invariants)

"반드시 성립한다"·"절대 일어나면 안 된다"를 한 줄씩 적고, 옆에 **그것을 기계로 강제하는 테스트**를 적는다(#312).
강제 칸이 `—`이면 그게 할 일이다. 설계에서 이런 문장이 나오면 정의점(docstring)에 쓰고 여기에도 한 줄 올린 뒤 테스트를 단다.

- 정의점은 파일만 적는다(줄 번호는 금방 낡는다) — 이유·실측은 그 파일 docstring·주석에 있다.
- 강제 칸의 `tests/…::이름`은 `tests/test_invariants_registry.py`가 실제로 정의돼 있는지 검사한다. 이 표가 낡으면 그 테스트가 깨진다.
  이름이 있다고 의미까지 맞는 것은 아니다 — 등록할 때 테스트 본문이 그 규칙을 단언하는지 읽고 넣는다.
- `(부분)`은 규칙의 일부만 단언한다는 뜻이다 — 빈 부분은 비고에.
- 넣지 않는 것: 모델에게 거는 프롬프트 규칙(`rag/prompt_texts.py`), 작업 절차(AGENTS.md "작업 흐름"), 배경 서술.
- 수집 기준: main `5b4f80a`(2026-10-25). 리뷰에서 행마다 테스트 본문을 읽어 의미를 대조했다(틀린 연결은 빼고 `(부분)`으로 고침), `rag/`·`routers/`·`schemas/`·루트 설정·AGENTS.md의 "절대·반드시·금지" 문장과 "불변·계약"을 단 테스트.

## 격리·보안

| ID | 불변식 | 정의점 | 강제 | 비고 |
|---|---|---|---|---|
| INV-01 | 테넌트 격리는 쿼리 WHERE 절로만 보장한다(RLS 없음) — 모든 테넌트 테이블 조회에 tenant 조건 | `rag/models.py` | `tests/test_integration_isolation.py::test_검색_후보에_타_테넌트_청크가_없다` · `tests/test_integration_isolation.py::test_대화_목록과_메시지_격리` · `tests/test_integration_isolation.py::test_폴더_격리` · `tests/test_integration_isolation.py::test_FAQ_목록_격리` · `tests/test_integration_isolation.py::test_문서_일괄변경_격리` · `tests/test_integration_isolation.py::test_문서_일괄삭제_격리` · `tests/test_integration_isolation.py::test_문서_목록_필터_격리` · `tests/test_integration_documents.py::test_exists_는_테넌트_격리` · `tests/test_integration_documents.py::test_다운로드는_테넌트_격리` · `tests/test_integration_cache.py::test_테넌트_간_캐시_격리` · `tests/test_integration_cache.py::test_FAQ_스냅샷_검증도_테넌트_격리` (부분) | 표면별 테스트 — "모든 조회"를 덮는 정적 강제는 없다(#90 전용 린터 후보). 일괄변경 META는 호출부 사전 스코프에 기댄다(가장 위험한 표면) |
| INV-02 | 없거나 타 테넌트 소유 conversation_id로 질의하면 404 — 404는 대화 없음(`ConversationNotFound`)에만 | `routers/kms.py` | `tests/test_integration_query_sse.py::test_없는_대화_id는_404` · `tests/test_integration_tenant_boundary.py::test_다른_테넌트_대화_id로_질의하면_404이고_어느_쪽에도_행이_생기지_않는다` · `tests/test_integration_query_contract.py::test_대화와_무관한_ValueError는_500이고_턴은_failed로_닫힌다` | |
| INV-03 | 테넌트·검색가능 필터는 엔진에서 건다(post-filter 금지) | `rag/os_search.py` | `tests/test_os_search_pure.py::test_필터는_엔진에서_걸린다` (부분) | 필터 dict 모양만 본다 — 실제 kNN·BM25 질의가 이 필터를 쓰는지는 미검증 |
| INV-04 | 생성 실패 시 예외 원문을 클라이언트에 보내지 않는다 | `rag/streaming.py` | `tests/test_invariants_turn.py::test_INV04_생성_실패의_예외_원문은_SSE에도_재접속_스트림에도_실리지_않는다` |  |
| INV-05 | 커밋된 설정·config 기본값에 평문 비밀값을 두지 않는다 | `AGENTS.md` | `tests/test_guardrails.py::test_커밋된_설정에_평문_비밀값이_없다` · `tests/test_guardrails.py::test_config의_비밀_필드_기본값은_비어_있다` |  |
| INV-06 | 첨부가 있는 대화의 답변은 공용 캐시에 넣지 않는다 | `rag/service.py` | `tests/test_integration_cache.py::test_첨부_대화_답변은_공용_캐시_유출_금지` |  |
| INV-07 | 채팅 첨부 추출은 저장하지 않는다(DB 행·blob 없음) | `routers/attachments.py` | `tests/test_invariants_docs.py::test_INV07_첨부_추출은_DB_행도_blob도_남기지_않는다` |  |
| INV-08 | 테스트는 개발계 공유 인프라(DB·Redis·OpenSearch)에 붙지 않는다 — 로컬 컨테이너 주소로 고정 | `conftest.py` | `tests/test_test_infra.py::test_설정이_보는_인프라_주소는_TEST_INFRA_그대로다` · `tests/test_test_infra.py::test_TEST_INFRA는_로컬이고_compose가_127_0_0_1에_연_포트와_같다` | 세션 픽스처 `_test_infra`가 실행 시점에도 로컬이 아니면 멈춘다. eval은 범위 밖(개발계 색인을 읽는다) |

## 색인·outbox·문서 버전

| ID | 불변식 | 정의점 | 강제 | 비고 |
|---|---|---|---|---|
| INV-10 | outbox 등재는 호출부 트랜잭션에 얹는다 — enqueue는 commit하지 않는다 | `rag/outbox.py` | `tests/test_integration_documents.py::test_409면_blob이_남지_않는다` · `tests/test_integration_index_sync.py::test_재업로드_커밋이_실패하면_구버전은_PG_엔진_모두_그대로다` (부분) | 원자성만 본다 — enqueue의 commit 호출 0은 직접 단언 없음 |
| INV-11 | 파괴적 엔진 변경(구버전 청크 삭제)은 커밋 뒤 outbox DROP 행으로만 | `rag/documents.py` | `tests/test_integration_index_sync.py::test_재업로드_커밋이_실패하면_구버전은_PG_엔진_모두_그대로다` · `tests/test_integration_index_sync.py::test_재업로드_성공은_구버전_DROP_행을_같은_커밋에_남긴다` | |
| INV-12 | outbox 핸들러는 처리 순서를 가정하지 않고 처리 시점 PG 상태로 판단한다 | `rag/outbox.py` | `tests/test_integration_index_sync.py::test_DROP은_되살아난_문서를_건너뛴다` · `tests/test_integration_index_sync.py::test_백오프로_미뤄진_v1보다_v2가_먼저_끝나도_신버전이_이긴다` | |
| INV-13 | outbox payload에는 id만 싣는다 | `rag/outbox.py` | `tests/test_integration_index_sync.py::test_재업로드_성공은_구버전_DROP_행을_같은_커밋에_남긴다` (부분) | DROP만 본다 — INDEX·META 행 미검증 |
| INV-14 | 결정적 실패는 재시도하지 않는다 | `rag/outbox.py` | `tests/test_outbox_pure.py::test_결정적_실패로_분류` · `tests/test_integration_index_sync.py::test_결정적_실패는_META_행도_1회로_확정하고_카운터가_오른다` (부분) | 분류(is_transient)만 본다 — 재시도하지 않음 자체와 XlsxUploadRejected 계열은 미검증 |
| INV-15 | 메타 갱신(META)은 재색인하지 않는다 | `rag/os_index.py` | `tests/test_integration_index_sync.py::test_문서_검색토글은_META_부분갱신으로_제외_복귀한다` | |
| INV-16 | reconcile은 엔진 스냅샷을 PG보다 먼저 읽는다(반대면 방금 색인된 청크를 지운다) | `rag/os_reconcile.py` | `tests/test_invariants_docs.py::test_INV16_reconcile은_두_스냅샷_사이에_ready가_된_문서의_청크를_지우지_않는다` (부분) | 문서 reconcile만 — 같은 계약의 reconcile_faqs·dry-run 대조(_diff)는 미검증 |
| INV-17 | 옛 blob은 호출부가 커밋 뒤에 지운다 | `rag/documents.py` | `tests/test_invariants_docs.py::test_INV17_재업로드_커밋이_실패하면_옛_blob은_디스크에_남는다` · `tests/test_invariants_docs.py::test_INV17_재업로드_커밋이_성공하면_옛_blob을_지운다` | 성공 경로 대조 포함 |
| INV-18 | 업로드 1건 = UUID 파일 1개 — 내용이 같아도 blob을 공유하지 않는다 | `routers/documents.py` | `tests/test_invariants_docs.py::test_INV18_같은_바이트를_여러_번_올려도_업로드마다_blob_파일이_따로_생긴다` |  |
| INV-19 | 재업로드는 설정을 계승하되 등록자는 계승하지 않는다 | `rag/documents.py` | `tests/test_integration_documents_folders.py::test_재업로드_등록자는_계승하지_않는다` | |
| INV-20 | 업로드 파일명은 경계에서 NFC로 정규화하고 이하 전부 그 값을 쓴다 | `routers/documents.py` | `tests/test_filename_unicode.py::test_NFD_파일명_업로드는_NFC로_저장된다` · `tests/test_filename_unicode.py::test_NFD와_NFC_동명_재업로드는_새_버전이_된다` · `tests/test_filename_unicode.py::test_NFD로_올린_문서의_인용은_NFC_파일명으로_잡힌다` | |
| INV-21 | 폴더·xlsx 사전검증은 blob을 쓰기 전에 하고, 실패하면 흔적(디스크·DB·대기열)을 남기지 않는다 | `routers/documents.py` | `tests/test_integration_documents_folders.py::test_없는_폴더는_404_문서도_blob도_안_남는다` · `tests/test_integration_documents_xlsx.py::test_A1_신규_xlsx_설명_없으면_400_흔적_없음` · `tests/test_integration_documents.py::test_409면_blob이_남지_않는다` | |
| INV-22 | 표 하나가 상한을 넘으면 문서를 색인하지 않는다(결정적 실패) | `rag/chunking/split.py` | `tests/test_chunking_docling.py::TestTableExpansionMeta::test_상한을_넘는_표는_TableTooLargeError_결정적_실패` | |
| INV-23 | xlsx 크기 거절 문구는 토큰을 말하지 않는다 | `rag/xlsx_chunking.py` | `tests/test_xlsx_chunking.py::TestChunkXlsx::test_크기_상한_초과시_거절_문구는_토큰을_말하지_않는다` | |
| INV-24 | 같은 테넌트 안 FAQ 대표 질문 중복 금지(DB UNIQUE) | `rag/models.py` | `tests/test_integration_faq.py::test_앱_검사를_지나친_경쟁은_DB_UNIQUE가_409로` · `tests/test_integration_faq.py::test_PATCH_경쟁도_DB_UNIQUE가_409로` | |
| INV-25 | FAQ 수정·삭제·토글은 그 FAQ 근거 캐시를 명시적으로 무효화한다 | `rag/cache.py` | `tests/test_integration_faq.py::test_내용_수정시_해당_FAQ_근거_캐시_무효화` · `tests/test_integration_faq.py::test_off_단독_토글도_캐시_무효화` · `tests/test_integration_faq.py::test_삭제시_청크_cascade와_캐시_무효화` | |
| INV-26 | docx 조문 헤딩 폴백은 번호 목록·표 행·인용 문단을 헤딩으로 올리지 않는다 | `rag/chunking/docx.py` | `tests/test_chunking_parsers.py::TestDocxFallbackHeadings::test_번호_목록과_표_행과_장애_같은_낱말은_승격하지_않는다` · `tests/test_chunking_parsers.py::TestDocxFallbackHeadings::test_인용_문단은_승격하지_않는다` | |
| INV-27 | 관측 장치(워커 메트릭 서버) 실패가 색인을 멈추지 않는다 | `rag/worker.py` | `tests/test_worker_metrics.py::test_포트가_점유돼도_색인은_계속한다` | |
| INV-28 | outbox 행은 지우지 않는다 — done·failed가 이력이자 관측 지점 | `rag/outbox.py` | `tests/test_invariants_static.py::test_INV28_outbox_행을_지우는_코드가_없다` · `tests/test_invariants_static.py::test_INV28_탐지기는_속성_참조와_스키마_접두_삭제도_잡는다` | outbox 모듈의 delete 호출 자체를 막는다(보수적). 속성 참조·스키마 접두 SQL도 잡는다 |
| INV-29 | ready ≡ 색인됨 — ready 승격은 mark_done과 같은 커밋에서만 | `rag/outbox.py` | `tests/test_integration_index_sync.py::test_커밋_단계만_실패해_failed가_된_문서의_청크는_다음_회차에_지워진다` | |
| INV-09 | 문서 삭제는 행·blob을 보존한다(과거 인용의 원본 다운로드) | `rag/documents.py` | `tests/test_integration_documents.py::test_다운로드는_소프트_삭제_후에도_된다` | |

## 턴·스트리밍·취소

| ID | 불변식 | 정의점 | 강제 | 비고 |
|---|---|---|---|---|
| INV-30 | turn_state 함수는 commit하지 않는다 — 호출부가 커밋 시점을 정한다 | `rag/turn_state.py` | `tests/test_invariants_static.py::test_INV30_turn_state는_commit을_호출하지_않는다` |  |
| INV-31 | PreparedRag는 route와 retrieval이 짝이다(knowledge ⟺ retrieval 존재) | `rag/prepared.py` | `tests/test_service_pure.py::TestRouteRetrievalInvariant::test_blocked인데_retrieval이_있으면_거부` · `tests/test_service_pure.py::TestRouteRetrievalInvariant::test_other인데_retrieval이_있으면_거부` · `tests/test_service_pure.py::TestRouteRetrievalInvariant::test_knowledge인데_retrieval이_없으면_거부` | |
| INV-32 | TurnStatus는 다섯 값뿐이고 밖의 값은 쓰기 시점에 거부한다(Document.status 4값과 별개 어휘) | `rag/models.py` | `tests/test_turn_status_contract.py::TestStrEnumCompat::test_다섯_값이_전부다` · `tests/test_turn_status_contract.py::TestWriteValidation::test_다섯_종_밖_값은_즉시_거부된다` (부분) | Enum 생성자 거부만 본다 — 쓰기 지점(finalize_turn) 거부는 미검증. Document.status 4값은 `tests/test_integration_contract_fixes.py::test_문서_상태_어휘에_parsing_embedding이_없다`가 상수로 고정 |
| INV-33 | 화면에 보낸 본문 = 저장되는 본문(인용 꼬리·잘린 버퍼는 어디에도 없다) | `rag/citation_tail.py` | `tests/test_citation_tail.py::TestTailSplitter::test_취소로_꼬리가_잘리면_truncated_버퍼는_어디에도_없다` · `tests/test_integration_query_sse.py::test_꼬리가_잘린_답변은_인용_없이_완주한다` (부분) | 화면·저장 각각만 본다 — 인용 꼬리 경로에서 Message.content == 방출 누적 대조 없음 |
| INV-34 | SSE: 빈 done을 합성하지 않고, error 뒤에도 done으로 닫는다 | `rag/streaming.py` | `tests/test_stream_resume.py::test_마커도_done도_없는_반쪽_스트림은_done_없이_닫힌다` · `tests/test_stream_resume.py::test_생성_실패의_error와_done도_재생된다` · `tests/test_integration_query_sse.py::test_생성_실패도_부분_답변과_소요시간을_남긴다` | |
| INV-35 | 스트림 ping은 Redis 스트림에 저장하지 않는다 | `rag/stream_resume.py` | `tests/test_invariants_turn.py::test_INV35_유휴_ping은_재접속_스트림에_저장되지_않는다` |  |
| INV-36 | 생성 태스크는 finally 맨 앞에서 동기로 등록을 풀고, pop과 cancel 사이에 await가 없다 | `rag/streaming.py` | `tests/test_invariants_turn.py::test_INV36_pop과_cancel_사이에_await가_끼어들_수_없다` · `tests/test_invariants_turn.py::test_INV36_실패한_생성_태스크는_finally_첫_await_전에_레지스트리에서_빠진다` · `tests/test_invariants_turn.py::test_INV36_완주한_생성_태스크는_캐시_적재_전에_이미_취소_대상이_아니다` | pop↔cancel은 동기 함수라는 구조적 단언까지(런타임 레이스 재현은 안 함) |
| INV-37 | 취소는 소유·상태 검증을 레지스트리 접근보다 먼저 한다 | `rag/cancellation.py` | `tests/test_cancellation.py::test_진행_중인_남의_생성은_취소할_수_없다` · `tests/test_cancellation.py::test_취소_대상이_아니면_404` | |
| INV-38 | 취소 엔드포인트는 동시 상한(concurrency_guard)을 거치지 않는다 | `routers/conversations.py` | `tests/test_cancellation.py::test_동시_상한이_가득_차도_취소는_동작한다` | |
| INV-39 | 취소 신호 발행 실패(Redis 순단)는 503 | `routers/conversations.py` | `tests/test_cancellation.py::test_발행_실패는_503` | 202(다른 인스턴스) 응답은 미검증 |
| INV-40 | 생성 경로는 lease를 태스크에 넘기고 끝에 슬롯을 반납한다 | `routers/kms.py` | `tests/test_stream_disconnect.py::test_생성_실패도_failed로_남고_정리가_끝난다` · `tests/test_stream_disconnect.py::test_스트림_중단해도_턴이_done으로_남는다` · `tests/test_invariants_turn.py::test_INV40_생성_경로는_연결이_끊겨도_태스크가_슬롯을_쥐고_끝에_한번만_반납한다` | 끊긴 동안 슬롯 유지·반납 1회 |
| INV-41 | 즉시 경로(캐시 히트·차단·근거 없음)는 lease를 넘기지 않는다(이중 반납·슬롯 유출 금지) | `routers/kms.py` | `tests/test_invariants_turn.py::test_INV41_근거없음과_차단_즉시_경로는_lease를_넘기지_않고_정확히_한번_반납한다` · `tests/test_invariants_turn.py::test_INV41_캐시_히트_즉시_경로는_lease를_넘기지_않고_정확히_한번_반납한다` | 근거 없음·차단·캐시 히트 — 반납 횟수까지 센다 |
| INV-42 | prepare 단계 실패는 LLM 판단 불가·일시 장애(연결·타임아웃·429/5xx, `outbox.is_transient`)만 503, 그 외는 500 — 폴백 추측 없음, 5xx는 스팬 ERROR·404는 아님 | `routers/kms.py` | `tests/test_integration_query_sse.py::test_인텐트_판단_실패는_503이고_질문은_남는다` · `tests/test_integration_query_contract.py::test_LLM_호출_실패는_503이고_턴은_failed로_닫힌다` · `tests/test_integration_query_contract.py::test_검색_저장소와_임베딩_장애는_503` · `tests/test_integration_query_contract.py::test_대화와_무관한_ValueError는_500이고_턴은_failed로_닫힌다` · `tests/test_integration_query_contract.py::test_스트림_전_오류는_SSE가_아니라_JSON` | |
| INV-43 | Redis가 죽으면 동시 상한은 통과시킨다(fail-open) | `rag/limiter.py` | `tests/test_limiter_fail_open.py::test_Redis가_죽으면_상한_없이_통과한다` · `tests/test_limiter_fail_open.py::test_Redis가_죽어도_질문은_받아진다` | |
| INV-44 | 프롬프트 이력에서 blocked·generating 턴은 짝째로 빼고, cancelled·failed 턴은 남긴다 | `rag/conversation.py` | `tests/test_guardrail_hardening.py::test_차단_턴은_이력에서_짝째로_제외` · `tests/test_guardrail_hardening.py::test_실패_턴은_이력에_남는다` · `tests/test_guardrail_hardening.py::test_취소_턴은_짝으로_이력에_남는다` · `tests/test_turn_status_contract.py::TestDerivedSets::test_격리와_미응답은_교집합이_없다` (부분) | generating 격리 단언 없음 |
| INV-45 | 취소·실패 표식은 조립 시점에만 붙이고 저장본(Message.content)은 바꾸지 않는다 | `rag/conversation.py` | `tests/test_guardrail_hardening.py::test_취소_턴은_프롬프트_이력에서_중단_표식` (부분) | 저장본 불변은 간접 |
| INV-46 | RETRY 인텐트는 저장 계층에 닿지 않는다 | `rag/llm_schemas.py` | `tests/test_retry_dispatch.py::test_RETRY_직전취소는_그_질문으로_KNOWLEDGE_재실행` (부분) | messages.intent 컬럼 미검증 |
| INV-47 | 캐시 적재는 done 전송 뒤에 따로 한다 | `rag/streaming.py` | `tests/test_invariants_turn.py::test_INV47_캐시_적재는_done_방출_뒤에_따로_한다` |  |
| INV-48 | 스테일 치유는 워커 cron만 한다(GET은 상태를 바꾸지 않는다) — 전역이다 | `rag/turn_state.py` | `tests/test_stale_sweep.py::test_전역이다_다른_테넌트도_함께_치유된다` · `tests/test_stale_sweep.py::test_GET은_더이상_상태를_바꾸지_않는다` |  |
| INV-49 | 질문과 generating 자리표시는 판단 이전에 커밋되고, 실패하면 failed로 닫힌다(#72) | `rag/service.py` | `tests/test_integration_query_sse.py::test_인텐트_판단_실패는_503이고_질문은_남는다` · `tests/test_integration_query_contract.py::test_LLM_호출_실패는_503이고_턴은_failed로_닫힌다` · `tests/test_integration_query_contract.py::test_대화와_무관한_ValueError는_500이고_턴은_failed로_닫힌다` | |

## 검색·인용·캐시·LLM

| ID | 불변식 | 정의점 | 강제 | 비고 |
|---|---|---|---|---|
| INV-50 | 인용 번호는 검색 출처 순서 1부터, 이어서 첨부 순서 — 컨텍스트 블록·꼬리 제약·파서가 같은 순서 | `rag/citation_labels.py` | `tests/test_prompts.py::TestContextBlocks::test_다른_문서는_등장_순서대로_번호` · `tests/test_prompts.py::TestCitationTail::test_후보_수는_검색_출처_더하기_첨부` · `tests/test_citation_tail.py::TestResolveCitations::test_첨부_인용은_FAQ_선례의_가짜_인용_객체` · `tests/test_invariants_docs.py::test_INV50_컨텍스트_블록_꼬리_제약_파서가_같은_입력에_같은_번호_객체_매핑을_낸다` |  |
| INV-51 | API `citations`와 SSE `done.citations`는 같은 이름·의미 | `schemas/conversations.py` | `tests/test_invariants_docs.py::test_INV51_일반_문서_인용은_SSE_done과_대화_API가_같은_키와_값이다` |  |
| INV-52 | 문서 출처 id는 양수, FAQ는 음수 네임스페이스(캐시 무효화 키와 짝) | `rag/service.py` | `tests/test_service_pure.py::TestSourceDocIds::test_문서는_양수_FAQ는_음수` · `tests/test_service_pure.py::TestSourceDocIds::test_같은_번호의_문서와_FAQ는_구분` | |
| INV-53 | 한 턴의 쿼리 벡터는 하나 — 같은 질의를 다시 임베딩하지 않는다 | `rag/retrieval_types.py` | `tests/test_integration_cache.py::test_턴당_임베딩_왕복은_1회` | |
| INV-54 | 리랭커는 후보 순서만 바꾼다(집합 불변) | `rag/reranker.py` | `tests/test_retrieval_ranking.py::TestRerankMaxpool::test_한_쿼리라도_실패하면_점수가_None` (부분) | 성공 경로 집합 동일성 미검증 |
| INV-55 | 식별자 가산은 리랭커 뒤에 하고 리랭커 입력은 바꾸지 않는다 | `rag/retriever.py` | `tests/test_identifier_channel.py::test_가산이_장조_혼동을_넘고_근거는_그_조만_오린다` (부분) | 리랭커 입력 불변 미검증 |
| INV-56 | LLM 구조화 출력은 부분·퍼지 복구를 하지 않는다 | `rag/llm_schemas.py` | `tests/test_llm_schemas.py::test_스키마_위반은_판단실패_부분복구_없음` · `tests/test_citation_tail.py::TestResolveCitations::test_숫자가_없으면_빈_목록` | |
| INV-57 | 대화 검색은 한 대화에 여러 메시지가 걸려도 1건이고, 첨부 본문은 검색하지 않는다 | `rag/conversation_search.py` | `tests/test_conversation_search.py::test_한_대화에_여러_매칭이어도_total은_1` · `tests/test_conversation_search.py::test_첨부_본문은_검색되지_않는다` | |
| INV-58 | 동시성 기본값은 config와 schema.sql DEFAULT가 같다 | `config.py` | `tests/test_docs_freshness.py::TestInternalNumberInvariants::test_동시성_기본값이_config와_schema에서_일치한다` | |
| INV-59 | FAQ 수정 뒤 옛 내용 기반 답변을 다시 적재하지 않는다(faq_versions 낙관적 검증) | `rag/cache.py` | `tests/test_integration_cache.py::test_생성_중_FAQ_수정되면_캐시_저장_스킵` · `tests/test_integration_cache.py::test_FAQ_변경_없으면_스냅샷_검증_통과_저장` | |

## 코드 규약

| ID | 불변식 | 정의점 | 강제 | 비고 |
|---|---|---|---|---|
| INV-70 | rag·config는 routers·eval을 import하지 않고, config는 프로젝트 모듈을 import하지 않는다 | `AGENTS.md` | `tests/test_architecture.py::test_rag와_config는_역방향으로_import하지_않는다` | |
| INV-71 | rag 모든 모듈은 계층 절에 층이 있다 · 0층은 문서화된 예외만 · 진입점 톱레벨 import 금지 · 조립점은 진입점만 | `AGENTS.md` | `tests/test_architecture.py::test_rag의_모든_모듈이_계층_절에_있다` · `tests/test_architecture.py::test_0층은_문서화된_예외_말고_rag를_톱레벨에서_쓰지_않는다` · `tests/test_architecture.py::test_진입점은_다른_rag_모듈이_톱레벨에서_import하지_않는다` · `tests/test_architecture.py::test_조립점은_진입점만_import한다` | |
| INV-72 | 함수 안 지연 import는 AGENTS.md 목록과 같다 · 톱레벨 순환·상대 import 금지 | `AGENTS.md` | `tests/test_architecture.py::test_지연_import는_AGENTS_목록과_같다` · `tests/test_architecture.py::test_rag_톱레벨_import에_순환이_없다` | |
| INV-73 | opensearchpy·litellm은 톱레벨에서 import하지 않는다 | `AGENTS.md` | `tests/test_invariants_static.py::test_INV73_opensearchpy와_litellm은_rag_어느_모듈도_톱레벨에서_import하지_않는다` |  |
| INV-74 | 문서 속 경로·심볼·숫자는 실존한다 | `AGENTS.md` | `tests/test_docs_freshness.py::TestPathReferences::test_문장_속_경로는_실존해야_한다` · `tests/test_docs_freshness.py::TestSymbolReferences::test_문장_속_심볼은_그_파일에_있어야_한다` (부분) | 경로·심볼만 — 숫자는 INV-58 등 개별 테스트에 흩어져 있다 |
| INV-75 | 품질 기준선보다 나빠지지 않는다(형식·버그 패턴·복잡도·크기·`_이름` import·모듈 docstring) | `AGENTS.md` | `tests/test_code_quality.py::test_품질_기준선보다_나빠지지_않는다` |  |
| INV-76 | 공유 deny가 전체 스테이징·강제 push·main 직접 push의 흔한 형태를 막고 평소 명령은 막지 않는다 | `AGENTS.md` | `tests/test_guardrails.py::test_deny가_있고_필수_명령을_막는다` · `tests/test_guardrails.py::test_deny는_평소_명령을_막지_않는다` |  |
| INV-77 | OS 클라이언트는 커넥션 상한(maxsize)을 명시한다 | `rag/os_client.py` | `tests/test_os_client_pure.py::TestConnectionPool::test_maxsize를_명시한다` | |
| INV-78 | BM25·HNSW 파라미터는 매핑에 명시하지 않는다(엔진 기본값) | `rag/os_client.py` | `tests/test_os_client_pure.py::test_bm25_hnsw_파라미터는_엔진_기본값을_쓴다` | |
| INV-79 | llm_schemas의 구조 결정 필드(safe·intent·standalone)에 기본값을 주지 않는다 | `rag/llm_schemas.py` | `tests/test_llm_schemas.py::test_스키마_위반은_판단실패_부분복구_없음` (부분) | 필드별 required 일반 단언 없음 |
| INV-80 | Prometheus 라벨에 고카디널리티 값(tenant_id·conversation_id·filename)을 쓰지 않는다 | `rag/metrics.py` | `tests/test_invariants_static.py::test_INV80_Prometheus_라벨에_고카디널리티_값을_쓰지_않는다` | 금지 토큰: 엔터티 id·사용자·자유 텍스트 계열 |
| INV-81 | `schemas/query.py`의 죽은 레거시를 새 코드에서 쓰지 않는다 | `AGENTS.md` | `tests/test_invariants_static.py::test_INV81_죽은_레거시_schemas_query를_아무도_import하지_않는다` |  |
| INV-82 | 측정 축은 정본 골드(`eval/gold_set_v2.jsonl`)만 읽는다 | `AGENTS.md` | `tests/test_invariants_static.py::test_INV82_분할본_gold_v2는_validate_gold_v2만_읽는다` · `tests/test_invariants_static.py::test_INV82_eval이_여는_골드_파일은_정본과_허용_목록뿐이다` | 분할본은 validate_gold_v2만, 옛 v1은 레거시 두 스크립트만(허용 목록 — 정리는 #306) |
| INV-83 | 기존 인덱스에는 덧붙이는 필드만 추가하고 매핑 변경은 건드리지 않는다 | `rag/os_client.py` | `tests/test_invariants_static.py::test_INV83_기존_인덱스에는_빠진_추가_필드만_정의_그대로_덧붙인다` · `tests/test_invariants_static.py::test_INV83_추가_필드는_재색인_없이_붙여도_되는_모양뿐이다` |  |
| INV-84 | BM25 질의는 lex_clause()로만 만든다(기본 match는 2.18 hybrid를 깬다) | `rag/os_client.py` | `tests/test_os_search_pure.py::test_nori_질의에_2_18_결함_회피가_붙어_있다` (부분) | 플래그 존재만 — 모든 호출부가 lex_clause를 쓰는지 미검증 |
