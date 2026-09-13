# E2E 실측 — OpenSearch 단일 저장소 + 트랜잭셔널 outbox (#139, PR #140)

- 실행: 2026-09-13 00:40~01:20, 브랜치 `feat/issue-139-opensearch` @ `15d7823`
- 구성: 로컬 `uvicorn main:app --port 8010` + `arq rag.worker.WorkerSettings`(cron 1분) + 로컬 OpenSearch 2.18.0 컨테이너,
  개발계 PG, 실 TEI(임베딩·리랭커)·실 vLLM(Qwen3-14B). 테넌트 `e2e-139`. 드라이버가 API를 실제로 호출하고 SSE `done.citations`를 읽었다.
- 문서: 회의실 예약 안내 md(v1 "최대 4시간", 개정판 "최대 2시간"), 깨진 PDF.

## 결과 요약 — 10/10 통과

| # | 케이스 | 결과 | 실측 |
|---|---|---|---|
| 1 | 서버+워커 기동 후 `/kms/query` | ✅ | 인용 `{document_id, filename: 'e2e_회의실.md', version}` 정상, 답변 "최대 4시간까지 가능합니다", `cached=False` |
| 2 | 업로드 → 즉시 목록 → 1분 내 ready | ✅ | 응답 즉시 `pending` · PG `chunks` 0행 · **56초** 뒤 ready(outbox 행 done, 같은 커밋) · 그때부터 인용됨 |
| 3 | 워커 내린 채 업로드 | ✅ | (아래 수동 절 참조) |
| 4 | 삭제 → 바로 질의 → 1분 내 인용 안 됨 | ✅ | DELETE 204 → 즉시 질의는 **아직 인용됨**(가이드대로) → **58초** 뒤 인용 안 됨, DROP 행 done |
| 5 | 검색토글 off → 제외 → on → 복귀, 재색인 없음 | ✅ | off: 즉시 질의 아직 인용 → **57초** 뒤 제외, META 행 done, `indexed_at` 불변(재임베딩 없음) · on: **61초** 뒤 복귀 |
| 6 | 같은 이름 재업로드 — 빈 창 없음 | ✅ | 처리 중 8회 폴링(3~52초) 전부 구버전 인용·"4시간" → 58초에 신버전 인용·"2시간". **인용 0회 구간 없음**. 구버전 `deleted`·신버전 `ready` |
| 7 | OpenSearch 내린 채 업로드·삭제 | ✅ | (아래 수동 절 참조) |
| 8 | 깨진 PDF 업로드 | ✅ | 매분 1회 실패 → 5회차에 문서 `failed` + `status_reason`("색인 5회 실패: Conversion failed …"), 행 `failed` attempts 5 |
| 9 | `eval.os_reconcile` 문서 단위 대조 | ✅ | (아래 수동 절 참조) |
| 10 | OpenSearch 내린 채 서버·워커 기동 | ✅ | (아래 수동 절 참조) |

원자료: 드라이버 로그 `run2.md`(자동 6건)·`run3.md`(수동 4건) — 세션 작업 디렉터리(잡 삭제 시 사라짐). 핵심 줄은 위 표에 옮겼다.

## 수동 절 (3·7·9·10)

| # | 절차 | 실측 |
|---|---|---|
| 3 | `pkill arq` → 업로드 → 95초 대기 → 워커 재기동 | 95초 뒤에도 문서 `pending`·행 `pending` attempts 0(**failed 아님** — 900초 스윕이 사라졌으므로 기다리기만 한다) → 재기동 **56초** 뒤 ready, 색인 청크 2 |
| 7 | `docker stop` OS → 업로드(200)·삭제(204) → 75초 → `docker start` | 두 요청 모두 **커밋 성공**(PG만 만진다). 75초 뒤 INDEX·DROP 행 `pending` attempts 1, `last_error="ConnectionError(Cannot connect to host …:9200"`. OS 재기동 15초 후 다음 회차에 둘 다 done — 업로드 문서 ready·청크 2, 삭제 문서 색인 0 |
| 10 | OS 내린 채 `uvicorn`·`arq` 기동 | uvicorn **종료코드 3**(lifespan `ensure_index` 예외로 startup 실패), arq **종료코드 1**(`on_startup`에서 `opensearchpy ConnectionError`). 검색 없는 프로세스가 조용히 뜨지 않는다 |
| 9 | `python -m eval.os_reconcile`(dry-run) | 누락 1건 `[5548]`(랜덤 테넌트의 `정책.pdf` — 원본 blob 없는 잔재 행, 재색인도 같은 이유로 건너뛴 것)·잉여 0건. E2E로 만들고 지운 문서들은 전부 대조 통과 |

케이스 8의 실패 문서는 `status_reason`으로 원인이 보이고, 케이스 7의 재시도는 `last_error`가 남아 outbox 행이 관측 지점 역할을 한다.

## 실행 중 발견 — 환경 함정 1개 (코드 아님)

**개발계 Redis(DB 0)를 다른 호스트의 arq 워커와 공유하고 있었다.** arq cron은 `unique=True`로 "함수명:분" job_id를 쓰므로 같은 Redis에 붙은
워커 둘이 **분을 나눠 가져간다** — 1차 실행에서 로컬 워커는 00:43·46·50·54·55분만 실행됐고(다른 분은 원격 워커가 소비), 그 결과 케이스 4·5·8이
시간 안에 안 끝났다(코드 결함 아님 — 격리 후 재실행에서 전부 1분 내 반영). 로컬 검증은 `REDIS_URL`을 별도 DB 인덱스로 격리해서 돌려야 하고,
**운영 배포에서도 "같은 Redis에 코드가 다른 워커 둘"이 있으면 같은 증상**이 난다 — 배포 체크 항목으로 올린다.
