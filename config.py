"""애플리케이션 설정 모듈.

.env 파일과 OS 환경변수에서 값을 읽어 Settings 객체로 변환.
다른 모듈에서 `from config import settings`로 가져다 씀.

우선순위: OS 환경변수 > .env 파일 > 코드 기본값.
"""
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# 이 파일(config.py)이 앱 루트에 있으므로, 그 디렉터리가 프로젝트 루트.
# cwd(서버 실행 위치)에 의존하지 않도록 절대경로로 고정한다.
PROJECT_ROOT = Path(__file__).resolve().parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        # .env.dev(개발계 공용 기본, git 커밋됨) → .env(로컬 오버라이드, gitignore) 순.
        # pydantic-settings는 '튜플의 뒤 파일이 우선'이라 .env를 뒤에 둬야 로컬이 이긴다.
        # (기존 순서가 반대로 돼 있어 .env.dev가 로컬을 덮던 잠복 버그 — 2026-08-08 발견·수정.
        #  두 파일 값이 그간 사실상 같아 증상이 없다가, 공용 DB 분리 시점에 드러남)
        # cwd 무관하게 절대경로 (부팅 이식성).
        env_file=(str(PROJECT_ROOT / ".env.dev"), str(PROJECT_ROOT / ".env")),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # database
    database_url: str
    # 커넥션 search_path — 앱 쿼리는 스키마를 명시하지 않으므로(FROM conversations) 여기서 해소.
    # vector 타입이 cdb_admin에 설치돼 있어 cdb_admin도 포함. 로컬(public 스키마)은 뒤의 public으로 폴백.
    # 환경별로 .env의 DB_SEARCH_PATH로 오버라이드. (존재하지 않는 스키마는 무시되므로 안전)
    db_search_path: str = "cc_kms_test,cdb_admin,public"

    # redis
    redis_url: str = "redis://localhost:6379/0"

    # 동시 in-flight 제한 (F6). GPU/모델 미정이라 env로 튜닝. tenant_quotas 행이 있으면 그게 우선.
    # 이 두 값은 tenant_quotas의 DDL 기본값과 같게 유지할 것 — 행 유무로 상한이 달라지면 안 된다(#24).
    # 부하 실측(#101): 인프라 붕괴점이 동시 ~32(완료 p95가 12초→63초로 폭증)이고 안전선은 24.
    # 그런데 테넌트가 23개라 테넌트당 10이면 이론상 동시 230까지 앱이 허용해 전역 상한이 없다 —
    # 한정된 GPU(14B 단일)에 맞춰 테넌트 상한을 조인다. 전역 동시 상한은 최종 스펙 확정 시 별도.
    concurrency_limit_default: int = 5    # 테넌트 동시 in-flight 기본 (quota 행 없을 때)
    # 1 = 멀티창으로 동시 두 질문 차단. 상담 보조 도구라 한 사람이 병렬 질의할 이유가 없고,
    # 자원이 한정적이라 열어줄 여유가 없다. **X-User-Id 헤더가 있을 때만 적용**된다(limiter
    # docstring) — FE가 헤더를 안 보내면 테넌트 상한만 걸리므로, 이 값이 실효하려면 FE 전송 필요.
    user_concurrency_default: int = 1     # 사용자(X-User-Id)별 동시 in-flight 기본
    inflight_max_seconds: int = 120       # in-flight 유령 판정 — 넘으면 카운트서 제거(강제종료 아님)
    sse_ping_interval_seconds: int = 15   # SSE 유휴 ping 주기 — 프록시 idle 종료 대비 (#56, 생성 경로만)
    # 재접속용 Redis Stream (#75). TTL은 종료 후 재접속 유예 — 레이스(이력 조회와 구독 사이에
    # 생성이 끝나는 창)를 매끄럽게 흡수한다. 스트림 하나가 답변 하나 크기(수 KB)라 넉넉해도 무해.
    stream_resume_ttl_seconds: int = 300
    # 배치 창 — 토큰마다 XADD하면 원격 Redis 왕복이 토큰 루프에 얹혀 latency를 오염시킨다.
    # 첫 배치는 이 창을 안 기다린다(첫 토큰 체감 보호). FE가 자체 타자기로 그려 청크 크기는 렌더에 무관.
    stream_resume_flush_seconds: float = 0.05
    # 재접속 리더의 XREAD 블록 창. ping 주기와 분리한 값이다 — 예전엔 sse_ping_interval_seconds를
    # 그대로 썼는데, 그러면 **마커 없는 스트림이 그 시간만큼 침묵한다**. 순단으로 미러링이 끊기고
    # 키 삭제까지 실패하면 반쪽 스트림이 TTL 내내 남는데, 리더는 재생을 마친 뒤 이 창이 만료돼야
    # 비로소 DB로 종료를 판정한다(실측 15.3초 — 화면엔 몇 글자만 뜨고 멈춘 것처럼 보인다).
    # 5초면 그 체감이 1/3로 줄고, 늘어나는 건 유휴 ping 빈도뿐이라 비용이 사실상 없다.
    # ping 주기를 통째로 줄이지 않는 이유: 그건 정상 생성 경로 전체에 걸리는 값이다.
    stream_resume_block_seconds: float = 5.0
    # flush 한 번의 상한. shared_redis에 소켓 타임아웃이 없어, 원격 Redis가 에러 없이 멈추면
    # (블랙홀 라우팅 등) 무한 대기한다 — 그 대기가 _run_generation의 finally에 있어 리미터
    # 반납까지 막는다. 미러링 실패가 생성을 막으면 안 된다는 원칙은 예외뿐 아니라 hang에도 적용돼야 한다.
    stream_resume_flush_timeout_seconds: float = 5.0

    # CORS — FE가 다른 origin에서 서빙될 때(배포·프록시 없는 로컬) 허용 목록.
    # 쉼표 구분 (.env 예: CORS_ALLOW_ORIGINS=http://localhost:5173,https://iccs.example.com)
    cors_allow_origins: str = "http://localhost:5173,http://localhost:3000"

    # vllm
    vllm_base_url: str = "http://localhost:11434/v1"
    vllm_model: str = "qwen3:4b"
    # 문서 기반 사실 제공 서비스라 다양성이 손해 — 낮게 고정 (2026-08-05).
    # None이면 서버 기본값(Qwen3 non-thinking 권장 0.7)이 쓰이는데, 실측 4회 반복에서 매번 다른 답이
    # 나왔다. 인용 형식 파싱(출처 꼬리)·시맨틱 캐시가 모두 결정성에 의존해 불리하다.
    # 0.0(greedy)은 배제 — Qwen3 모델 카드가 "성능 저하·무한 반복" 위험으로 명시 금지.
    # 0.2는 greedy를 피하는 최소 대역이고 저온이라 top_p·top_k는 거의 개입하지 않아 따로 안 보낸다.
    llm_temperature: float | None = 0.2
    llm_enable_thinking: bool = False     # Qwen3 thinking 모드. 서버가 --reasoning-parser로 떠 있어 추론은 reasoning 필드로
                                          # 분리되고 우리는 .content만 읽으므로 답변엔 안 섞인다. 다만 추론 토큰만큼
                                          # 첫 토큰 지연·생성 시간이 늘고 max_tokens 예산을 함께 먹는다. 실험용 토글(기본 off).
                                          # 주의: 전 호출(인텐트·condense·생성) 공유 — 운영에 켤 땐 호출별 분리 필요

    # embedding (F99: TEI 원격 서버, dense-only). 저장 벡터도 이 서버 출력이어야 검색 정상 — 재인제스트로 일치 보장
    embed_base_url: str = "http://localhost:38889"    # TEI 임베딩 서버 (BGE-M3, /embed) — 실주소는 .env
    embed_timeout: float = 30.0
    embed_dimensions: int | None = None

    # OTel 트레이싱 (#7). 미설정(빈 값)이면 완전 no-op — 계측 코드는 돌지만 스팬이 기록되지 않음.
    # 로컬 Phoenix: http://localhost:6006/v1/traces (docker-compose.local.yml). 개발계 전용 권장.
    otel_endpoint: str = ""
    otel_text_limit: int = 500        # 스팬에 담는 청크 본문·답변 절단 길이 (질문·재작성은 전문)

    # 워커 프로세스 지표 (#151). arq에는 HTTP 서버가 없어 웹의 /metrics(main.py)가 워커를 못 덮는다 —
    # 값을 주면 워커가 startup에서 그 포트로 자기 /metrics를 연다(0이면 안 연다).
    # 핵심은 prometheus_client 기본 ProcessCollector의 process_resident_memory_bytes다:
    # docling 변환(#143)이 컨테이너 mem_limit 안에 머무는지 보는 유일한 수단.
    # **Linux에서만 값이 나온다** — ProcessCollector가 /proc를 읽으므로 macOS에선 조용히 빈 값이다.
    # 기본 0인 이유: 로컬에서 워커를 여러 개 띄울 때 포트 충돌을 기본 동작으로 만들지 않기 위해서다.
    worker_metrics_port: int = 0
    # 바인드 주소. 기본 루프백은 **호스트 네트워크로 띄운 워커**에만 맞다 —
    # 브리지 네트워크 컨테이너에서 루프백에 리슨하면 컨테이너 안에서만 닿아, 포트를 published해도
    # 호스트의 프로메테우스가 못 긁는다(도커 프록시는 컨테이너 IP로 붙는다). 그 구성에서는
    # 0.0.0.0으로 열고 노출 차단은 compose의 published 주소(127.0.0.1:PORT:PORT)에 맡긴다.
    worker_metrics_host: str = "127.0.0.1"

    # 질의 재작성 의미 확장(#5). on이면 '멀티턴에서만' condense 자리 1콜로 멀티쿼리(재작성 1 +
    # 어휘 변형 2)를 뽑아 검색 — rerank on이면 쿼리별 채점 max-pool 정렬, rerank off/실패면 RRF 폴백.
    # 단일턴은 on이어도 현행 경로 그대로(LLM 스킵) — 단일턴 확장은 실측상 손실이라 게이트(service).
    # 기본 on (#5 검증 완료: 재작성 98%, mt Hit@1 +2.2pp·R@20 1.000, 생성 EPCov/Cite +2.8/+5.5pp).
    # off면 코드·프롬프트 모두 도입 전과 동일 — 문제 시 .env 한 줄로 원복.
    condense_multi_query_enabled: bool = True

    # reranker (F99: TEI /rerank, cross-encoder 재정렬). on/off 토글 한 줄.
    rerank_enabled: bool = True                       # False면 dense-only 순서 그대로 (리랭크 skip). .env로 오버라이드 가능
    # 하이브리드 어휘 채널(#135→#139): 엔진 BM25(Nori) 상위 중 dense 미포함분을 리랭커 풀에 주입.
    # 실측(모의 450 + 인티큐브 실문서 75문항)은 주입 상방 0 — 그래도 실코퍼스의 상품코드·고유명
    # 분포 대비 정식 채널로 켜둔다(사용자 결정). 끄면 dense-only.
    hybrid_lexical_enabled: bool = True
    hybrid_lexical_inject: int = 15                   # dense 미포함 BM25 상위 주입 수 (어블레이션 조립과 동일)
    rerank_base_url: str = "http://localhost:38890"   # TEI 리랭커 서버 (bge-reranker-v2-m3, /rerank) — 실주소는 .env
    rerank_timeout: float = 30.0

    # 검색 저장소 — OpenSearch 하나 (#139 도입 확정 2026-09-12, 사유=확장성). PG는 문서·FAQ·폴더·
    # 대화·캐시·outbox의 정본이고, 청크(본문·메타·벡터·어휘 필드)는 엔진에만 있다. 서빙은 엔진에서
    # 끝난다(rag/os_client.py 상단). 기동 시 인덱스 존재를 보장한다(ensure_index_soft) — 엔진에 못 붙어도
    # 프로세스는 뜨고 ERROR 로그를 남긴다(로컬 개발 편의, 사용자 결정). 그 상태에서 검색·색인은 호출 시점에 실패한다.
    #
    # 받아들인 것: 색인 반영 지연. 모든 변경은 트랜잭셔널 outbox(rag/outbox.py)로 durable하게 기록되고
    # 단일 워커 cron이 1분마다 처리한다 — 유실은 없지만 즉시 반영도 없다. 제품 결정: "문서 변경은
    # 검색에 최대 1~5분 뒤 반영될 수 있다"(답변 캐시는 즉시 무효화됨).
    # 남은 것: BM25 df 스코프가 인덱스 전체(테넌트 간 idf 간섭 — 실측 표시 지표 미동). 테넌트별
    # 인덱스가 대안. 배포는 자체 설치 단일 노드(매핑 1샤드·0레플리카)라 노드 다운 = 검색 불가 —
    # 인제스션은 outbox가 pending으로 들고 있다가 복구 후 반영한다.
    opensearch_url: str = "http://localhost:9200"     # 실주소는 .env
    opensearch_index: str = "kms_chunks_v1"
    opensearch_timeout: float = 30.0

    # 컨텍스트 예산 (F100). context_window는 vLLM --max-model-len과 반드시 일치시킬 것.
    context_window: int = 30720
    generation_reserve_tokens: int = 3000    # 답변 생성 몫 = max_tokens (한글 ~4,500자 상한 — 폭주 방지용, 정상 답변은 미도달)
    history_budget_tokens: int = 2000        # 이전 대화 참고 몫 (최신 턴부터 예산 소진까지) — 답변 생성용
    condense_history_budget_tokens: int = 600  # 질의 재작성(condense)용 히스토리 예산 — 참조 해소엔 최근 몇 턴이면 충분,
                                               # 길면 이전 답변의 수치가 질의에 주입됨 (실측: 1751tk 1/5 → 493tk 5/5, 2026-07-20)
    max_attachments: int = 1                 # 컨텍스트 유지 첨부 개수 (넘으면 오래된 것 제외 — FE 고정 안내)
                                             # 2→1 (#63 운영 결정): 복수 첨부는 "이 문서" 단수 지시의
                                             # 대상 모호·이월+신규 공존 등 엣지 표면만 넓혔다. 파일럿에서
                                             # 두 문서 대조 실수요가 나오면 되돌린다 (ATTACHMENT_MAX_ITEMS와 짝)

    # cache (exact 캐시 제거됨 — semantic만)
    # 히트 = floor 이상 후보 + doc집합 동일 + LLM 재사용 판정 승인 (#113).
    # 자동 서빙 임계(구 0.95)는 제거 — 유사도 0.96~0.99에서도 답이 반대인 쌍 4건 실측,
    # 유사도 단독으론 어떤 값도 서빙을 정당화 못 한다. 규칙 기반 기계 가드도 제거 —
    # 판정기가 전 negative(22/22)를 잡는 것이 실측된 뒤로는 유지보수 부채(오차단 실사례 1건).
    # floor는 판정 콜을 아낄 후보 게이트일 뿐이다: 실측 paraphrase 최저 0.8058 포괄.
    semantic_cache_floor: float = 0.80
    # 미히트 캐시 보존 기간(#16) — last_hit_at 기준이라 히트마다 연장, 인기 답변은 영구 생존.
    # TTL이 아님: 정확성은 무효화·doc집합 비교가 담당하고 이건 죽은 row 위생(LIMIT 1 가림 완화).
    cache_retention_days: int = 90

    # storage
    blob_storage_dir: str = str(PROJECT_ROOT / "docs")

    # PDF 파서 — docling 재도입 (#143). PDF 인제스션(chunk_file) 경로만 대상이고 채팅 첨부
    # (extract_text)는 pdfplumber 유지 — 첨부는 사용자 대기 경로라 콜드 스타트 4.4초가 그대로 보인다.
    # 값은 전부 2026-09-11 실측(M3 CPU, 실문서 21건)에서 왔고, **Linux 워커에서 재측정 후 조정한다**
    # — macOS와 glibc 할당자의 메모리 반납 패턴이 다를 수 있다. 소비처는 rag/chunking.py(연결 예정).
    docling_enabled: bool = True              # PDF 인제스션의 기본 경로. False = pdfplumber 비상 경로
                                              # (표·부산물 처리 없음, #141 휴리스틱은 #143에서 제거). 폴백 아님 — 수동 스위치
    docling_table_mode: str = "accurate"      # fast | accurate. accurate이 병합 셀을 한 셀로 잡는다(출장 지급표
                                              # '부회장/사장' 실측). RAM 차이 +110MB(687→796MB)라 정확도를 택한다
    docling_do_ocr: bool = False              # 켜도 텍스트 PDF 안의 그림은 못 읽는다(자리표시만) — RAM만 +200MB.
                                              # 스캔 PDF가 들어오는 날 재검토. 텍스트 PDF 속 그림은
                                              # OCR이 아니라 VLM 캡션으로 읽는다(#162, 아래 vlm_caption_*)
    docling_do_cell_matching: bool = False    # 7월 설정 계승 — 켜면 한글 표 셀의 공백이 뭉친다 ("편도 3,000원")
    docling_device: str = "cpu"               # 'auto'는 서버에서 GPU/MPS 탐색을 시도한다. 앱은 GPU 장비에 안 올린다
    docling_num_threads: int = 4              # 워커 vCPU 수에 맞춤. 공식 실측: 4→16스레드에 쪽당 1.67→1.09초
    # 메모리 노브 — **리눅스 컨테이너 실측 기준**(2026-09-13). 이전 주석은 macOS에서 샘플링한
    # RSS(0.35~0.85GB)에 기대 있었고 피크가 아니었다. 같은 문서를 ru_maxrss 피크로 다시 재면
    # 맥 1,708MB / 리눅스 2,641MB다 — 눈금은 배포 대상(리눅스 x86_64)에서만 의미가 있다.
    # (docling_page_batch_size는 제거했다 — 2.126의 기본 StandardPdfPipeline은 스레드 구조로
    #  재작성돼 그 값을 읽지 않는다. 읽는 것은 옛 LegacyStandardPdfPipeline뿐이고, 디버그 로그로
    #  해당 코드가 한 번도 실행되지 않음을 확인했다.)
    docling_layout_batch_size: int = 1         # 레이아웃 모델에 한 번에 넣는 쪽 수. 기본 4 → 1로 내리면
                                               # 피크 2,270~2,486MB → 1,779~1,843MB(3회 반복 재현).
                                               # 산출물은 sha256까지 동일하고 오히려 0.5~1.2초 빠르다.
                                               # 맥에서는 효과가 편차 내로 묻힌다 — 플랫폼 의존적이다.
    docling_queue_max_size: int = 4            # 스테이지 간 큐 길이(기본 100). 긴 문서에서 처리 대기 쪽이
                                               # 쌓이는 상한 — layout_batch와 함께 내려야 효과가 있다
    docling_max_concurrency: int = 1          # **arq max_jobs(10)와 별개인 변환 동시 상한.** chunk_file은 to_thread라
                                              # 잡 10개가 겹치면 변환도 10개 겹쳐 메모리가 10배 난다. 세마포어로 막는다.
                                              # 임베딩 I/O는 max_jobs가 계속 겹치게 둔다 — 변환만 직렬화
    docling_document_timeout_seconds: float = 300.0   # 사슬: 이 값 < arq job_timeout 600 < DOC_STALE_SECONDS 900.
                                              # 600 안에서 임베딩 몫을 남긴다. 실측 최대 26쪽 6초라 50배 여유
    docling_max_num_pages: int = 300          # 병적 입력 가드. **초과분을 자르는 게 아니라 문서를 거부한다**
                                              # (ConversionError → failed, 2026-09-13 실측). 파일 크기 상한은 라우터(DOC_MAX_FILE_BYTES 10MB)가
                                              # 업로드 시점에 이미 막으므로 여기선 쪽 수만. 10MB 텍스트 PDF ≈ 100~200쪽
    docling_artifacts_path: str | None = None # 모델 가중치 디렉터리. None이면 첫 변환 때 HF에서 내려받는다 —
                                              # NCP VM은 외부망 확인 필요. 이미지에 미리 굽고 경로를 주는 쪽이 안전

    # 그림 캡션 — docling picture description 훅으로 그림을 사내 VLM에 보내 설명을 받아 청크에 싣는다 (#162).
    # 대상은 **PictureItem뿐**이다. 격자가 있는 표 이미지는 docling이 TableItem으로 분류해 이 훅을
    # 타지 않는다(실측 2026-09-14: 표 이미지 PDF → PictureItem 0·TableItem 1). 그건 별도 이슈다.
    # VLM이 죽어도 docling이 예외를 삼킨다(실측: 닫힌 포트 조준 → 예외 없이 7.5초, 캡션 0개) —
    # 그래서 실패 폴백(자리표시 유지)에 우리 try/except가 필요 없다. 문서는 그대로 ready로 진행한다.
    # **on/off 스위치를 두지 않는다**(사용자 결정): VLM이 없거나 못 닿으면 캡션이 안 붙을 뿐이고
    # 그 상태가 곧 기존 동작이라, 끄는 것과 결과가 같다.
    vlm_caption_url: str = "http://localhost:18892/v1/chat/completions"   # **base가 아니라 전체 엔드포인트다**
                                              # — docling이 이 값을 그대로 POST 대상으로 쓴다. 실주소는 .env
    vlm_caption_model: str = "qwen25-vl"      # OpenAI 호환 body의 model 필드. Qwen2.5-VL-7B-Instruct(Apache 2.0)
    vlm_caption_scale: float = 4.0            # **그림 크롭을 VLM에 보낼 때의 해상도 배수(72dpi 기준).**
                                              # docling 기본 2.0 → 4.0. 1.0(434×60px)에서는 글자가 안 보여 VLM이
                                              # 원문에 없는 내용을 지어냈다(3회 재현). 4.0(1736×239px)에서 원문
                                              # 완전 일치(3회 재현). PdfPipelineOptions.images_scale과 **다른 값**이다
                                              # — 그쪽은 캡션 품질과 무관함을 실측으로 분리 확인했다(2026-09-14)
    vlm_caption_timeout_seconds: float = 20.0 # 요청 하나의 상한(docling 기본값 유지). 예산: 이 값 × 그림 수가
                                              # docling_document_timeout_seconds(300)를 넘으면 문서가 failed 된다.
                                              # 실문서는 최대 2장/문서라 여유가 크지만, 그림 많은 문서가 들어오면 재검토


# 모듈 import가 곧 프로세스당 1회이므로 이 전역 자체가 싱글톤이다 (팩토리·캐시 불필요).
settings = Settings()

