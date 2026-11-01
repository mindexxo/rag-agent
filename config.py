"""애플리케이션 설정 모듈 — OS 환경변수 > .env > .env.dev > 코드 기본값 순으로 읽어 `settings` 하나로 노출한다."""

import os
from pathlib import Path
from typing import Literal, Self, cast

from dotenv import dotenv_values
from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# cwd(서버 실행 위치)에 의존하지 않도록 이 파일 위치를 프로젝트 루트로 고정한다.
PROJECT_ROOT = Path(__file__).resolve().parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        # pydantic-settings는 튜플의 뒤 파일이 우선이다 — .env(로컬 오버라이드)를 .env.dev(개발계 공용) 뒤에 둬야 로컬이 이긴다.
        env_file=(str(PROJECT_ROOT / ".env.dev"), str(PROJECT_ROOT / ".env")),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # database
    database_url: str
    # 앱 쿼리는 스키마를 명시하지 않으므로(FROM conversations) search_path로 해소한다.
    # vector 타입이 cdb_admin에 설치돼 있어 함께 넣고, 로컬은 public으로 폴백한다. 없는 스키마는 무시된다.
    db_search_path: str = "cc_kms_test,cdb_admin,public"

    # redis
    redis_url: str = "redis://localhost:6379/0"

    # 동시 in-flight 제한. tenant_quotas 행이 있으면 그 값이 우선이다.
    # 이 두 값은 tenant_quotas의 DDL 기본값과 같게 유지할 것 — 행 유무로 상한이 달라지면 안 된다.
    # 인프라 안전선이 전역 동시 약 24인데 앱엔 전역 상한이 없어(테넌트 수 × 테넌트 상한),
    # 테넌트 상한을 낮게 둔다.
    concurrency_limit_default: int = 5  # 테넌트 동시 in-flight 기본 (quota 행 없을 때)
    # 1 = 멀티창으로 동시 두 질문 차단. **X-User-Id 헤더가 있을 때만 적용**된다 — FE가 헤더를 안 보내면
    # 테넌트 상한만 걸린다.
    user_concurrency_default: int = 1  # 사용자(X-User-Id)별 동시 in-flight 기본
    inflight_max_seconds: int = 120  # in-flight 유령 판정 — 넘으면 카운트서 제거(강제종료 아님)
    sse_ping_interval_seconds: int = 15  # SSE 유휴 ping 주기 — 프록시 idle 종료 대비 (생성 경로만)
    # 재접속용 Redis Stream. TTL은 종료 후 재접속 유예 — 이력 조회와 구독 사이에 생성이 끝나는 창을 흡수한다.
    # 스트림 하나가 수 KB라 넉넉해도 무해하다.
    stream_resume_ttl_seconds: int = 300
    # 배치 창 — 토큰마다 XADD하면 원격 Redis 왕복이 토큰 루프에 얹힌다.
    # 첫 배치는 이 창을 안 기다린다(첫 토큰 체감 보호). FE가 자체 타자기로 그려 청크 크기는 렌더에 무관.
    stream_resume_flush_seconds: float = 0.05
    # 재접속 리더의 XREAD 블록 창. 마커 없는 반쪽 스트림(미러링이 끊기고 키 삭제도 실패한 경우)은
    # 재생 뒤 이 창이 만료돼야 DB로 종료를 판정하므로, 그동안 화면이 멈춘 것처럼 보인다.
    # ping 주기와 따로 두는 이유: ping 주기는 정상 생성 경로 전체에 걸린다. 짧게 해도 늘어나는 건 유휴 ping뿐이다.
    stream_resume_block_seconds: float = 5.0
    # flush 한 번의 상한. shared_redis에 소켓 타임아웃이 없어 원격 Redis가 에러 없이 멈추면 무한 대기하고,
    # 그 대기가 _run_generation의 finally에 있어 리미터 반납까지 막는다 — 미러링 hang이 생성을 막지 않게 한다.
    stream_resume_flush_timeout_seconds: float = 5.0

    # CORS — FE가 다른 origin에서 서빙될 때(배포·프록시 없는 로컬) 허용 목록.
    # 쉼표 구분 (.env 예: CORS_ALLOW_ORIGINS=http://localhost:5173,https://iccs.example.com)
    cors_allow_origins: str = "http://localhost:5173,http://localhost:3000"

    # LLM — 공급자 하나를 고르고 그 접속 정보를 준다. 공급자별 구현은 rag/llm/ (정의점 rag/llm/__init__.py).
    # 이름에 벤더를 넣지 않는다 — 옛 VLLM_BASE_URL·VLLM_MODEL이 남아 있으면 기동 시 멈춘다(_reject_renamed_env).
    llm_provider: Literal["vllm", "gemini", "openai", "anthropic"] = "vllm"  # 벤더 추가 = 여기 + rag/llm 등록부
    # 상용 3종은 LiteLLM 경유(rag/llm/litellm.py). llm_model은 접두 없이 벤더 모델명(예: gemini-2.5-flash, gpt-5-mini,
    # claude-sonnet-5), 키는 llm_api_key 하나(벤더별 환경변수 GEMINI_API_KEY 등은 읽지 않는다).
    llm_base_url: str = "http://localhost:11434/v1"
    llm_model: str = "qwen3:4b"
    llm_api_key: str = "EMPTY"  # 사내 vLLM은 키를 안 본다. 상용 벤더 키는 .env에만 둔다(이 파일·.env.dev에 적지 말 것)
    # 문서 기반 사실 제공이라 다양성이 손해다 — 출처 꼬리 파싱·시맨틱 캐시가 결정성에 기댄다.
    # None이면 서버 기본값(Qwen3 권장 0.7)이 쓰여 답이 매번 달라진다.
    # 0.0(greedy)은 Qwen3 모델 카드가 성능 저하·무한 반복 위험으로 금지한다. 0.2는 greedy를 피하는 최소 대역이고
    # 저온이라 top_p·top_k는 따로 안 보낸다.
    llm_temperature: float | None = 0.2
    # 추론(thinking) 수준 — None 또는 "none"이면 끔. 생성의 값이고, 보조 호출(인텐트·condense·캐시 판정)은
    # llm_aux_reasoning_effort가 비어 있으면 이 값을 따른다. 벤더별 변환:
    #   vllm(Qwen3)  : none→enable_thinking=False, 그 외→True (서버가 --reasoning-parser라 추론은 reasoning 필드로 분리,
    #                  답변엔 안 섞이지만 첫 토큰 지연·max_tokens 예산을 먹는다)
    #   gemini       : LiteLLM이 thinking_level/thinking_budget으로 변환
    #   openai(GPT-5): reasoning_effort 그대로. ⚠ 추론이 켜진 모델은 temperature를 1 외 값으로 보내면 400
    #   anthropic    : none→thinking 끔, 그 외→모델에 따라 adaptive 또는 budget. ⚠ 4.7+는 temperature<1이면 400
    # 거부되는 조합은 400으로 바로 드러난다 — 조용히 빼지 않는다.
    llm_reasoning_effort: Literal["none", "minimal", "low", "medium", "high"] | None = None
    # 보조 호출(의도 분류·질의 재작성·캐시 재사용 판정) 전용. 출력이 짧은 JSON 판정이라 하위 모델·낮은 생각 수준으로
    # 비용과 첫 토큰 지연을 줄인다. **비우면 생성 값(llm_model·llm_reasoning_effort)과 같다.**
    # 공급자·주소·키·온도는 생성과 공유하므로 같은 벤더의 상하위 짝(예: gpt-5/gpt-5-mini, Sonnet/Haiku, Gemini Pro/Flash)만 쓴다.
    # 같은 계열이면 온도 제약도 같다(gpt-5는 생각을 켜면 1.0만).
    llm_aux_model: str = ""
    llm_aux_reasoning_effort: Literal["", "none", "minimal", "low", "medium", "high"] = ""  # ""=생성 값을 따름, none=끔
    # ⚠ 벤더마다 받는 값이 다르다 — gpt-5-mini는 'none'을 400으로 거부하고 'minimal'부터 받는다.

    # embedding (dense-only). 저장 벡터도 같은 공급자·모델 출력이어야 하므로 바꾸면 재인제스트 필수.
    # 공급자(정의점 rag/search/embeddings.py): tei(사내, 기본) | openai | gemini | voyage — 상용은 LiteLLM 경유, 모델명은 접두 없이
    # (예: text-embedding-3-large, gemini-embedding-001, voyage-3.5), 차원은 항상 1024로 요청하고 응답을 검사한다.
    embed_provider: Literal["tei", "openai", "gemini", "voyage"] = "tei"
    embed_model: str = (
        ""  # 상용일 때 모델·키 필수(_require_vendor_model_and_key). TEI는 컨테이너당 모델 고정이라 안 본다
    )
    embed_api_key: str = ""  # 상용 키 — .env에만 (LLM과 별개 키: 벤더가 다를 수 있다)
    embed_base_url: str = "http://localhost:38889"  # TEI 임베딩 서버 (BGE-M3, /embed) — 실주소는 .env. 상용이면 무시
    embed_timeout: float = 30.0
    embed_dimensions: int | None = None  # TEI 전용 MRL 절단 — 상용은 rag/search/embeddings.EMBED_DIM(1024) 고정

    # OTel 트레이싱. 빈 값이면 no-op — 계측 코드는 돌지만 스팬이 기록되지 않는다.
    # 로컬 Phoenix: http://localhost:6006/v1/traces (docker-compose.local.yml). 개발계 전용 권장.
    otel_endpoint: str = ""
    otel_text_limit: int = 500  # 스팬에 담는 청크 본문·답변 절단 길이 (질문·재작성은 전문)

    # 워커 프로세스 지표. arq에는 HTTP 서버가 없어 웹의 /metrics(main.py)가 워커를 못 덮는다 —
    # 값을 주면 워커가 startup에서 그 포트로 자기 /metrics를 연다(0이면 안 연다).
    # docling 변환이 컨테이너 mem_limit 안에 머무는지 보는 수단(process_resident_memory_bytes)이다.
    # **Linux에서만 값이 나온다** — ProcessCollector가 /proc를 읽으므로 macOS에선 조용히 빈 값이다.
    # 기본 0: 로컬에서 워커를 여러 개 띄울 때 포트 충돌을 기본 동작으로 만들지 않는다.
    worker_metrics_port: int = 0
    # 바인드 주소. 기본 루프백은 **호스트 네트워크로 띄운 워커**에만 맞다 —
    # 브리지 네트워크 컨테이너에서 루프백에 리슨하면 포트를 published해도 호스트의 프로메테우스가 못 긁는다
    # (도커 프록시는 컨테이너 IP로 붙는다). 그 구성에서는 0.0.0.0으로 열고 노출 차단은
    # compose의 published 주소(127.0.0.1:PORT:PORT)에 맡긴다.
    worker_metrics_host: str = "127.0.0.1"

    # 질의 재작성 의미 확장. on이면 멀티턴에서만 condense 자리 1콜로 멀티쿼리(재작성 1 + 어휘 변형 2)를 뽑아
    # 검색한다 — rerank on이면 쿼리별 채점 max-pool 정렬, rerank off/실패면 RRF 폴백.
    # 단일턴은 확장이 손해라 on이어도 현행 경로 그대로다(service의 게이트). off면 확장 없이 condense 하나.
    condense_multi_query_enabled: bool = True

    # reranker (TEI /rerank, cross-encoder 재정렬).
    rerank_enabled: bool = True  # False면 dense-only 순서 그대로 (리랭크 skip)
    # 하이브리드 어휘 채널: 엔진 BM25(Nori) 상위 중 dense 미포함분을 리랭커 풀에 주입한다.
    # 상품코드·고유명처럼 dense가 놓치는 키워드형 질의 대비. 끄면 dense-only.
    hybrid_lexical_enabled: bool = True
    hybrid_lexical_inject: int = 15  # dense 미포함 BM25 상위 주입 수
    # 리랭크 공급자(정의점 rag/search/reranker.py): tei(사내, 기본) | cohere | voyage | jina — 상용은 LiteLLM arerank 경유,
    # 모델명은 접두 없이(예: rerank-v5.0-pro, rerank-3, jina-reranker-v2-base-multilingual — Cohere 목록은
    # /v2/models?endpoint=rerank). 점수 눈금이 벤더마다 달라 공급자를 바꾸면 ident_boost를 다시 맞춰야 한다.
    rerank_provider: Literal["tei", "cohere", "voyage", "jina"] = "tei"
    rerank_model: str = ""  # 상용일 때 모델·키 필수(_require_vendor_model_and_key)
    rerank_api_key: str = ""  # 상용 키 — .env에만
    rerank_base_url: str = (
        "http://localhost:38890"  # TEI 리랭커 서버 (bge-reranker-v2-m3, /rerank) — 실주소는 .env. 상용이면 무시
    )
    rerank_timeout: float = 30.0
    # 식별자 채널(정의점 rag/identifiers·rag/search/retriever). 질문에 사전 식별자('11조')가 있을 때만 발동:
    # 정확 일치 조각을 리랭커 풀에 주입하고, 리랭크 **뒤** 점수에 ident_boost를 더해 재정렬한다.
    # 0.3 = 장·조 혼동 점수 차(0.20~0.25)보다 크고 리랭커의 확신(0.3 이상 차이)은 뒤집지 않는 값.
    # 더 크게 잡으면 식별자와 무관한 순위 문제까지 보정치로 덮는다. TEI 리랭커 눈금 기준이다.
    ident_channel_enabled: bool = True
    ident_boost: float = 0.3
    ident_inject: int = 10  # dense·BM25 미포함 정확 일치 조각 주입 상한

    # 검색 저장소 — OpenSearch. PG는 문서·FAQ·폴더·대화·캐시·outbox의 정본이고, 청크(본문·메타·벡터·어휘 필드)는
    # 엔진에만 있다(rag/search/os_client.py 상단). 기동 시 인덱스 존재를 보장한다(ensure_index_soft) — 엔진에 못 붙어도
    # 프로세스는 뜨고 ERROR 로그를 남기며, 검색·색인은 호출 시점에 실패한다.
    #
    # 색인 반영은 즉시가 아니다: 모든 변경은 트랜잭셔널 outbox(rag/outbox.py)로 기록되고 단일 워커 cron이 1분마다
    # 처리한다 — 문서 변경은 검색에 최대 1~5분 뒤 반영될 수 있다(답변 캐시는 즉시 무효화됨).
    # BM25 df 스코프가 인덱스 전체라 테넌트 간 idf가 섞인다(대안은 테넌트별 인덱스). 단일 노드(1샤드·0레플리카)라
    # 노드 다운 = 검색 불가 — 인제스션은 outbox가 pending으로 들고 있다가 복구 후 반영한다.
    opensearch_url: str = "http://localhost:9200"  # 실주소는 .env
    opensearch_index: str = "kms_chunks_v1"
    opensearch_timeout: float = 30.0
    # 커넥션 풀 상한. 이 값이 곧 aiohttp.TCPConnector(limit=)이라 **프로세스 전체의 동시 OS 요청 상한**이다
    # (opensearchpy 기본 10). 한 턴이 3~5회 부른다(dense 쿼리별 + BM25 + mget). ClientTimeout(total)은
    # **큐 대기까지 포함**해서, 풀이 작으면 대기 → 타임아웃 → retry_on_timeout 재시도로 부하가 되레 는다.
    # DB 풀·앱 동시성보다 넉넉하게 둬 여기가 먼저 막히지 않게 한다 — 진짜 상한은 OpenSearch search 스레드풀이고,
    # 이 값은 그 앞에서 큐만 안 만드는 역할이다.
    opensearch_maxsize: int = 100

    # DB 커넥션 풀. 한 턴이 세션 2개를 열지만 요청 세션이 먼저 반납돼 동시 점유는 턴당 약 0.66개다
    #   (동시 40 → 약 26개, 동시 60 → 약 40개). 총 100은 거기에 2.5배 여유다. 풀이 모자라면 QueuePool 에러가 난다.
    # 공용 개발계라 앱·워커·사람 세션이 PostgreSQL max_connections(1000)를 나눠 쓴다 — 무한정 올리지 않는다.
    db_pool_size: int = 50
    db_max_overflow: int = 50  # 총 상한 100

    # 컨텍스트 예산. context_window는 vLLM --max-model-len과 반드시 일치시킬 것.
    context_window: int = 30720
    generation_reserve_tokens: int = (
        3000  # 답변 생성 몫 = max_tokens (한글 ~4,500자 상한 — 폭주 방지용, 정상 답변은 미도달)
    )
    history_budget_tokens: int = 2000  # 이전 대화 참고 몫 (최신 턴부터 예산 소진까지) — 답변 생성용
    condense_history_budget_tokens: int = (
        600  # 질의 재작성(condense)용 히스토리 예산 — 참조 해소엔 최근 몇 턴이면 충분,
    )
    # 길면 이전 답변의 수치가 재작성 질의에 섞여 들어간다
    max_attachments: int = 1  # 컨텍스트 유지 첨부 개수 (넘으면 오래된 것 제외 — FE 고정 안내)
    # 복수 첨부는 "이 문서" 단수 지시의 대상이 모호해진다. 두 문서 대조 수요가 생기면 올린다
    # (ATTACHMENT_MAX_ITEMS와 짝)

    # cache (semantic만)
    # 히트 = floor 이상 후보 + doc집합 동일 + LLM 재사용 판정 승인.
    # 유사도만으로 서빙하는 임계는 두지 않는다 — 유사도 0.96~0.99에서도 답이 반대인 쌍이 있다.
    # floor는 판정 콜을 아낄 후보 게이트일 뿐이고, 0.80은 paraphrase 최저 유사도(약 0.806)를 포괄한다.
    semantic_cache_floor: float = 0.80
    # 미히트 캐시 보존 기간 — last_hit_at 기준이라 히트마다 연장, 인기 답변은 영구 생존.
    # TTL이 아님: 정확성은 무효화·doc집합 비교가 담당하고 이건 죽은 row 위생(LIMIT 1 가림 완화).
    cache_retention_days: int = 90

    # storage
    blob_storage_dir: str = str(PROJECT_ROOT / "docs")

    # 문서 파서 — docling(PDF·DOCX·XLSX). 인제스션(chunk_file) 경로만 대상이고 채팅 첨부(extract_text)는
    # pdfplumber·python-docx·openpyxl을 쓴다 — 첨부는 사용자 대기 경로라 docling 콜드 스타트(수 초)가 그대로 보이고,
    # 변환을 색인과 나누면 색인 작업 뒤에 줄을 선다. 소비처는 rag/chunking/.
    # 아래 메모리·시간 값은 배포 대상(리눅스 x86_64 컨테이너)에서 잰 PDF 기준이다 — macOS 값은 눈금이 다르다.
    # docling을 끄는 스위치는 없다 — 저품질 색인을 조용히 만드느니 docling 실패는 failed로 끝낸다.
    docling_table_mode: str = "accurate"  # fast | accurate. accurate이 병합 셀을 한 셀로 잡는다.
    # RAM 비용은 약 +110MB라 정확도를 택한다
    docling_do_ocr: bool = True  # 이미지로 렌더된 표(TableItem)의 셀을 읽는다. 그림(PictureItem)은
    # VLM 캡션이 읽고, 이 둘은 서로 닿지 않는 경로다. 엔진은 워커 안이
    # 아니라 **원격 Triton**(아래 ocr_kserve_*) — 인프로세스면 워커 피크
    # 메모리가 약 +1.2GB, 변환 시간이 쪽당 약 5배라 밖으로 뺐다.
    # 원격 실패는 문서 failed로 확정된다(조용한 저품질 색인 금지) — Triton이
    # 오래 죽어 있을 때 이 스위치로 끄면 OCR 없이(표 이미지는 빈 격자) 색인이 계속된다.
    docling_do_cell_matching: bool = False  # 켜면 한글 표 셀의 공백이 뭉친다 ("편도 3,000원")
    docling_device: str = "cpu"  # 'auto'는 서버에서 GPU/MPS 탐색을 시도한다. 앱은 GPU 장비에 안 올린다
    docling_num_threads: int = 4  # 워커 vCPU 수에 맞춤
    docling_layout_batch_size: int = 1  # 레이아웃 모델에 한 번에 넣는 쪽 수. 기본 4 → 1로 내리면
    # 리눅스 피크 메모리가 약 2.3~2.5GB → 1.8GB로 준다.
    # 산출물은 동일하고 속도도 손해가 없다. 맥에서는 효과가 편차 내로 묻힌다.
    docling_queue_max_size: int = 4  # 스테이지 간 큐 길이(기본 100). 긴 문서에서 처리 대기 쪽이
    # 쌓이는 상한 — layout_batch와 함께 내려야 효과가 있다
    docling_document_timeout_seconds: float = (
        900.0  # **문서 1건의 변환 마감 — PDF·DOCX·XLSX 전 형식, 그림 캡션 단계 포함.**
    )
    # rag/chunking/convert_process.ConvertProcess가 변환 프로세스를 kill해 끊는다(docling의
    # document_timeout은 PDF 쪽 루프에서만 검사해 캡션·DOCX·XLSX를 못 막는다 — 그래서 안 쓴다).
    # 넘기면 TimeoutError → outbox가 일시 실패로 백오프 재시도(최대 약 24시간).
    # 값 근거: 정상 최악 = PDF 100쪽(쪽당 약 6.1초 → 약 610초, OCR 요청 포함)
    # + 그림 100장(docling_max_pictures, 장당 약 2초 → 약 200초) ≈ 810초.
    # drain 회차 예산·arq job_timeout(rag/outbox.py DRAIN_ROUND_SECONDS)이 이 값에서 계산된다
    # (마감 + 꼬리 300 + 여유 180). 변환 동시성은 프로세스가 하나라 1이다.
    docling_max_num_pages: int = 100  # 병적 입력 가드(PDF만 — DOCX·XLSX엔 쪽 개념이 없어 넘기지 않는다).
    # **초과분을 자르는 게 아니라 문서를 거부한다**(ConversionError → failed).
    # 위 timeout보다 **항상 먼저** 걸리게 둔 값이다 — 쪽당 약 6.1초면 timeout은 약 145쪽에서
    # 걸리는데, 그 사이 구간은 900초를 기다린 끝에 사유 불명의 failed가 되고 그동안 다른 변환도
    # 막힌다(변환 동시성 1). 처리 가능한 100~145쪽 문서도 거부하는 대신 즉시·명확히 거절한다.
    # 올리려면 timeout부터 다시 잴 것. 파일 크기 상한은 라우터(DOC_MAX_FILE_BYTES 10MB)가
    # 업로드 시점에 이미 막으므로 여기선 쪽 수만. 10MB 텍스트 PDF ≈ 100~200쪽
    docling_max_pictures: int = 100  # 문서당 캡션 대상 그림 상한. **넘으면 문서를 거부한다**(결정적 실패 —
    # 앞 N장만 캡션하면 조용한 불완전 색인이 된다).
    # 대상 = docling이 VLM에 보낼 그림(면적 하한 picture_area_threshold 통과분) — 로고·아이콘은
    # 안 센다. 셈은 rag/chunking/document_converter.PictureGuard. 전 형식 공통(PDF·DOCX·XLSX).
    # 값 근거: 그림 100장 캡션 약 200초(동시 1) — PDF 100쪽 최악(약 610초)과
    # 합쳐 약 810초 < 마감 900초. 화면 캡처 매뉴얼(수십 장)까지 받는다.
    docling_artifacts_path: str | None = None  # 모델 가중치 디렉터리. None이면 첫 변환 때 HF에서 내려받는다 —
    # NCP VM은 외부망 확인 필요. 이미지에 미리 굽고 경로를 주는 쪽이 안전

    # 원격 OCR — Triton(worker15 GPU)에 올린 RapidOCR을 docling KserveV2OcrOptions로 부른다.
    # 모델·파라미터는 docling 인프로세스와 동일(serving/triton-ocr/ 참조). 출력은 인프로세스와 글자 내용이 같고,
    # GPU에서는 OCR 박스 좌표가 미세하게 달라 줄 묶음(글머리 유무·문단 병합)만 드물게 다르다.
    # 언어(korean)·use_cls=False는 서버 상수다(serving/triton-ocr/model_repository/ocr/1/model.py).
    # OCR 엔진. triton = 사내(위 docling_do_ocr·아래 ocr_kserve_*) | llm = 상용 — Triton을 안 부르고
    # (docling_do_ocr 무시) 사진 표·스캔 쪽을 캡션과 같은 LLM(vlm_caption_url·model·api_key·temperature)이 읽는다.
    # 라우팅은 rag/chunking/ocr_llm.py. 캡션과 연결을 공유하는 이유: 한 모델이 둘 다 Triton보다 잘 읽고,
    # 같은 값을 두 벌 적으면 한쪽만 바꾸는 실수가 난다 — 다른 모델이 필요해지면 그때 나눈다.
    ocr_provider: Literal["triton", "llm"] = "triton"
    ocr_llm_max_tokens: int = 4000  # 쪽 하나·표 하나의 출력 상한. 캡션보다 큰 이유: 표 한 쪽 출력이 1,500 토큰을 넘는다
    ocr_llm_timeout_seconds: float = (
        180.0  # 요청 하나의 상한. gpt-5-mini 기준 표 하나 최대 약 40초(184칸), 보통 10초 안쪽
    )
    ocr_llm_concurrency: int = (
        4  # 문서 하나 안에서 동시에 보내는 OCR 요청 수. 하나씩이면 스캔 100쪽(쪽 상한)이 약 800초로
    )
    # 변환 마감 900초에 걸린다. 캡션(vlm_caption_concurrency 1)과 다른 이유: 1은 사내 vLLM이 동시
    # 요청에서 결과가 흔들려서이고, 상용 gpt-5 계열은 온도 1.0 필수라 어차피 매번 다르다
    ocr_kserve_url: str = "http://localhost:18893"  # KServe v2 **base**(스킴+호스트:포트). vlm_caption_url과 달리
    # 경로를 붙이지 않는다 — docling이 /v2/models/{model}/infer를 조립한다. 실주소는 .env
    ocr_kserve_model_name: str = "ocr"  # Triton 모델 리포지토리 이름(serving/triton-ocr/model_repository/ocr)
    ocr_kserve_timeout_seconds: float = (
        60.0  # 요청 하나(=비트맵 영역 하나) 상한. docling 기본값 유지. 넘으면 그 문서는 failed
    )
    # 표 이미지 OCR의 한계 — 설정으로 못 고치는 것:
    #   · 한글 단어 사이 띄어쓰기 소실("지원 기준"→"지원기준"). RapidOCR은 PaddleOCR의 use_space_char 같은 스위치가 없고,
    #     한국어 v5 모델이 한글↔숫자 사이에서만 공백을 낸다. 대안 모델 없음(한국어 v5 server 미존재, v6 다국어는 한글 미인식).
    #     검색 영향은 작다(셀 코사인 0.93~0.99, bigram 겹침 100%).
    #   · 기호: ₩→W, ○→O, △→∆, ☑·☐·→·- 누락(글자 사전 한계). 숫자·금액은 거의 정확하다.
    #   · 병합 셀 범위 소실(글자는 남음 — 2단 분류 머리글이 둘 다 필요할 때만 답이 틀림), 세로쓰기 헤더 미인식.
    #   · 작은 글씨·JPEG 재압축·표 안의 표 이미지는 docling 레이아웃이 **표가 아니라 그림**으로 분류 → OCR이 아니라 VLM 캡션
    #     경로(vlm_caption_*)로 간다. 선명하면 캡션이 값을 거의 다 옮기고, 작은 글씨·저품질은 틀린다.
    #   표 이미지 전용 경고는 두지 않는다 — 기존 이미지 비율 경고(rag/ingestion.py IMAGE_WARN_PERCENT)와
    #   업로드 안내(FE 공유)로 전달한다.

    # 그림 캡션 — docling picture description 훅으로 그림을 VLM에 보내 설명을 받아 청크에 싣는다.
    # 대상은 **PictureItem뿐**이다. 격자가 있는 표 이미지는 docling이 TableItem으로 분류해 이 훅을
    # 타지 않는다 — 그쪽은 위 ocr_kserve_*의 원격 OCR이 읽는다.
    # VLM이 죽어도 docling은 예외를 삼키고 빈 캡션을 준다. rag/chunking/document_converter.PictureGuard가
    # 빈 캡션을 CaptionFailedError(일시 실패)로 올려 문서를 백오프 재시도한다 — VLM이 살아나면 저절로 ready가 된다.
    # **on/off 스위치는 두지 않는다.**
    vlm_caption_url: str = "http://localhost:18892/v1/chat/completions"  # **base가 아니라 전체 엔드포인트다**
    # — docling이 이 값을 그대로 POST 대상으로 쓴다. 실주소는 .env
    vlm_caption_model: str = "qwen25-vl"  # OpenAI 호환 body의 model 필드. Qwen2.5-VL-7B-Instruct(Apache 2.0)
    # 상용 VLM 키. 비면 헤더 없이(사내), 있으면 docling 옵션 headers에 Authorization: Bearer로 실린다.
    # 벤더 선택은 따로 없다 — docling이 OpenAI 호환 body 하나만 보내므로 **주소가 곧 벤더**다: OpenAI
    # https://api.openai.com/v1/chat/completions, Gemini https://generativelanguage.googleapis.com/v1beta/openai/chat/completions
    # (Anthropic 호환 창구는 공식 문서상 테스트용 — 캡션은 OpenAI·Gemini로). 키는 .env에만.
    vlm_caption_api_key: str = ""
    vlm_caption_scale: float = 4.0  # **그림 크롭을 VLM에 보낼 때의 해상도 배수(72dpi 기준).**
    # docling 기본 2.0 → 4.0. 낮으면 글자가 안 보여 VLM이 원문에 없는
    # 내용을 지어낸다. PdfPipelineOptions.images_scale과 **다른 값**이다
    # — 그쪽은 캡션 품질과 무관하다
    vlm_caption_temperature: float = 0.0  # OpenAI 호환 body의 temperature. 0 = greedy → 같은 크롭이면 같은 캡션.
    # 샘플링하면 재인제스트마다 표 캡션이 마크다운/평문을 오가 검색 순위가
    # 흔들린다. 캡션은 "원문 그대로 옮기기"라 다양성이 필요 없다.
    vlm_caption_timeout_seconds: float = (
        45.0  # 요청 하나의 상한. docling은 읽기 타임아웃을 **재시도하지 않는다**(Retry read=0)
    )
    # — 넘기면 빈 캡션 → CaptionFailedError로 문서째 재시도. docling 기본 20초는 정상 캡션도 가끔 넘는다.
    # 실패가 보인 그림에서 멈춘다(동시 1이라 첫 실패 뒤 요청이 없다) — VLM이 멈췄을 때의 비용은 이 값 한 번이다
    # (rag/chunking/document_converter.PictureGuard).
    vlm_caption_area_threshold: float = (
        0.01  # 캡션을 요청할 그림의 최소 면적(쪽 면적 대비). docling 기본 0.05면 내용이 있는 작은 캡처
    )
    # (1~5%)가 캡션 없이 자리표시만 남는다. 아이콘·버튼은 0.4% 안팎이라 1%가 둘을 가른다
    # (A4 기준 약 2.5×2.5cm). XLSX는 면적을 시트 전체 대비로 계산해 이 값과 무관하게 걸린다.
    vlm_caption_max_tokens: int = (
        4000  # 캡션 하나의 출력 토큰 상한. docling은 요청에 상한을 안 넣는다 — 7B가 탐욕 디코딩
    )
    # (온도 0)에서 반복 고리에 빠지면 '확정⏎확정⏎…'을 수천 자 쏟는다(화면 캡처처럼 같은 글자가 많은 그림,
    # 프롬프트 문구와 무관). 그대로 두면 반복 글자가 색인되고, 요청 상한(45초)을 넘기면 캡션 실패 → 문서 재시도로
    # 번진다. 정상 캡션은 700토큰 안쪽이다.
    # 4,000인 이유: 상용 gpt-5 계열은 속으로 생각하는 토큰(reasoning)도 이 상한에 들어 그림 하나에 1,000~1,500을 쓴다 —
    # 1,500이면 빈 캡션(finish=length)이 잦다. 대가: 사내 7B가 반복 고리에 빠지면 4,000까지 쏟는다
    # (반복 글자가 더 길게 색인되거나 45초 요청 상한에 걸려 재시도로 번질 수 있다).
    vlm_caption_concurrency: int = 1  # 문서 하나 안에서 동시에 보내는 캡션 요청 수(docling 기본 1).
    # **1이어야 캡션이 결정적이다.** 동시로 보내면 vLLM이 요청을 묶어 계산하고, 묶음 크기에 따라
    # 부동소수 합산 순서가 달라져 온도 0이어도 확률이 비슷한 토큰에서 갈린다 — 재색인마다 캡션이
    # 바뀌어 검색 순위가 흔들린다(동시 2 이상에서 실문서 캡처 그림 일부가 매번 달라진다).
    # 속도 대가: 그림 100장 약 200초(동시 4면 약 70초).
    # 올리려면 vLLM의 배치 불변(batch-invariant) 모드 같은 서버 쪽 결정성부터 확인할 것.

    @model_validator(mode="after")
    def _require_vendor_model_and_key(self) -> Self:
        """상용 공급자를 고르면 모델명과 API 키가 있어야 한다 — 기동 시점에 거부한다.

        빈 모델명·빈 키는 첫 호출(색인·검색·답변 중)에야 실패해 원인(설정 누락)이 늦게 드러난다. 형식만 본다 —
        벤더에 실제로 붙어 보지는 않는다(기동이 외부 가용성에 묶이지 않게). 캡션 VLM은 벤더 선택이 없어 주소로 상용 여부를 가른다.
        """
        for name in ("embed", "rerank"):
            provider = getattr(self, f"{name}_provider")
            if provider != "tei" and not getattr(self, f"{name}_model"):
                raise ValueError(f"{name.upper()}_PROVIDER={provider}이면 {name.upper()}_MODEL이 필요합니다")
            if provider != "tei" and not getattr(self, f"{name}_api_key"):
                raise ValueError(f"{name.upper()}_PROVIDER={provider}이면 {name.upper()}_API_KEY가 필요합니다")
        if self.llm_provider != "vllm" and self.llm_api_key in ("", "EMPTY"):
            raise ValueError(f"LLM_PROVIDER={self.llm_provider}이면 LLM_API_KEY가 필요합니다")
        if any(host in self.vlm_caption_url for host in _COMMERCIAL_VLM_HOSTS) and not self.vlm_caption_api_key:
            raise ValueError("VLM_CAPTION_URL이 상용 주소면 VLM_CAPTION_API_KEY가 필요합니다")
        return self

    @model_validator(mode="after")
    def _reject_renamed_env(self) -> Self:
        """이름을 바꾼 설정이 옛 이름으로 남아 있으면 멈춘다.

        extra="ignore"라 옛 이름은 에러 없이 버려지고 새 이름의 **기본값(localhost)** 으로 조용히 붙는다 —
        "설정했는데 엉뚱한 서버에 붙는" 사고를 막으려고 OS 환경변수와 두 env 파일을 직접 본다.
        옛 이름을 받아 주는 호환이 아니다.
        """
        found = [k for k in os.environ if k.upper() in _RENAMED_ENV]
        for f in cast("tuple[str, ...]", self.model_config["env_file"]):  # 위 SettingsConfigDict가 튜플로 준다
            if Path(f).exists():
                found += [k for k in dotenv_values(f) if k.upper() in _RENAMED_ENV]
        if found:
            hints = ", ".join(f"{k.upper()} → {_RENAMED_ENV[k.upper()]}" for k in dict.fromkeys(found))
            raise ValueError(f"이름이 바뀐 설정이 남아 있습니다 — 새 이름으로 바꾸세요: {hints}")
        return self


# 캡션 VLM은 "주소가 곧 벤더"라 상용 여부를 주소로 판정한다(vlm_caption_url 주석).
_COMMERCIAL_VLM_HOSTS = ("api.openai.com", "generativelanguage.googleapis.com", "api.anthropic.com")

# 옛 환경변수 이름 → 새 이름. 기동 시 Settings._reject_renamed_env가 본다.
_RENAMED_ENV = {
    "VLLM_BASE_URL": "LLM_BASE_URL",
    "VLLM_MODEL": "LLM_MODEL",
    "LLM_ENABLE_THINKING": "LLM_REASONING_EFFORT",
}


# 모듈 import가 곧 프로세스당 1회이므로 이 전역 자체가 싱글톤이다.
settings = Settings()
