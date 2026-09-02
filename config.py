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

    # 질의 재작성 의미 확장(#5). on이면 '멀티턴에서만' condense 자리 1콜로 멀티쿼리(재작성 1 +
    # 어휘 변형 2)를 뽑아 검색 — rerank on이면 쿼리별 채점 max-pool 정렬, rerank off/실패면 RRF 폴백.
    # 단일턴은 on이어도 현행 경로 그대로(LLM 스킵) — 단일턴 확장은 실측상 손실이라 게이트(service).
    # 기본 on (#5 검증 완료: 재작성 98%, mt Hit@1 +2.2pp·R@20 1.000, 생성 EPCov/Cite +2.8/+5.5pp).
    # off면 코드·프롬프트 모두 도입 전과 동일 — 문제 시 .env 한 줄로 원복.
    condense_multi_query_enabled: bool = True

    # reranker (F99: TEI /rerank, cross-encoder 재정렬). on/off 토글 한 줄.
    rerank_enabled: bool = True                       # False면 dense-only 순서 그대로 (리랭크 skip). .env로 오버라이드 가능
    rerank_base_url: str = "http://localhost:38890"   # TEI 리랭커 서버 (bge-reranker-v2-m3, /rerank) — 실주소는 .env
    rerank_timeout: float = 30.0

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
    # 0.90→0.95 상향(#16): 검색용 임베딩은 부정("가능/불가능" 0.949)·수치("7일/14일" 0.918) 차이에
    # 둔감해 0.90에선 오답 재생 실측. 0.95 위 누수(~0.99까지 실측, #113)는 기계 가드
    # (cache.guard_blocks)가 막는다. 하향은 기각(#113 40쌍 실측): 0.92~0.95 대역에 가드로 못
    # 잡는 답-상이 쌍(0.9218)이 실존하는데 얻는 히트는 1건 — 재론은 섀도 모드 실데이터로.
    semantic_cache_threshold: float = 0.95
    # 2차 검증 판정기(#113): 임계 아래 [floor, threshold) 대역의 후보를 LLM이 재사용 판정.
    # 임계가 잘라내던 구어체 paraphrase를 회수한다(40쌍 실측: 판정 39/40·오답 방향 0·흔들림 0).
    # 판정 콜은 대역 후보가 존재할 때만 1회 — 히트(≥임계)는 판정 없이 기존 경로 그대로.
    semantic_cache_verifier: bool = True
    semantic_cache_verifier_floor: float = 0.80   # 실측 paraphrase 최저 0.8058 포괄, 그 아래는 잡음
    # 미히트 캐시 보존 기간(#16) — last_hit_at 기준이라 히트마다 연장, 인기 답변은 영구 생존.
    # TTL이 아님: 정확성은 무효화·doc집합 비교가 담당하고 이건 죽은 row 위생(LIMIT 1 가림 완화).
    cache_retention_days: int = 90

    # storage
    blob_storage_dir: str = str(PROJECT_ROOT / "docs")


# 모듈 import가 곧 프로세스당 1회이므로 이 전역 자체가 싱글톤이다 (팩토리·캐시 불필요).
settings = Settings()

