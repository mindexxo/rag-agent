"""애플리케이션 설정 모듈.

.env 파일과 OS 환경변수에서 값을 읽어 Settings 객체로 변환.
다른 모듈에서 `from config import settings`로 가져다 씀.

우선순위: OS 환경변수 > .env 파일 > 코드 기본값.
"""
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# 이 파일(config.py)이 앱 루트에 있으므로, 그 디렉터리가 프로젝트 루트.
# cwd(서버 실행 위치)에 의존하지 않도록 절대경로로 고정한다.
PROJECT_ROOT = Path(__file__).resolve().parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        # .env.dev(개발계 공용 기본, git 커밋됨) → .env(로컬 오버라이드, gitignore) 순.
        # 뒤 파일이 우선이라, 로컬에 .env가 있으면 그게 이기고 없으면 .env.dev를 쓴다.
        # cwd 무관하게 절대경로 (부팅 이식성).
        env_file=(str(PROJECT_ROOT / ".env"), str(PROJECT_ROOT / ".env.dev")),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # application
    app_env: str = "local"
    log_level: str = "INFO"

    # database
    database_url: str
    database_admin_url: str | None = None
    # 커넥션 search_path — 앱 쿼리는 스키마를 명시하지 않으므로(FROM conversations) 여기서 해소.
    # vector 타입이 cdb_admin에 설치돼 있어 cdb_admin도 포함. 로컬(public 스키마)은 뒤의 public으로 폴백.
    # 환경별로 .env의 DB_SEARCH_PATH로 오버라이드. (존재하지 않는 스키마는 무시되므로 안전)
    db_search_path: str = "cc_kms_test,cdb_admin,public"

    # redis
    redis_url: str = "redis://localhost:6379/0"

    # 동시 in-flight 제한 (F6). GPU/모델 미정이라 env로 튜닝. tenant_quotas 행이 있으면 그게 우선.
    concurrency_limit_default: int = 10   # 테넌트 동시 in-flight 기본 (quota 행 없을 때)
    user_concurrency_default: int = 3     # 사용자(X-User-Id)별 동시 in-flight 기본
    inflight_max_seconds: int = 120       # in-flight 유령 판정 — 넘으면 카운트서 제거(강제종료 아님)

    # security
    default_tenant_id: str = "demo"

    # CORS — FE가 다른 origin에서 서빙될 때(배포·프록시 없는 로컬) 허용 목록.
    # 쉼표 구분 (.env 예: CORS_ALLOW_ORIGINS=http://localhost:5173,https://iccs.example.com)
    cors_allow_origins: str = "http://localhost:5173,http://localhost:3000"

    # vllm
    vllm_base_url: str = "http://localhost:11434/v1"
    vllm_model: str = "qwen3:4b"
    llm_temperature: float | None = None  # None이면 모델/서버 기본값(Qwen3 0.6). 경계 문항 일관성 실험용 — 낮을수록 결정적

    # embedding (F99: TEI 원격 서버, dense-only). 저장 벡터도 이 서버 출력이어야 검색 정상 — 재인제스트로 일치 보장
    embed_base_url: str = "http://localhost:38889"    # TEI 임베딩 서버 (BGE-M3, /embed) — 실주소는 .env
    embed_timeout: float = 30.0

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
    max_attachments: int = 2                 # 컨텍스트 유지 첨부 개수 (넘으면 오래된 것 제외 — FE 고정 안내)

    # cache (exact 캐시 제거됨 — semantic만)
    semantic_cache_threshold: float = 0.90

    # guardrail
    guardrail_output_enabled: bool = False

    # storage
    blob_storage_dir: str = str(PROJECT_ROOT / "docs")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()

