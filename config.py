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
    concurrency_limit_default: int = 10   # 테넌트 동시 in-flight 기본 (quota 행 없을 때)
    user_concurrency_default: int = 10    # 사용자(X-User-Id)별 동시 in-flight 기본
    inflight_max_seconds: int = 120       # in-flight 유령 판정 — 넘으면 카운트서 제거(강제종료 아님)

    # CORS — FE가 다른 origin에서 서빙될 때(배포·프록시 없는 로컬) 허용 목록.
    # 쉼표 구분 (.env 예: CORS_ALLOW_ORIGINS=http://localhost:5173,https://iccs.example.com)
    cors_allow_origins: str = "http://localhost:5173,http://localhost:3000"

    # vllm
    vllm_base_url: str = "http://localhost:11434/v1"
    vllm_model: str = "qwen3:4b"
    # 문서 기반 사실 제공 서비스라 다양성이 손해 — 낮게 고정 (2026-08-05).
    # None이면 서버 기본값(Qwen3 non-thinking 권장 0.7)이 쓰이는데, 실측 4회 반복에서 매번 다른 답이
    # 나왔다. is_refusal의 문자열 판정·인용 형식 파싱·시맨틱 캐시가 모두 결정성에 의존해 불리하다.
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
    max_attachments: int = 2                 # 컨텍스트 유지 첨부 개수 (넘으면 오래된 것 제외 — FE 고정 안내)

    # cache (exact 캐시 제거됨 — semantic만)
    # 0.90→0.95 상향(#16): 검색용 임베딩은 부정("가능/불가능" 0.949)·수치("7일/14일" 0.918) 차이에
    # 둔감해 0.90에선 오답 재생 실측. 0.95도 긴 부정 쌍(~0.97)은 누수 가능 — 운영 데이터로 재판단.
    semantic_cache_threshold: float = 0.95
    # 미히트 캐시 보존 기간(#16) — last_hit_at 기준이라 히트마다 연장, 인기 답변은 영구 생존.
    # TTL이 아님: 정확성은 무효화·doc집합 비교가 담당하고 이건 죽은 row 위생(LIMIT 1 가림 완화).
    cache_retention_days: int = 90

    # storage
    blob_storage_dir: str = str(PROJECT_ROOT / "docs")


# 모듈 import가 곧 프로세스당 1회이므로 이 전역 자체가 싱글톤이다 (팩토리·캐시 불필요).
settings = Settings()

