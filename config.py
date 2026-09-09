"""애플리케이션 설정 모듈.

.env 파일과 OS 환경변수에서 값을 읽어 Settings 객체로 변환.
다른 모듈에서 `from config import settings`로 가져다 씀.

우선순위: OS 환경변수 > .env 파일 > 코드 기본값.
"""
from pathlib import Path

from pydantic import model_validator
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
    # 하이브리드 어휘 채널(#135): FTS(bigram tsvector) 회수 → 앱 BM25 재점수 → dense 미포함
    # 상위를 리랭커 풀에 주입. 현행 모의 코퍼스 실측은 이득 0(주입 상방 0/450, 토크나이저
    # bigram·kiwi 둘 다 — #133 A/B)이지만 실코퍼스(상품코드·고유명 분포) 대비 정식 채널로
    # 켠다(사용자 결정). 통계(df·N·avgdl)는 질의 시 계산 — 테넌트 수만 청크 도달 시
    # pg_search 설치 가능 여부 확인 후 불가면 통계 테이블 승격(이슈 #135 결정 기록).
    hybrid_lexical_enabled: bool = True
    hybrid_lexical_inject: int = 15                   # dense 미포함 BM25 상위 주입 수 (어블레이션 조립과 동일)
    rerank_base_url: str = "http://localhost:38890"   # TEI 리랭커 서버 (bge-reranker-v2-m3, /rerank) — 실주소는 .env
    rerank_timeout: float = 30.0

    # 검색 백엔드 (#139) — 'pg' | 'opensearch'. **기본은 pg이고, 바꾸지 마라.**
    # 'opensearch'는 "PG 단일 스택 대신 검색 엔진을 쓰면 나은가"를 실측하기 위한 실험 경로다.
    # 현행 근거로는 이득이 0이다(eval/report_os_ablation_v1.md: kNN 구현 차이 0·BM25 구현 차이
    # 0·토크나이저 차이 잡음). 채택/기각 판정은 실문서(#138) 대기.
    #
    # **운영으로 켜기 전 남은 것** (정합 관리 3층의 현재 수준은 rag/opensearch.py 상단이 정본):
    #  1) 색인 반영의 **보장 수준**. 실무 표준 구성을 택해 검색 결과의 텍스트·메타를 엔진이
    #     돌려주므로(PG로 되묻지 않는다) **색인이 서빙의 정본**이다. 쓰기 경로가 즉시 반영하고
    #     (청크 변경=재색인, 메타 변경=_update_by_query 부분 갱신) 실패는 지표로 드러나지만,
    #     반영이 **최선노력**이라 엔진 장애 중의 변경은 `_os_index --repair`까지 낡은 채로
    #     남는다. 게다가 --repair는 id 집합만 대조하므로 **메타 드리프트는 못 잡는다**.
    #     운영 채택 시에는 outbox 테이블 + arq 드레인으로 이 구멍을 먼저 닫아야 한다
    #     (사유·구조는 rag/opensearch.py의 "정합 관리" 절이 정본).
    #  2) PG `chunks.dense`의 처분. 지금은 pg가 기본이라 PG의 벡터가 **정본**이고 OpenSearch
    #     쪽이 사본이다 — 채택하면 관계가 뒤집히고, 그때 PG 벡터를 남길 근거는 약하다:
    #     OS 스냅샷이 이미 백업이고, 재임베딩 비결정성(1.4e-4)은 검색 품질에 잡음 수준이며,
    #     모델을 바꾸면 어차피 전량 재임베딩이다. 실측 크기는 컬럼 3.4MB + HNSW 인덱스 6.5MB
    #     (868청크, 2026-09-10) — 수십만 청크면 GB 단위가 되고 백업·WAL로 증폭된다.
    #     채택 시 순서: ① `idx_chunks_dense_hnsw` DROP(검색을 OS가 하므로 순수 이득)
    #     ② OS 스냅샷 설정·복구 리허설을 마친 **뒤** 컬럼 DROP. 순서를 바꾸면 백업 없는
    #     유일 저장소가 되는 구간이 생긴다. 채택 전에는 손대지 마라 — 이득 없이 롤백 길만 막는다.
    #  3) BM25 통계 스코프. 단일 인덱스라 Lucene의 df가 인덱스 전체다 — 다른 테넌트의 데이터가
    #     우리 테넌트의 idf를 움직인다(실측: 다른 세션이 실문서 211청크를 넣자 색인 대상이
    #     602→813. 단 그 35% 증가로 표시 지표는 미동 없었다 — 리포트 참조). 앱 BM25는 통계를
    #     테넌트 단위로 잡아 이 성질이 없다. 테넌트별 인덱스 또는 스코프 설계가 대안이다.
    # 청크 본문·메타는 어느 백엔드든 PG에서 읽는다(PG가 정본) — rag/opensearch.py docstring.
    search_backend: str = "pg"
    opensearch_url: str = "http://localhost:9200"     # 실주소는 .env (개발계 이관 시 포트 23336)
    opensearch_index: str = "kms_chunks_v1"
    opensearch_timeout: float = 30.0

    # PG가 검색용 파생 컬럼(chunks.dense·lex_tsv·lex_len)을 계속 들지 (#139).
    # **search_backend와 별개 스위치인 이유**: 둘을 한 값으로 묶으면 백엔드를 바꾸는 순간
    # 스키마 마이그레이션이 강제되고, 마이그레이션 전에 업로드하면 dense NOT NULL 위반으로
    # 인제스션이 죽는다. 전환은 두 단계여야 한다 —
    #   1단계: search_backend=opensearch (컬럼은 그대로) → 검색만 엔진으로. 되돌리기 자유.
    #   2단계: 아래를 False로 + 스키마 마이그레이션 → PG는 본문만 든다.
    #
    # False로 바꾸면:
    #  - 인제스션·FAQ 색인이 PG에 dense/lex를 쓰지 않고, 계산한 임베딩을 OpenSearch로 직접 넘긴다.
    #  - 재색인·재동기화(eval/_os_index)는 PG 벡터를 못 읽으므로 **chunks.text에서 임베딩 입력을
    #    재조립해 다시 임베딩한다**(rag/opensearch.lex_text가 인제스션과 동일 조립임이 근거).
    #    즉 색인 재구축 비용이 '몇 초'에서 'TEI 임베딩 배치'로 올라간다 — 그게 이 선택의 값이다.
    #  - `search_backend='pg'`로 되돌릴 수 없다(어휘·벡터 컬럼이 비어 있다). 아래 검증이 막는다.
    #
    # 선결 마이그레이션 순서(schema.sql 하단에 기록, 실행은 사람이):
    #   ① ALTER TABLE chunks ALTER COLUMN dense DROP NOT NULL;   ← 이거만 하면 False 운영 가능
    #   ② DROP INDEX idx_chunks_dense_hnsw; DROP INDEX idx_chunks_lex_gin;  ← 검색을 안 하므로
    #   ③ OpenSearch 스냅샷 설정 + 복구 리허설을 **마친 뒤**
    #      ALTER TABLE chunks DROP COLUMN dense, DROP COLUMN lex_tsv, DROP COLUMN lex_len;
    # ②③ 사이를 건너뛰면 백업 없는 유일 저장소가 되는 구간이 생긴다.
    pg_vector_columns: bool = True

    @model_validator(mode='after')
    def _check_search_backend(self):
        """조합 검증 — 조용히 빈 결과를 내는 설정을 기동 시점에 막는다.

        pg 백엔드 + 파생 컬럼 없음은 "dense가 비었는데 dense로 검색"이라 검색이 0건이 된다.
        런타임에 no_evidence로만 드러나면 원인을 찾기 어려우므로 여기서 끊는다.
        """
        if self.search_backend not in ('pg', 'opensearch'):
            raise ValueError(f"search_backend는 'pg'|'opensearch'만: {self.search_backend!r}")
        if self.search_backend == 'pg' and not self.pg_vector_columns:
            raise ValueError(
                "search_backend='pg'인데 pg_vector_columns=False — PG 검색이 쓸 벡터·어휘 "
                "컬럼이 없어 결과가 항상 비게 된다. 되돌리려면 재인제스트(재임베딩)가 필요하다.")
        return self

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


# 모듈 import가 곧 프로세스당 1회이므로 이 전역 자체가 싱글톤이다 (팩토리·캐시 불필요).
settings = Settings()

