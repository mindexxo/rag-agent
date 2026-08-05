# Windows 로컬 DB 셋업 (pgvector + 스키마)

FE 개발자가 **개발계 대신 로컬 DB/Redis**로 백엔드를 돌릴 때의 DB 준비 절차.

- **전제**: Windows에 PostgreSQL 설치 완료 (예: EDB 설치관리자, v17). `psql` 사용 가능.
- **범위**: pgvector 확장 설치 → DB 생성 → 스키마(테이블 DDL) 적용.
- LLM/임베딩(vLLM·TEI)은 로컬에 못 띄우므로 **사내망(VPN)** 그대로 사용. DB·Redis만 로컬로 바꾼다.

---

## 1. pgvector 확장 설치 (Windows = 소스 빌드)

Windows엔 pgvector 공식 바이너리가 없어 **직접 빌드**해야 한다. 우리 스키마의 `chunks.dense VECTOR(1024)`와
hnsw 인덱스가 이 확장을 요구한다.

**사전 준비**: Visual Studio 2022 (Community 무료) 설치 시 **"C++를 사용한 데스크톱 개발(Desktop development with C++)"** 워크로드 포함.

**빌드 + 설치** (명령 프롬프트):
```bat
REM PostgreSQL 설치 경로로 맞출 것 (버전 폴더 확인)
set "PGROOT=C:\Program Files\PostgreSQL\17"

cd %TEMP%
git clone --branch v0.8.0 https://github.com/pgvector/pgvector.git
cd pgvector

REM VS 빌드 환경 로드 (에디션 경로는 설치에 맞게: Community/Professional)
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"

nmake /F Makefile.win
nmake /F Makefile.win install
```

- `--branch v0.8.0` : pgvector 최신 릴리스 태그로 (github.com/pgvector/pgvector/releases 확인). **0.5.0 이상**이면 hnsw 지원 → 우리 스키마 OK.
- `PGROOT`가 **실제 설치된 PostgreSQL 버전 폴더**를 가리켜야 한다. 버전이 안 맞으면 뒤 `CREATE EXTENSION`에서 실패.


---

## 2. 데이터베이스 생성

```bat
psql -U postgres
```
```sql
CREATE DATABASE kms;
\q
```
(설치 시 정한 `postgres` 비밀번호 사용)

---

## 3. 확장 + 스키마(테이블 DDL) 적용

`kms` DB에 접속해서 아래 SQL을 전부 실행한다. (DBeaver에 붙여넣어도 되고, `psql -U postgres -d kms` 접속 후 붙여도 됨)
확장 활성화 + 테이블 8개 + 인덱스가 모두 포함돼 있고, `IF NOT EXISTS`라 여러 번 실행해도 안전하다.
로컬 postgres는 슈퍼유저라 `CREATE EXTENSION`이 그냥 통과된다 (개발계처럼 권한·cdb_admin 이슈 없음). 스키마는 `public`.

```sql
-- 확장 (vector만 필요 — pg_trgm은 현재 미사용이라 생략)
CREATE EXTENSION IF NOT EXISTS vector;

-- 폴더 (검색 참조 제어용 1단 그룹)
CREATE TABLE IF NOT EXISTS folders (
    id            BIGSERIAL PRIMARY KEY,
    tenant_id     TEXT        NOT NULL,
    name          TEXT        NOT NULL,
    description   TEXT,
    is_searchable BOOLEAN     NOT NULL DEFAULT TRUE,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (tenant_id, name)
);

-- 문서
CREATE TABLE IF NOT EXISTS documents (
    id             BIGSERIAL PRIMARY KEY,
    tenant_id      TEXT        NOT NULL,
    filename       TEXT        NOT NULL,
    mime           TEXT        NOT NULL,
    blob_path      TEXT        NOT NULL,
    version        INTEGER     NOT NULL DEFAULT 1,
    version_label  TEXT,
    is_active      BOOLEAN     NOT NULL DEFAULT FALSE,
    status         TEXT        NOT NULL DEFAULT 'pending',
    status_reason  TEXT,
    page_count     INTEGER,
    char_count     INTEGER,
    uploaded_by    TEXT,
    uploaded_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    indexed_at     TIMESTAMPTZ,
    description   TEXT,
    folder_id      BIGINT      REFERENCES folders(id) ON DELETE SET NULL,
    is_searchable  BOOLEAN     NOT NULL DEFAULT TRUE,
    UNIQUE (tenant_id, filename, version)
);
CREATE INDEX IF NOT EXISTS idx_docs_tenant_status
    ON documents (tenant_id, status);
CREATE UNIQUE INDEX IF NOT EXISTS uq_docs_one_active_per_name
    ON documents (tenant_id, filename)
    WHERE is_active = TRUE;

-- FAQ
CREATE TABLE IF NOT EXISTS faqs (
    id         BIGSERIAL PRIMARY KEY,
    tenant_id  TEXT        NOT NULL,
    question   TEXT        NOT NULL,
    variants   JSONB       NOT NULL DEFAULT '[]',
    answer     TEXT        NOT NULL,
    is_active  BOOLEAN     NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_faqs_tenant ON faqs (tenant_id);

-- 청크 (검색 인덱스, dense-only 1024차원)
CREATE TABLE IF NOT EXISTS chunks (
    id            BIGSERIAL PRIMARY KEY,
    document_id   BIGINT       REFERENCES documents(id) ON DELETE CASCADE,
    faq_id        BIGINT       REFERENCES faqs(id) ON DELETE CASCADE,
    tenant_id     TEXT         NOT NULL,
    chunk_index   INTEGER      NOT NULL,
    text          TEXT         NOT NULL,
    token_count   INTEGER,
    page          INTEGER,
    heading_path  TEXT[]       NOT NULL DEFAULT '{}',
    metadata      JSONB        NOT NULL DEFAULT '{}'::jsonb,
    dense         VECTOR(1024) NOT NULL,
    UNIQUE (document_id, chunk_index),
    CHECK ((document_id IS NOT NULL AND faq_id IS NULL) OR (document_id IS NULL AND faq_id IS NOT NULL))
);
CREATE UNIQUE INDEX IF NOT EXISTS uq_chunks_faq ON chunks (faq_id) WHERE faq_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_chunks_tenant_doc
    ON chunks (tenant_id, document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_dense_hnsw
    ON chunks USING hnsw (dense vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- LLM 응답 캐시
CREATE TABLE IF NOT EXISTS answer_cache (
    id               BIGSERIAL PRIMARY KEY,
    tenant_id        TEXT         NOT NULL,
    cache_key        TEXT         NOT NULL,
    query_text       TEXT         NOT NULL,
    query_embedding  VECTOR(1024) NOT NULL,
    answer           TEXT         NOT NULL,
    sources          JSONB        NOT NULL,
    source_doc_ids   BIGINT[]     NOT NULL,
    model            TEXT         NOT NULL,
    created_at       TIMESTAMPTZ  NOT NULL DEFAULT now(),
    last_hit_at      TIMESTAMPTZ  NOT NULL DEFAULT now(),
    hit_count        INTEGER      NOT NULL DEFAULT 0,
    UNIQUE (tenant_id, cache_key)
);
CREATE INDEX IF NOT EXISTS idx_cache_src_doc_ids
    ON answer_cache USING GIN (source_doc_ids);
CREATE INDEX IF NOT EXISTS idx_cache_embedding_hnsw
    ON answer_cache USING hnsw (query_embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
CREATE INDEX IF NOT EXISTS idx_cache_tenant_created
    ON answer_cache (tenant_id, created_at DESC);

-- 대화 (멀티턴)
CREATE TABLE IF NOT EXISTS conversations (
    id           BIGSERIAL PRIMARY KEY,
    tenant_id    TEXT         NOT NULL,
    title        TEXT,
    created_by   TEXT,
    created_at   TIMESTAMPTZ  NOT NULL DEFAULT now(),
    last_used_at TIMESTAMPTZ  NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_conv_tenant_last
    ON conversations (tenant_id, last_used_at DESC);

CREATE TABLE IF NOT EXISTS messages (
    id              BIGSERIAL PRIMARY KEY,
    conversation_id BIGINT       NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    tenant_id       TEXT         NOT NULL,
    role            TEXT         NOT NULL,
    content         TEXT         NOT NULL,
    standalone_query TEXT,
    sources         JSONB,
    attachments     JSONB,
    status          TEXT         NOT NULL DEFAULT 'done',
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT now(),
    user_id         TEXT,
    latency_ms      INTEGER,
    cache_kind      TEXT,
    cited_docs      JSONB,
    is_refusal      BOOLEAN,
    question_message_id BIGINT REFERENCES messages(id) ON DELETE SET NULL
);
CREATE INDEX IF NOT EXISTS idx_msg_conv_created
    ON messages (conversation_id, created_at);

-- 사용량 제한 / 쿼터
CREATE TABLE IF NOT EXISTS tenant_quotas (
    tenant_id           TEXT        PRIMARY KEY,
    rpm_limit           INTEGER     NOT NULL DEFAULT 60,
    user_rpm_limit      INTEGER     NOT NULL DEFAULT 20,
    daily_query_limit   INTEGER     NOT NULL DEFAULT 5000,
    daily_upload_mb     INTEGER     NOT NULL DEFAULT 500,
    concurrency_limit   INTEGER     NOT NULL DEFAULT 8,
    user_concurrency    INTEGER     NOT NULL DEFAULT 3,
    deep_mode_enabled   BOOLEAN     NOT NULL DEFAULT TRUE,
    deep_rpm_limit      INTEGER     NOT NULL DEFAULT 20,
    is_blocked          BOOLEAN     NOT NULL DEFAULT FALSE,
    block_reason        TEXT,
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

> 이 SQL은 repo의 `schema.sql` 기반이다 (현재 미사용인 `pg_trgm` 확장만 생략). `schema.sql`이 바뀌면 이 문서도 같이 갱신할 것.

---

## 4. 적용 확인

```bat
psql -U postgres -d kms -c "\dt"
```
→ 테이블 8개가 보이면 성공.

확장 확인:
```bat
psql -U postgres -d kms -c "SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';"
```
→ `vector` 한 줄이면 완료.

---

## 5. 앱 연결 설정 (`.env`)

repo 루트에 `.env` 생성 (gitignore됨). `.env.dev` 위에 **DB만 로컬로 덮어쓴다**:

```
DATABASE_URL=postgresql+asyncpg://postgres:<비밀번호>@localhost:5432/kms
DB_SEARCH_PATH=public
REDIS_URL=redis://localhost:6379/0
```

- `DB_SEARCH_PATH=public` — 로컬은 `public` 스키마라 이렇게. (개발계는 `cc_kms_test,cdb_admin,public`)
- vLLM/TEI 주소는 `.env.dev` 값(사내 GPU) 그대로 상속 → **VPN 필요**.

---

## 6. Redis (별도 설치)

앱/워커가 Redis도 필요하다. Windows 네이티브 Redis는 공식 미지원이라 **Memurai**(Redis 호환, 개발용 무료)가 가장 간단하다.
- 설치하면 기본 포트 `6379`로 뜬다 → 위 `.env`의 `REDIS_URL` 그대로 사용.
- 채팅·검색만 테스트하면 Redis 없이도 동작하지만, **문서 업로드 인덱싱(워커)까지 테스트하려면 필수**.

---

## 다음: 백엔드 실행

DB·Redis 준비 후 (VPN 연결 상태에서):
```bat
uvicorn main:app --reload --port 8000
arq rag.worker.WorkerSettings
```
확인: `curl http://localhost:8000/kms/conversations -H "X-Tenant-Id: demo"` → `[]`
