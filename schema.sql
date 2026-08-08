-- schema.sql — Phase 1 테이블 + 인덱스 (RLS는 rls.sql 별도)
-- 적용: psql postgres -f schema.sql
-- idempotent: 여러 번 돌려도 안전.

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- ---------- 폴더 (F2: 1단 그룹, 검색 참조 제어 전용) ----------
-- 트리 없음 — 계층이 필요해지면 parent_id 컬럼 추가로 확장.
CREATE TABLE IF NOT EXISTS folders (
    id            BIGSERIAL PRIMARY KEY,
    tenant_id     TEXT        NOT NULL,
    name          TEXT        NOT NULL,
    description   TEXT,
    is_searchable BOOLEAN     NOT NULL DEFAULT TRUE,   -- 폴더 단위 참조 on/off
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (tenant_id, name)
);

-- ---------- 문서 ----------
CREATE TABLE IF NOT EXISTS documents (
    id             BIGSERIAL PRIMARY KEY,
    tenant_id      TEXT        NOT NULL,
    filename       TEXT        NOT NULL,
    mime           TEXT        NOT NULL,
    blob_path      TEXT        NOT NULL,
    version        INTEGER     NOT NULL DEFAULT 1,
    is_active      BOOLEAN     NOT NULL DEFAULT FALSE,
    status         TEXT        NOT NULL DEFAULT 'pending',
    status_reason  TEXT,
    page_count     INTEGER,
    char_count     INTEGER,
    uploaded_by    TEXT,
    uploaded_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    indexed_at     TIMESTAMPTZ,
    -- F1a: 표 설명 (xlsx 검색 보강용 — 업로드 시 입력, 워커가 청크에 병합). 표는 의미적으로 빈약해 검색 다리 필요
    description   TEXT,
    -- F2: 폴더 소속(미분류 허용) + 문서 단위 참조 on/off (is_active는 버전 정책 소관 — 별개)
    folder_id      BIGINT      REFERENCES folders(id) ON DELETE SET NULL,
    is_searchable  BOOLEAN     NOT NULL DEFAULT TRUE,
    UNIQUE (tenant_id, filename, version)
);
CREATE INDEX IF NOT EXISTS idx_docs_tenant_status
    ON documents (tenant_id, status);
CREATE UNIQUE INDEX IF NOT EXISTS uq_docs_one_active_per_name
    ON documents (tenant_id, filename)
    WHERE is_active = TRUE;

-- ---------- FAQ (F3: 전용 저장. 검색은 chunks로 통합 — 관문·원문반환 없음) ----------
CREATE TABLE IF NOT EXISTS faqs (
    id         BIGSERIAL PRIMARY KEY,
    tenant_id  TEXT        NOT NULL,
    question   TEXT        NOT NULL,
    variants   JSONB       NOT NULL DEFAULT '[]',   -- 유사 질문(구어체 변형) 배열
    answer     TEXT        NOT NULL,
    is_active  BOOLEAN     NOT NULL DEFAULT TRUE,   -- 항목 단위 검색 on/off
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_faqs_tenant ON faqs (tenant_id);

-- ---------- 청크 (검색 인덱스 — 출처: document 또는 faq 정확히 하나) ----------
CREATE TABLE IF NOT EXISTS chunks (
    id            BIGSERIAL PRIMARY KEY,
    document_id   BIGINT       REFERENCES documents(id) ON DELETE CASCADE,   -- 문서 출처 (F3부터 nullable)
    faq_id        BIGINT       REFERENCES faqs(id) ON DELETE CASCADE,        -- FAQ 출처 (F3, 항목당 1청크)
    tenant_id     TEXT         NOT NULL,
    chunk_index   INTEGER      NOT NULL,
    text          TEXT         NOT NULL,
    token_count   INTEGER,
    page          INTEGER,
    heading_path  TEXT[]       NOT NULL DEFAULT '{}',
    metadata      JSONB        NOT NULL DEFAULT '{}'::jsonb,
    dense         VECTOR(1024) NOT NULL,
    -- [dense-only, F99] sparse 제거. 하이브리드 원복 시 해제 + 재인제스트
    -- sparse        SPARSEVEC(250002) NOT NULL,
    UNIQUE (document_id, chunk_index),
    CHECK ((document_id IS NOT NULL AND faq_id IS NULL) OR (document_id IS NULL AND faq_id IS NOT NULL))
);
CREATE UNIQUE INDEX IF NOT EXISTS uq_chunks_faq ON chunks (faq_id) WHERE faq_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_chunks_tenant_doc
    ON chunks (tenant_id, document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_dense_hnsw
    ON chunks USING hnsw (dense vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
-- [dense-only, F99] sparse 인덱스 제거. 하이브리드 원복 시 해제
-- CREATE INDEX IF NOT EXISTS idx_chunks_sparse_hnsw
--     ON chunks USING hnsw (sparse sparsevec_ip_ops)
--     WITH (m = 16, ef_construction = 64);

-- ---------- LLM 응답 캐시 ----------
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

-- ---------- 대화 (멀티턴) ----------
CREATE TABLE IF NOT EXISTS conversations (
    id           BIGSERIAL PRIMARY KEY,
    tenant_id    TEXT         NOT NULL,
    title        TEXT,
    created_by   TEXT,                       -- 생성 사용자 (#10부터 항상 저장 — X-User-Id 미전송 시 'test-user')
    created_at   TIMESTAMPTZ  NOT NULL DEFAULT now(),
    last_used_at TIMESTAMPTZ  NOT NULL DEFAULT now(),
    deleted_at   TIMESTAMPTZ                 -- 소프트 삭제 시각 (#10). NULL=활성 — 삭제 시각 자체가 감사 데이터
);
CREATE INDEX IF NOT EXISTS idx_conv_tenant_last
    ON conversations (tenant_id, last_used_at DESC);

CREATE TABLE IF NOT EXISTS messages (
    id              BIGSERIAL PRIMARY KEY,
    conversation_id BIGINT       NOT NULL REFERENCES conversations(id),  -- CASCADE 해제(#10): 삭제는 소프트 — 실수 하드 DELETE에도 이력 보존
    tenant_id       TEXT         NOT NULL,
    role            TEXT         NOT NULL,
    content         TEXT         NOT NULL,
    standalone_query TEXT,
    sources         JSONB,
    -- 채팅 첨부: 첨부한 턴의 user 메시지에 추출 텍스트가 함께 저장됨 [{filename, text}].
    -- 인덱싱 안 함(전역 검색과 격리), 질의 시 대화 내 첨부를 모아 컨텍스트에 주입.
    -- 고객 개인 문서 — 대화는 소프트 삭제(#10)라 메시지·첨부 이력이 감사 목적으로 보존됨.
    attachments     JSONB,
    status          TEXT         NOT NULL DEFAULT 'done',   -- assistant 생성 상태: generating|done|failed|blocked (user 메시지는 항상 'done')
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT now(),
    -- 운영 지표 씨앗 (2026-07-18)
    user_id         TEXT,        -- user 메시지: 질문한 상담원 (X-User-Id)
    latency_ms      INTEGER,     -- assistant 메시지: 생성 소요(ms)
    cache_kind      TEXT,        -- assistant 메시지: 'semantic'=캐시 재생, NULL=신규 생성
    cited_docs      JSONB,       -- assistant: 실인용 파일명 배열 (저장 시 확정 — 지표 집계는 이 컬럼만)
    is_refusal      BOOLEAN,     -- assistant(status=done만): 거절 답변 여부 (저장 시 확정 — 문구 변경에도 과거 통계 불변)
    question_message_id BIGINT REFERENCES messages(id) ON DELETE SET NULL,  -- assistant: 답한 user 메시지 (미답변 짝짓기)
    intent          TEXT         -- assistant: 라우팅 결과 'KNOWLEDGE'|'OTHER' (2026-08-07 — 답변률 분모를 지식 질문으로 한정). NULL=차단 턴·컬럼 도입 전 행
);
-- 기존 DB 반영: ALTER TABLE messages ADD COLUMN IF NOT EXISTS intent TEXT;  (추가형 — 재구축 불필요)
CREATE INDEX IF NOT EXISTS idx_msg_conv_created
    ON messages (conversation_id, created_at);

-- ---------- 사용량 제한 / 쿼터 ----------
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
