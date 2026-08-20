-- schema.sql — Phase 1 테이블 + 인덱스
-- 적용: psql postgres -f schema.sql
-- idempotent: 여러 번 돌려도 안전.
--
-- RLS는 사용하지 않는다 — 테넌트 격리는 애플리케이션 쿼리의 WHERE 절이 유일한 방어선이고,
-- 누락 검출은 통합 테스트가 담당한다. 규약 전문은 rag/models.py 모듈 docstring 참조.

CREATE EXTENSION IF NOT EXISTS vector;
-- 대화 히스토리 검색(#28)의 ILIKE 부분일치 인덱스용. 운영 DB가 이미 쓰는 확장이라 맞췄다.
-- 3-gram이라 검색어가 3글자 미만이면 인덱스를 못 탄다 — '배송'·'환불' 같은 2글자 상담
-- 키워드는 순차 스캔이 된다(수용한 한계, 이슈 #28 참조). 2글자를 인덱스로 태우려면
-- pg_bigm이 필요한데 운영에 없어서, 환경별 성능 차이를 만들지 않으려고 pg_trgm으로 통일.
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
-- filename은 애플리케이션 경계에서 NFC로 정규화된 값만 저장된다 (#34, text_norm.py).
-- 위 UNIQUE는 코드포인트 단위라, 정규형이 섞이면 시각적으로 같은 이름이 별개 문서로 통과한다.
--
-- 기존 DB 반영(#34) — NFD로 저장된 과거 행 정리. **UPDATE 전에 충돌을 먼저 확인할 것**:
--   ① 정규화 대상 확인
--      SELECT id, tenant_id, filename, version FROM documents
--       WHERE filename <> normalize(filename, NFC);
--   ② 정규화하면 제약이 깨지는 쌍이 있는지 (있으면 UPDATE가 실패한다 — 수동 정리 선행).
--      **두 제약을 각각 봐야 한다** — 아래 uq_docs_one_active_per_name는 부분 유니크라
--      version이 달라도 걸린다. NFD판·NFC판이 둘 다 active로 공존하는 게 이 버그의 전형이다.
--      -- ②-a UNIQUE (tenant_id, filename, version)
--      SELECT tenant_id, normalize(filename, NFC) AS nfc_name, version, count(*)
--        FROM documents GROUP BY 1, 2, 3 HAVING count(*) > 1;
--      -- ②-b uq_docs_one_active_per_name (tenant_id, filename) WHERE is_active
--      SELECT tenant_id, normalize(filename, NFC) AS nfc_name, count(*)
--        FROM documents WHERE is_active GROUP BY 1, 2 HAVING count(*) > 1;
--   ③ 적용
--      UPDATE documents SET filename = normalize(filename, NFC)
--       WHERE filename <> normalize(filename, NFC);
-- 청크 임베딩에도 파일명이 prefix로 들어가므로(rag/index_text.py) 정합을 완전히 맞추려면
-- 해당 문서 재인제스트가 이상적이다. 검색 품질 영향은 작아 각주·지표 복구는 UPDATE만으로 충분.

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
    UNIQUE (document_id, chunk_index),
    CHECK ((document_id IS NOT NULL AND faq_id IS NULL) OR (document_id IS NULL AND faq_id IS NOT NULL))
);
CREATE UNIQUE INDEX IF NOT EXISTS uq_chunks_faq ON chunks (faq_id) WHERE faq_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_chunks_tenant_doc
    ON chunks (tenant_id, document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_dense_hnsw
    ON chunks USING hnsw (dense vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- ---------- LLM 응답 캐시 ----------
-- 기존 DB 반영(#56): 인용 방식 전환(인라인 라벨 → 출처 꼬리)으로 옛 캐시 행(라벨 박힌
-- answer·후보 전체 sources)은 새 계약과 혼재 불가 — 배포 시 1회 실행:
--   TRUNCATE answer_cache;   -- 재생성 가능한 데이터라 안전
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
-- 소유 조회 전용 (#28). 목록·검색·count가 전부 tenant+created_by+미삭제로 거르는데 위 인덱스엔
-- tenant뿐이라, 사용자 대화 100건을 찾으려고 테넌트 대화 2만 건을 훑고 버렸다(실측). 검색은
-- 그 2만 건마다 ILIKE·EXISTS를 평가해 1초가 걸렸고, 이 인덱스로 후보가 100건이 되며 사라진다:
--   목록 2.9→1.7ms / count 6.5ms(Seq Scan)→3.5ms / 검색 1125→9.5ms.
-- deleted_at 부분 인덱스인 이유: 조회 경로는 전부 미삭제만 보므로 삭제분을 넣을 이유가 없다.
CREATE INDEX IF NOT EXISTS idx_conv_owner_last
    ON conversations (tenant_id, created_by, last_used_at DESC)
    WHERE deleted_at IS NULL;
-- 제목 부분일치 검색 (#28). ILIKE '%…%'는 선행 와일드카드라 btree를 못 타므로 트라이그램 GIN.
CREATE INDEX IF NOT EXISTS idx_conv_title_trgm
    ON conversations USING gin (title gin_trgm_ops);

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
    status          TEXT         NOT NULL DEFAULT 'done',   -- assistant 생성 상태: generating|done|failed|blocked|cancelled (user 메시지는 항상 'done')
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT now(),
    -- 운영 지표 씨앗 (2026-07-18)
    user_id         TEXT,        -- user 메시지: 질문한 상담원 (X-User-Id)
    latency_ms      INTEGER,     -- assistant 메시지: 생성 소요(ms)
    cache_kind      TEXT,        -- assistant 메시지: 'semantic'=캐시 재생, NULL=신규 생성
    cited_docs      JSONB,       -- assistant: 실인용 파일명 배열 (저장 시 확정 — 지표 집계는 이 컬럼만).
                                 -- **근거없음(ungrounded) = 이 배열이 비어 있음** — 별도 컬럼 없다 (#61).
                                 -- 옛 is_refusal(거절 문구 부분일치)을 대체한 판정이다. NULL도 없음으로 본다
                                 -- (routers/conversations.py의 `m.cited_docs or []` 관례와 같은 결론).
                                 -- 주의: #56 이전 행은 인라인 인용 시절이라 이 컬럼이 비어 있어도
                                 -- 실제로는 근거를 댄 답변이다 — 그 구간 지표는 무의미하다.
    question_message_id BIGINT REFERENCES messages(id) ON DELETE SET NULL,  -- assistant: 답한 user 메시지 (미답변 짝짓기)
    intent          TEXT,        -- assistant: 라우팅 결과 'KNOWLEDGE'|'OTHER' (2026-08-07 — 답변률 분모를 지식 질문으로 한정). NULL=차단 턴·컬럼 도입 전 행
    block_reason    TEXT,        -- assistant(status='blocked'): 입력 가드(classify_and_guard) 차단 사유. 출력 가드는 제거됨(#26)
    -- 인간 피드백 1단계 (#8): 상담원의 답변 평가 — cited_docs와 조인해 "👎가 몰리는 문서" 집계
    feedback        BOOLEAN,     -- assistant: TRUE=👍 FALSE=👎 NULL=미선택/취소
    feedback_tag    TEXT,        -- assistant: 👎 사유 슬러그(wrong_info|wrong_source|outdated_doc|insufficient). 검증은 앱(Pydantic Literal)
    feedback_text   TEXT         -- assistant: 👎 자유 서술 (옵셔널, 앱에서 500자 제한) — 태그가 못 잡는 사유 발굴용
);
-- 기존 DB 반영: ALTER TABLE messages ADD COLUMN IF NOT EXISTS intent TEXT;  (추가형 — 재구축 불필요)
-- 기존 DB 반영(#22): ALTER TABLE messages ADD COLUMN IF NOT EXISTS block_reason TEXT;
-- 기존 DB 반영(#8): ALTER TABLE messages ADD COLUMN IF NOT EXISTS feedback BOOLEAN;
--                  ALTER TABLE messages ADD COLUMN IF NOT EXISTS feedback_tag TEXT;
--                  ALTER TABLE messages ADD COLUMN IF NOT EXISTS feedback_text TEXT;
-- 기존 DB 반영(#61): 거절 문구 판정 폐기 — cited_docs 빈 배열이 그 자리를 대신한다.
--                  ALTER TABLE messages DROP COLUMN IF EXISTS is_refusal;
--   왜 재구성이 아니라 DROP인가: 이 컬럼은 문구 매칭 결과의 스냅샷이라 재계산이 불가능했고
--   (그게 "문구 변경에도 과거 통계 불변"이라는 원래 설명의 뜻이다), 대체 신호(cited_docs)는
--   이미 같은 행에 같은 시점으로 저장돼 있다. 과거 값이 사라지는 것은 인지하고 택한 손실 —
--   운영 배포 전이라(NCP↔사내GPU 사설연동 미비) 실사용 추이가 아직 없어 비용이 0인 시점이다.
CREATE INDEX IF NOT EXISTS idx_msg_conv_created
    ON messages (conversation_id, created_at);
-- 고착 generating 회수용 (#46) — 워커 cron이 5분마다 전역 스윕하는데, status 인덱스가 없으면
-- messages 전체를 매번 순차 스캔한다. generating 행만 담는 부분 인덱스라 크기가 사실상 0.
CREATE INDEX IF NOT EXISTS idx_msg_generating
    ON messages (created_at) WHERE status = 'generating';
-- 대화 내용 부분일치 검색 (#28). attachments는 대상 아님 — 고객 개인 문서 본문이라 검색 제외.
CREATE INDEX IF NOT EXISTS idx_msg_content_trgm
    ON messages USING gin (content gin_trgm_ops);
-- 기존 DB 반영(#46): CREATE INDEX IF NOT EXISTS idx_msg_generating ON messages (created_at) WHERE status = 'generating';
-- 기존 DB 반영(#28): CREATE EXTENSION IF NOT EXISTS pg_trgm;
--                   CREATE INDEX IF NOT EXISTS idx_conv_owner_last ON conversations (tenant_id, created_by, last_used_at DESC) WHERE deleted_at IS NULL;
--                   CREATE INDEX IF NOT EXISTS idx_conv_title_trgm  ON conversations USING gin (title gin_trgm_ops);
--                   CREATE INDEX IF NOT EXISTS idx_msg_content_trgm ON messages      USING gin (content gin_trgm_ops);

-- ---------- 동시 요청 상한 (테넌트별 오버라이드) ----------
-- 행이 없으면 config 기본값이 적용된다 — 이 테이블은 특정 테넌트만 다르게 줄 때 쓴다.
-- 실시간 in-flight 카운터는 Redis ZSET(rag/limiter.py). 여기엔 정책값만 있다.
-- 기본값은 config(CONCURRENCY_LIMIT_DEFAULT·USER_CONCURRENCY_DEFAULT)와 같은 값으로 유지할 것.
CREATE TABLE IF NOT EXISTS tenant_quotas (
    tenant_id           TEXT        PRIMARY KEY,
    concurrency_limit   INTEGER     NOT NULL DEFAULT 10,   -- 동시 in-flight (테넌트 합산)
    user_concurrency    INTEGER     NOT NULL DEFAULT 10,   -- 동시 in-flight (사용자별)
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);
-- 기존 DB 반영(#24): 미사용 컬럼 제거 + 기본값 정합. 정의만 있고 코드가 읽지 않던 컬럼들 —
-- 값을 넣어도 아무 일이 일어나지 않는데 "동작한다"고 읽히는 상태였다.
-- ALTER TABLE tenant_quotas DROP COLUMN IF EXISTS rpm_limit;
-- ALTER TABLE tenant_quotas DROP COLUMN IF EXISTS user_rpm_limit;
-- ALTER TABLE tenant_quotas DROP COLUMN IF EXISTS daily_query_limit;
-- ALTER TABLE tenant_quotas DROP COLUMN IF EXISTS daily_upload_mb;
-- ALTER TABLE tenant_quotas DROP COLUMN IF EXISTS deep_mode_enabled;
-- ALTER TABLE tenant_quotas DROP COLUMN IF EXISTS deep_rpm_limit;
-- ALTER TABLE tenant_quotas DROP COLUMN IF EXISTS is_blocked;
-- ALTER TABLE tenant_quotas DROP COLUMN IF EXISTS block_reason;
-- ALTER TABLE tenant_quotas ALTER COLUMN concurrency_limit SET DEFAULT 10;   -- 8 → 10
-- ALTER TABLE tenant_quotas ALTER COLUMN user_concurrency  SET DEFAULT 10;   -- 3 → 10
