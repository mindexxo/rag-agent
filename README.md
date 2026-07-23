# 상담도우미 (KMS) 백엔드

콜센터 상담원을 돕는 멀티테넌트 RAG 백엔드. 상담원이 통화 중 질문하면 등록된 문서를
근거로 답변한다. FastAPI + PostgreSQL(pgvector) + Redis, LLM/임베딩은 사내 GPU 서버(vLLM·TEI) 호출.

- 도메인 로직: `rag/` · API 라우팅: `routers/` · 스키마: `schemas/`

---

## 실행 방법

DB·Redis·GPU 모두 **ICCS 개발계**를 바라본다. 설정은 `.env.dev`에 채워져 있으며,
**사내망 VPN 연결 상태에서만** 접근된다.

### 0. 사전 준비

- Python 3.14
- 사내망 VPN 연결

#### Python 3.14 설치

이미 있으면 건너뛴다. 먼저 확인:

```bash
python3 --version   # Python 3.14.x 이면 OK  (Windows는  python --version)
```

**macOS** — Homebrew:
```bash
brew install python@3.14
# 설치 후 python3.14 명령이 생긴다
```
Homebrew에 `python@3.14` 포뮬러가 없으면 pyenv로:
```bash
brew install pyenv
pyenv install 3.14
pyenv local 3.14        # 이 프로젝트 폴더 안에서 실행 → 폴더 기본 파이썬이 3.14가 됨
```

**Windows** — 둘 중 하나:
- [python.org](https://www.python.org/downloads/)에서 3.14 설치 프로그램 실행 (설치 화면에서 **"Add python.exe to PATH" 체크**)
- 또는 winget: `winget install Python.Python.3.14`

#### 가상환경 + 의존성

**macOS / Linux**
```bash
python3.14 -m venv .venv   # 3.14가 기본 python3면 python3 -m venv 도 됨
source .venv/bin/activate
pip install -r requirements.txt
```

**Windows**
```bat
py -3.14 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 1. DB 비밀번호 입력

`.env.dev`에 개발계 접속 정보가 들어 있고, **DB 비밀번호만 비어 있다.**
`.env.dev`의 `DATABASE_URL`에서 `<비밀번호_담당자문의>` 부분을 실제 비밀번호로 바꾼다
(비밀번호는 담당자에게 문의).

```
DATABASE_URL=postgresql+asyncpg://cc_app_user:<비밀번호_담당자문의>@pg-nsvul.vpc-cdb-kr.ntruss.com:15432/cc_postgre
```

> 앱은 `.env.dev`를 자동으로 읽는다 (별도 복사 불필요).

### 2. 앱 실행

```bash
uvicorn main:app --reload --port 8000
```

기동 확인:

```bash
curl http://localhost:8000/kms/conversations -H "X-Tenant-Id: demo"
# 200 + [] 이면 정상
```

### 3. 워커 실행 (문서 업로드 시 필요)

문서 업로드의 청킹·임베딩·인덱싱은 백그라운드 워커(arq)가 처리한다.
채팅·FAQ·통계만 쓰면 없어도 되지만, **문서 업로드를 쓰면 반드시 함께 띄운다.**

```bash
# 앱과 다른 터미널에서
arq rag.worker.WorkerSettings
```
