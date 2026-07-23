# 콜센터 상담도우미(RAG) 백엔드 이미지
# 웹 앱(uvicorn)과 워커(arq)가 이 동일 이미지를 공유하고 실행 명령만 다르게 띄운다.
FROM python:3.14-slim

# 파이썬 로그 즉시 출력 / .pyc 미생성
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# 의존성 먼저 복사·설치 (코드만 바뀔 때 이 레이어 캐시 재사용 → 재빌드 빠름)
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# 애플리케이션 코드 복사
COPY . .

# 비루트 사용자로 구동
RUN useradd -m appuser
USER appuser

EXPOSE 8000

# 기본 실행 = 웹 앱. 워커는 compose에서 command로 이 값을 덮어써서 띄운다.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
