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
# 한 레이어로 묶는다 — uninstall한 파일이 아래 레이어에 남으면 지운 의미가 없다.
#
# ① CPU 전용 torch: linux/x86_64의 PyPI 기본 휠은 CUDA 빌드라 그대로 두면 nvidia-* 3.2GB가
#    따라 들어오고(이미지 6.8GB) import torch만으로 515MB를 먹는다. 앱은 GPU 장비에 올리지
#    않으므로(docling_device=cpu) 전부 쓰이지 않는 무게다. CPU 휠은 233MB.
# ② headless opencv: docling==2.126.0 → docling-slim[standard] → rapidocr → opencv-python(GUI)
#    사슬로 **OCR을 껐는데도** OCR 엔진과 GUI opencv가 딸려온다. 그 opencv는 libGL/libxcb를
#    링크하는데 slim 베이스엔 없어서 import cv2가 실패하고 PDF 인제스션이 전부 깨진다
#    (2026-09-13 개발계 실측). 시스템 라이브러리를 까는 방법(+215MB)도 되지만, 쓰지도 않는
#    OCR 엔진을 빼고 headless로 바꾸는 쪽이 이미지도 작고 의존도 정직하다.
#    OCR은 켜져 있지만(#168) 엔진은 워커가 아니라 원격 Triton(serving/triton-ocr/)에 있다 — 워커는
#    KServe HTTP 클라이언트(requests)만 쓴다. 그래서 rapidocr·onnxruntime을 여기 들이지 않는다.
#    인프로세스로 되돌리면 피크 +1.2GB·시간 79%가 워커에 다시 얹힌다(이슈 #168 실측).
RUN pip install --upgrade pip \
    && pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt \
    && pip uninstall -y rapidocr opencv-python \
    && pip install --no-cache-dir opencv-python-headless

# 애플리케이션 코드 복사
COPY . .

# 비루트 사용자로 구동
RUN useradd -m appuser
USER appuser

EXPOSE 8000

# 기본 실행 = 웹 앱. 워커는 compose에서 command로 이 값을 덮어써서 띄운다.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
