# Triton + RapidOCR(한국어) — docling 원격 OCR 서버 (#168). worker15 GPU에 띄운다.
#  BASE: Triton 릴리스는 CUDA 버전을 못박는다 — 호스트 드라이버가 지원하는 CUDA 이하를 골라야 한다.
#  ORT : GPU면 onnxruntime-gpu — **버전을 BASE의 CUDA에 맞춰 핀한다.** PyPI onnxruntime-gpu 1.23+는 CUDA 13
#        빌드라 25.02(CUDA 12.8)에서 CUDA EP 로드가 실패하고 rapidocr가 조용히 CPU로 내려간다(2026-09-17 실측 —
#        model.py의 _assert_cuda_ep가 그 상태를 기동 실패로 바꾼다). 1.22.0이 CUDA 12 + cuDNN 9 계열의 마지막(PyPI에 1.22.1은 없다).
#        로컬 CPU 검증(맥 arm64 네이티브)이면 --build-arg ORT=onnxruntime.
ARG BASE=nvcr.io/nvidia/tritonserver:25.02-py3
FROM ${BASE}
ARG ORT="onnxruntime-gpu==1.22.0"
ARG LANGS=korean
ENV PIP_BREAK_SYSTEM_PACKAGES=1
# rapidocr는 GUI opencv를 요구하지만 서버에 디스플레이가 없다 — 워커 Dockerfile과 같은 이유로 headless로 바꾼다.
RUN pip install --no-cache-dir "rapidocr==3.9.2" ${ORT} \
 && pip uninstall -y opencv-python \
 && pip install --no-cache-dir opencv-python-headless
COPY model_repository /models
COPY warmup.py /tmp/warmup.py
RUN python3 /tmp/warmup.py ${LANGS} && rm /tmp/warmup.py
EXPOSE 8000 8001 8002
CMD ["tritonserver", "--model-repository=/models"]
