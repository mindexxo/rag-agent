# triton-ocr — docling 원격 OCR 서버 (#168)

PDF 인제스션의 표 이미지(TableItem) 셀을 읽는 한국어 RapidOCR을 **워커 밖** Triton에 올린다.
워커(`rag/chunking.py`)는 docling `KserveV2OcrOptions`로 이 서버를 부른다. 왜 밖으로 뺐는지·실측치는
이슈 #168과 `config.py`의 `docling_do_ocr`·`ocr_kserve_*` 주석에 있다.

## 구성

| 파일 | 역할 |
|---|---|
| `model_repository/ocr/config.pbtxt` | docling 클라이언트 계약: 입력 `lang_type`(1,1 BYTES)·`image`(1,H,W,3 UINT8) → 출력 `boxes`(N,4,2)·`txts`(N)·`scores`(N). 출력에 배치 차원이 없어야 해서 `max_batch_size: 0` |
| `model_repository/ocr/1/model.py` | Python backend. docling 인프로세스 `RapidOcrModel`과 **같은 파라미터**로 RapidOCR 엔진을 조립한다(det ch-v5-mobile / rec korean-v5-mobile / cls ch-v4-mobile, `text_score` 0.5, 호출 시 `use_cls=False`). `use_cuda=True`인데 CUDA EP가 안 올라오면 기동을 실패시킨다 |
| `warmup.py` | 이미지 빌드 시 모델 파일을 미리 받는다 — 기동 시 외부망 불필요 |
| `Dockerfile` | `BASE`(Triton 릴리스 = CUDA 버전)와 `ORT`(onnxruntime 빌드)를 **짝으로** 고른다 |

## 빌드·기동 (worker15, GPU #3 — TEI·VLM과 같은 자리)

```bash
# 호스트 드라이버 535(CUDA 12.2) → Triton 25.02(CUDA 12.8, forward-compat) + onnxruntime-gpu 1.22.0(CUDA 12)
docker build --build-arg BASE=nvcr.io/nvidia/tritonserver:25.02-py3 -t kms-triton-ocr:25.02 .
docker rm -f kms-triton-ocr
docker run -d --name kms-triton-ocr --gpus device=3 -p 18893:8000 kms-triton-ocr:25.02
```

확인 — 셋 다 봐야 한다. **ready만 보고 끝내지 마라**: 첫 배포 때 CUDA EP 로드가 실패해도 rapidocr가
CPU로 조용히 내려가 ready가 떴다(2026-09-17). 지금은 model.py가 그 경우 기동을 실패시키지만 로그로 재확인한다.

```bash
curl -sf localhost:18893/v2/health/ready && echo ready
docker logs kms-triton-ocr 2>&1 | grep -E "\[ocr\]"        # 'CUDA EP OK: [CUDAExecutionProvider, ...]' 와 'engines=['korean']'
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv | grep tritonserver
```

## 로컬 검증 (맥 arm64 네이티브, CPU)

```bash
docker build --platform linux/arm64 --build-arg BASE=nvcr.io/nvidia/tritonserver:25.08-py3 --build-arg ORT=onnxruntime -t triton-ocr:local-cpu .
docker run -d --name triton-ocr-local -p 18893:8000 -e TRITON_OCR_USE_CUDA=false triton-ocr:local-cpu
OCR_KSERVE_URL=http://localhost:18893 pytest tests/test_chunking_docling.py -k RemoteOcr
```

## 바꿀 때 지켜야 하는 것

- **클라이언트 상수와 짝이다**: `rag/chunking.py`의 `_OCR_LANG`·`_OCR_SCALE`, 여기 `parameters.langs`·`text_score`.
  한쪽만 바꾸면 결과가 조용히 달라진다. 동일성은 같은 PDF를 인프로세스/원격으로 변환해 markdown을 대조해 확인한다
  (2026-09-17: CPU Triton 바이트 동일. GPU는 표 이미지 바이트 동일, 26쪽 실문서는 글자 동일·줄 묶음 6곳 차이 — GPU conv
  수치 차이로 OCR 박스 좌표가 미세하게 달라 docling 줄 그룹핑이 바뀐다. `use_tf32=0`·`cudnn_conv_algo_search=DEFAULT`가
  없으면 유령 텍스트까지 생긴다).
- `ORT` 핀은 `BASE`의 CUDA 메이저와 맞춘다. PyPI `onnxruntime-gpu` 1.23+는 CUDA 13 빌드다.
- 워커 `Dockerfile`은 rapidocr·onnxruntime을 **일부러 안 들인다** — 엔진은 여기뿐이다.
