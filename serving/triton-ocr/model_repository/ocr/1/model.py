"""RapidOCR 파이프라인을 Triton Python backend로 감싼다 (#168).

docling이 인프로세스 `RapidOcrModel`(artifacts_path=None 분기)로 엔진을 조립하는 파라미터를 **그대로**
재현한다 — 모델(det: ch PP-OCRv5 mobile / rec: <lang> PP-OCRv5 mobile / cls: ch PP-OCRv4 mobile)과
text_score가 같아야 원격 결과가 인프로세스 결과와 동일하다. 각도 분류기(cls)는 로드는 하되 호출 시
use_cls=False로 끈다 — 켜면 한글 표 셀 값이 뒤집혔다('35,000원'→'공000'58', #168 실측).

계약(입출력 이름·형상)은 config.pbtxt 주석 참조. 좌표는 받은 이미지 픽셀 기준 그대로 돌려준다 —
scale 나누기·쪽 오프셋 보정은 docling 클라이언트가 한다.
"""
import json
import os

import numpy as np
import triton_python_backend_utils as pb_utils

_EMPTY_BOXES = np.zeros((0, 4, 2), dtype=np.float32)
_EMPTY_TXTS = np.array([], dtype=object)
_EMPTY_SCORES = np.zeros((0,), dtype=np.float32)


def _rapidocr_params(lang: str, *, use_cuda: bool, text_score: float) -> dict:
    """docling/models/stages/ocr/rapid_ocr_model.py의 params 딕셔너리와 같은 값."""
    from rapidocr import ModelType, OCRVersion

    mobile = ModelType.MOBILE
    params = {
        "Global.text_score": text_score,
        "Global.font_path": None,
        "EngineConfig.onnxruntime.use_cuda": use_cuda,
        "EngineConfig.onnxruntime.cuda_ep_cfg.device_id": 0,
        # docling은 lang→PP-OCR 버전을 푸는데, korean은 v6 다국어 목록에 없어 v5로 풀린다
        # (rapidocr.utils.model_resolver.PP_OCRV6_LANGS 확인). det/cls는 언어 무관하게 'ch'.
        "Det.ocr_version": OCRVersion.PPOCRV5, "Det.lang_type": "ch", "Det.model_type": mobile,
        "Cls.ocr_version": OCRVersion.PPOCRV4, "Cls.lang_type": "ch", "Cls.model_type": mobile,
        "Rec.ocr_version": OCRVersion.PPOCRV5, "Rec.lang_type": lang, "Rec.model_type": mobile,
        "Det.model_path": None, "Cls.model_path": None, "Rec.model_path": None, "Rec.rec_keys_path": None,
        "Det.use_cuda": use_cuda, "Cls.use_cuda": use_cuda, "Rec.use_cuda": use_cuda,
        "Det.use_dml": False, "Cls.use_dml": False, "Rec.use_dml": False,
    }
    if use_cuda:
        # CUDA EP를 CPU 결과에 맞춘다(2026-09-17 worker15 실측). rapidocr 기본 cuda_ep_cfg는
        # cudnn_conv_algo_search=EXHAUSTIVE이고 ORT CUDA EP는 A100에서 conv에 TF32를 기본으로 쓴다 —
        # 그 조합에서 같은 표 이미지가 CPU와 달리 유령 텍스트('상') 한 조각을 더 검출해 헤딩으로 나왔다.
        # 임계(text_score 0.5·det 박스 임계) 근처 값이 수치 오차로 뒤집힌 것이다. TF32를 끄고 알고리즘
        # 선택을 고정하면 CPU FP32와 수치가 같아지고 실행 간 결정성도 생긴다(EXHAUSTIVE는 실행마다 다른
        # 알고리즘을 고를 수 있다). 부수 효과: 크롭마다 입력 형상이 달라 EXHAUSTIVE가 요청마다 탐색을
        # 반복하던 비용도 사라진다. rapidocr는 이 dict를 그대로 ORT provider options로 넘긴다
        # (rapidocr/inference_engine/onnxruntime/provider_config.py).
        params["EngineConfig.onnxruntime.cuda_ep_cfg.cudnn_conv_algo_search"] = "DEFAULT"
        params["EngineConfig.onnxruntime.cuda_ep_cfg.use_tf32"] = 0
    return params


def _assert_cuda_ep() -> None:
    """use_cuda=True인데 CUDA EP가 실제로 안 올라오면 **기동을 실패**시킨다.

    onnxruntime은 CUDA 라이브러리 버전이 안 맞으면 경고만 내고 CPU EP로 조용히 내려가고, rapidocr도 그대로
    따라간다("inference part is automatically shifted to CPUExecutionProvider"). worker15 첫 배포가 그랬다:
    PyPI onnxruntime-gpu 1.24는 CUDA 13 빌드라 Triton 25.02(CUDA 12.8)에서 libcublasLt.so.13을 못 찾았고,
    GPU 장비에서 CPU로 돌면서 health는 ready였다(2026-09-17). 그 상태를 정상으로 두지 않는다 — 모델 로드가
    실패하면 Triton이 ready를 안 주므로 배포 단계에서 바로 드러난다.
    """
    import glob
    import os

    import onnxruntime as ort
    import rapidocr

    if "CUDAExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError(f"onnxruntime {ort.__version__}: CUDAExecutionProvider가 없다 — onnxruntime-gpu가 아니다")
    models = glob.glob(os.path.join(os.path.dirname(rapidocr.__file__), "models", "*det*.onnx"))
    if not models:
        raise RuntimeError("rapidocr 모델 파일이 없어 CUDA EP를 검증할 수 없다 — 이미지 빌드의 warmup이 안 돌았다")
    sess = ort.InferenceSession(models[0], providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    if "CUDAExecutionProvider" not in sess.get_providers():
        raise RuntimeError(f"onnxruntime {ort.__version__}: CUDA EP 로드 실패(CUDA/cuDNN 버전 불일치) — "
                           "CPU로 조용히 내려가는 대신 기동을 실패시킨다. Dockerfile의 ORT 핀과 BASE의 CUDA 버전을 맞춰라")
    pb_utils.Logger.log_info(f"[ocr] onnxruntime {ort.__version__} CUDA EP OK: {sess.get_providers()}")


def _as_str(value) -> str:
    return value.decode("utf-8") if isinstance(value, (bytes, np.bytes_)) else str(value)


class TritonPythonModel:
    def initialize(self, args):
        cfg = json.loads(args["model_config"])
        params = {k: v["string_value"] for k, v in cfg.get("parameters", {}).items()}
        env_cuda = os.environ.get("TRITON_OCR_USE_CUDA")
        use_cuda = (env_cuda if env_cuda is not None else params.get("use_cuda", "true")).strip().lower() == "true"
        text_score = float(params.get("text_score", "0.5"))
        langs = [s.strip() for s in params.get("langs", "korean").split(",") if s.strip()]

        if use_cuda:
            _assert_cuda_ep()
        from rapidocr import RapidOCR

        self._engines = {lang: RapidOCR(params=_rapidocr_params(lang, use_cuda=use_cuda, text_score=text_score))
                         for lang in langs}
        pb_utils.Logger.log_info(f"[ocr] engines={list(self._engines)} use_cuda={use_cuda} text_score={text_score}")

    def execute(self, requests):
        responses = []
        for request in requests:
            lang_arr = pb_utils.get_input_tensor_by_name(request, "lang_type").as_numpy()
            lang = _as_str(lang_arr.reshape(-1)[0])
            engine = self._engines.get(lang)
            if engine is None:
                responses.append(pb_utils.InferenceResponse(
                    error=pb_utils.TritonError(f"unsupported lang_type {lang!r}; configured={list(self._engines)}")))
                continue
            image = pb_utils.get_input_tensor_by_name(request, "image").as_numpy()   # (1,H,W,3) uint8 RGB
            img = np.ascontiguousarray(image[0])
            result = engine(img, use_det=None, use_cls=False, use_rec=None)

            if result is None or result.boxes is None or not result.txts:
                boxes, txts, scores = _EMPTY_BOXES, _EMPTY_TXTS, _EMPTY_SCORES
            else:
                boxes = np.asarray(result.boxes, dtype=np.float32).reshape(-1, 4, 2)
                txts = np.array([t.encode("utf-8") for t in result.txts], dtype=object)
                scores = np.asarray(result.scores, dtype=np.float32).reshape(-1)

            responses.append(pb_utils.InferenceResponse(output_tensors=[
                pb_utils.Tensor("boxes", boxes),
                pb_utils.Tensor("txts", txts),
                pb_utils.Tensor("scores", scores),
            ]))
        return responses
