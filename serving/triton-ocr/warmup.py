"""이미지 빌드 시 모델 파일을 미리 받아둔다 — 기동 시 외부망이 필요 없게. model.py와 같은 params."""
import importlib.util, sys
spec = importlib.util.spec_from_file_location("ocr_model", "/models/ocr/1/model.py")
# triton_python_backend_utils는 빌드 컨텍스트에 없다 — 파라미터 함수만 필요하므로 스텁을 꽂는다.
import types; sys.modules["triton_python_backend_utils"] = types.ModuleType("triton_python_backend_utils")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
from rapidocr import RapidOCR
import numpy as np
for lang in sys.argv[1].split(","):
    eng = RapidOCR(params=m._rapidocr_params(lang, use_cuda=False, text_score=0.5))
    eng(np.full((64, 256, 3), 255, dtype=np.uint8), use_cls=False)   # 세 모델 다 한 번씩 로드·실행
    print("warmup ok:", lang)
