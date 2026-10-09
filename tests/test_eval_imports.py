"""eval 축 모듈이 전부 import되는지 — retrieval_mt가 8/24~10/9 사이 깨진 채 아무도 몰랐다(#209). 축을 추가하면 여기 넣는다."""
import importlib

import pytest


@pytest.mark.parametrize("mod", ["eval.retrieval", "eval.retrieval_v2", "eval.retrieval_mt", "eval.generation",
                                 "eval.run_all", "eval.reindex_documents"])
def test_eval_module_imports(mod):
    importlib.import_module(mod)


def test_ragas_eval_imports_when_installed():
    # ragas·langchain은 requirements에서 선택 설치(주석) — 없는 환경에선 건너뛴다. 모듈이 import 시 load_dotenv()를
    # 부르므로 환경변수 부수효과가 있다(리뷰 지적) — 이 테스트가 마지막에 돌아도 다른 테스트는 이미 값을 읽은 뒤다.
    pytest.importorskip("ragas")
    importlib.import_module("eval.ragas_eval")
