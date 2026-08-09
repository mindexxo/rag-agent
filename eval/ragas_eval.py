"""RAGAS 채점 러너.

judge 선택 (RAGAS_JUDGE 환경변수):
- vllm   (기본): 사내 vLLM(worker15의 Qwen3-14B, OpenAI 호환) — 비용 0·rate limit 없음·데이터 사내 잔류.
  생성 모델과 동일 모델이라 self-judge 편향 있음 → **상대 비교(A/B·회귀 감시) 전용**.
  절대값은 외부 judge 기준선과 비교 불가 (judge가 다르면 스케일이 다름).
- openai: 외부 강모델 — 절대값 리포트·스팟 체크용 (기존 기준선과 동일 스케일).

embeddings = 로컬 BGE-M3 재활용 (answer_relevancy용, 외부 콜 절약)

metric (reference 불필요 3축 — context_recall류는 완성형 정답문이 gold에 없어 보류):
- faithfulness       : 답변 주장이 retrieved_contexts에 근거하는가 (환각)
- answer_relevancy   : 답변이 질문에 맞는가 (동문서답)
- context_precision  : top5 각 청크가 답변에 유용했는가 — Hit@1이 못 보는 '비-gold 청크 노이즈' 축

실행: python -m eval.ragas_eval                        # 사내 vLLM judge, SMOKE=3
      SMOKE=0 python -m eval.ragas_eval                # 사내 vLLM judge, 전체
      RAGAS_JUDGE=openai python -m eval.ragas_eval     # 외부 judge (rate limit 주의)
      (vllm judge의 실제 대상 = .env의 VLLM_BASE_URL — 현재 worker15:18888)
"""
import os

from dotenv import load_dotenv
load_dotenv()

from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, LLMContextPrecisionWithoutReference, LLMContextRecall, answer_correctness
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig

from rag.embeddings import embed_texts_sync, embed_query_sync
from eval.ragas_adapter import build_dataset

JUDGE = os.getenv("RAGAS_JUDGE", "vllm")   # vllm=사내 vLLM(worker15) | openai=외부 (docstring 참조)
OPENAI_JUDGE_MODEL = "gpt-5-mini"
SMOKE = int(os.getenv("SMOKE", "3"))       # 0이면 전체
WITH_REF = os.getenv("RAGAS_REF", "0") == "1"   # 1이면 검수 정답문 보유 문항만 + reference 축 추가


class LocalBGEEmbeddings(Embeddings):
    """BGE-M3 dense를 LangChain Embeddings 인터페이스로 래핑."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [e.dense for e in embed_texts_sync(texts)]

    def embed_query(self, text: str) -> list[float]:
        return embed_query_sync(text).dense


def compute(smoke: int | None = None) -> dict:
    """RAGAS 채점 실행 → 요약 반환 (per-sample CSV도 저장). 출력은 main이 담당.

    smoke: None이면 환경변수 SMOKE 사용. 반환: {'faithfulness', 'answer_relevancy', 'n', 'csv'}
    """
    import math

    ds = build_dataset("retrieved", ref_only=WITH_REF)
    n_smoke = SMOKE if smoke is None else smoke
    if n_smoke:
        ds = ds[:n_smoke]

    if JUDGE == "vllm":
        from config import settings
        judge = LangchainLLMWrapper(ChatOpenAI(
            model=settings.vllm_model,
            base_url=settings.vllm_base_url,     # 사내 vLLM — OpenAI 호환 API
            api_key="EMPTY",
            temperature=0.2,                     # 채점 일관성 (운영 생성과 동일값, greedy 금지 — Qwen3 모델 카드)
            timeout=300,
        ))
    else:
        judge = LangchainLLMWrapper(
            ChatOpenAI(model=OPENAI_JUDGE_MODEL),
            bypass_temperature=True,   # GPT-5 계열은 temperature=1만 허용 → RAGAS 강제주입 차단
        )
    emb = LangchainEmbeddingsWrapper(LocalBGEEmbeddings())

    result = evaluate(
        dataset=ds,
        metrics=([faithfulness, answer_relevancy, LLMContextPrecisionWithoutReference()]
                 + ([LLMContextRecall(), answer_correctness] if WITH_REF else [])),
        llm=judge,
        embeddings=emb,
        run_config=RunConfig(max_workers=6, max_retries=10),
    )

    # 퇴근 후에도 남게 파일로 저장 (per-sample + 집계).
    # 파일명에 실행 시각·샘플 수 — 고정 이름 덮어쓰기로 본측정 결과를 날린 사고(07-18) 재발 방지
    from datetime import datetime
    from pathlib import Path
    df = result.to_pandas()
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    out = Path(f"eval/results/ragas_retrieved_{stamp}_n{len(ds)}.csv")
    df.to_csv(out, index=False)

    def _mean(col):
        v = df[col].mean() if col in df.columns else float("nan")
        return None if (v is None or math.isnan(v)) else float(v)

    return {
        "faithfulness": _mean("faithfulness"),
        "answer_relevancy": _mean("answer_relevancy"),
        "context_precision": _mean("llm_context_precision_without_reference"),
        "context_recall": _mean("context_recall"),
        "answer_correctness": _mean("answer_correctness"),
        "n": len(ds),
        "csv": str(out),
        "result": result,
    }


def main():
    print(f"judge={JUDGE}  |  SMOKE={SMOKE or '0(전체)'}")
    r = compute()
    print(r["result"])
    print(f"→ saved {r['csv']}")


if __name__ == "__main__":
    main()
