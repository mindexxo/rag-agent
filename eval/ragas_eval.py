"""RAGAS 채점 러너.

judge 선택 (RAGAS_JUDGE 환경변수):
- vllm   (기본): 사내 vLLM(worker15의 Qwen3-14B, OpenAI 호환) — 비용 0·rate limit 없음·데이터 사내 잔류.
  생성 모델과 동일 모델이라 self-judge 편향 있음 → **상대 비교(A/B·회귀 감시) 전용**.
  절대값은 외부 judge 기준선과 비교 불가 (judge가 다르면 스케일이 다름).
- openai: 외부 강모델 — 절대값 리포트·스팟 체크용 (기존 기준선과 동일 스케일).

embeddings = 로컬 BGE-M3 재활용 (answer_relevancy용, 외부 콜 절약)

metric — 한 번 실행에 두 집합을 채점한다(#216):
  [생성축 전체] reference 불필요 3축
- faithfulness       : 답변 주장이 retrieved_contexts에 근거하는가 (환각)
- answer_relevancy   : 답변이 질문에 맞는가 (동문서답)
- context_precision  : top5 각 청크가 답변에 유용했는가 — Hit@1이 못 보는 '비-gold 청크 노이즈' 축
  [모범답안(expected_answer) 보유 문항만] 정답 대조 2축 — RAGAS_REF=0이면 생략
- context_recall     : 모범답안의 내용을 검색 청크가 담고 있었나 (검색의 근거 누락)
- answer_correctness : 답이 모범답안과 사실이 일치하나 (정답률을 직접)
  예전엔 RAGAS_REF=1이 채점 대상 자체를 보유 문항으로 줄여 고난도 107문항이 3축에서도 빠졌고, 전체를 보려면
  두 번 돌려 551문항 3축을 중복 채점했다. 모범답안은 기본 5유형에만 있다(고난도·trap 등은 규약상 없음).

- claude: Claude(claude -p 헤드리스) 교차채점 (#103) — 자기채점 다변화, API 키·과금 없음.
  콜=서브프로세스라 느림(전체 450 ≈ 19~38시간) → 스모크 규모로만. judge 다르면 절대값 비교 불가.
  주의: 스모크 표본이 앞쪽 쉬운 문항(단일 사실)만이면 어느 judge든 만점이라 무의미하다 —
  judge 견해차를 보려면 약점 축(paraphrase·multi_doc)이 표본에 들도록 골라야 한다(리뷰 발견).

실행: python -m eval.ragas_eval                        # 사내 vLLM judge, SMOKE=3
      SMOKE=0 python -m eval.ragas_eval                # 사내 vLLM judge, 전체
      RAGAS_JUDGE=openai python -m eval.ragas_eval     # 외부 judge (rate limit 주의)
      RAGAS_JUDGE=claude SMOKE=10 python -m eval.ragas_eval   # Claude 교차채점 스모크 (#103)
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
from eval.ragas_adapter import build_samples
from ragas import EvaluationDataset

JUDGE = os.getenv("RAGAS_JUDGE", "vllm")   # vllm=사내 vLLM(worker15) | openai=외부 (docstring 참조)
OPENAI_JUDGE_MODEL = "gpt-5-mini"
SMOKE = int(os.getenv("SMOKE", "3"))       # 0이면 전체
WITH_REF = os.getenv("RAGAS_REF", "1") == "1"   # 기본 on: 정답 대조 2축을 보유 문항에 추가 채점. 0이면 생략(#216 의미 변경)


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

    rows = build_samples("retrieved")
    n_smoke = SMOKE if smoke is None else smoke
    if n_smoke:
        rows = rows[:n_smoke]
    ref_rows = [r for r in rows if r["has_ref"]] if WITH_REF else []
    ds = EvaluationDataset.from_list([r["sample"] for r in rows])

    max_workers = 32
    if JUDGE == "vllm":
        from config import settings
        judge = LangchainLLMWrapper(ChatOpenAI(
            model=settings.vllm_model,
            base_url=settings.vllm_base_url,     # 사내 vLLM — OpenAI 호환 API
            api_key="EMPTY",
            temperature=0.2,                     # 채점 일관성 (운영 생성과 동일값, greedy 금지 — Qwen3 모델 카드)
            timeout=300,
            # thinking off + 생성 상한 (#216). 이 판정기는 rag/llm.LlmClient를 안 거쳐 운영의 enable_thinking=False·
            # max_tokens가 빠져 있었다 → Qwen3가 추론을 길게 생성(서버 reasoning-parser가 떼어내 텍스트엔 안 보임).
            # 실측: 같은 판정형 프롬프트 9.5s·354토큰 → 1.6s·54토큰. 길어진 추론이 RAGAS 작업 타임아웃(180s)을 넘겨
            # 조용히 NaN이 됐다(28행 스모크 정답대조 8/26, #114 14B 채점본 결측 수백 건). ⚠ 이 전후 RAGAS 수치는 비교 불가.
            max_tokens=settings.generation_reserve_tokens,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        ))
    elif JUDGE == "claude":
        # Claude 교차채점 (#103) — claude -p 헤드리스를 langchain 어댑터로 감싼다.
        # 콜=서브프로세스라 vLLM 연속배칭이 없다 → max_workers를 4로 낮춘다(longcontext_claude
        # 선례와 동일 근거). 전체 450은 19~38시간 추정이라 스모크 규모(SMOKE=10 내외)로만 쓴다.
        from eval.claude_client import ClaudeCliClient
        from eval.claude_langchain import ClaudeCliChatModel
        judge = LangchainLLMWrapper(
            ClaudeCliChatModel(client=ClaudeCliClient(model=os.getenv("CLAUDE_MODEL", "sonnet"))))
        max_workers = 4
    else:
        judge = LangchainLLMWrapper(
            ChatOpenAI(model=OPENAI_JUDGE_MODEL),
            bypass_temperature=True,   # GPT-5 계열은 temperature=1만 허용 → RAGAS 강제주입 차단
        )
    emb = LangchainEmbeddingsWrapper(LocalBGEEmbeddings())

    # max_workers 6→32 (8/29): 6은 클라이언트 병목이었다 — 6동시로 돌 때 vLLM 대기 큐 0·
    # KV 캐시 18%로 서버가 놀았다(n450 3축 2시간 6분). claude judge는 4로 낮춤(#103, 위).
    # timeout 180(기본)→600 (#216): 14B judge는 문항당 판정이 길어 180초에 걸려 조용히 NaN이 됐다 — 28행 스모크에서
    # 정답 대조 8/26 결측, #114 때 540건 채점본 결측 수백 건(별도 backfill로 메움). NaN은 평균에서 빠져 분모가 몰래 준다.
    run_config = RunConfig(max_workers=max_workers, max_retries=10, timeout=600)
    result = evaluate(dataset=ds, metrics=[faithfulness, answer_relevancy, LLMContextPrecisionWithoutReference()],
                      llm=judge, embeddings=emb, run_config=run_config)
    df = result.to_pandas()
    df.insert(0, "id", [r["id"] for r in rows])          # 문항 id — 두 집합을 합치고 사람이 추적할 수 있게
    ref_result = None
    if ref_rows:
        ref_result = evaluate(dataset=EvaluationDataset.from_list([r["sample"] for r in ref_rows]),
                              metrics=[LLMContextRecall(), answer_correctness],
                              llm=judge, embeddings=emb, run_config=run_config)
        rdf = ref_result.to_pandas()[["context_recall", "answer_correctness"]]
        rdf.insert(0, "id", [r["id"] for r in ref_rows])
        df = df.merge(rdf, on="id", how="left")          # 모범답안 없는 행은 빈값 — 평균에서 자동 제외

    # 퇴근 후에도 남게 파일로 저장 (per-sample + 집계).
    # 파일명에 실행 시각·샘플 수 — 고정 이름 덮어쓰기로 본측정 결과를 날린 사고(07-18) 재발 방지
    from datetime import datetime
    from pathlib import Path
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
        "n_ref": len(ref_rows),                          # 정답 대조 2축의 분모(#216)
        # 지표별 결측(NaN) 수 — 평균은 결측을 빼고 계산되므로 0이 아니면 분모가 몰래 줄어든 것이다
        "missing": {c: int(df[c].isna().sum()) - (len(rows) - len(ref_rows) if c in ("context_recall", "answer_correctness") else 0)
                    for c in ("faithfulness", "answer_relevancy", "llm_context_precision_without_reference",
                              "context_recall", "answer_correctness") if c in df.columns},
        "csv": str(out),
        "result": result,
        "ref_result": ref_result,
    }


def main():
    print(f"judge={JUDGE}  |  SMOKE={SMOKE or '0(전체)'}")
    r = compute()
    print(r["result"])
    if any(r["missing"].values()):
        print("⚠ 결측(판정 실패·타임아웃):", {k: v for k, v in r["missing"].items() if v})
    if r["ref_result"] is not None:
        print(f"[정답 대조 n_ref={r['n_ref']}]", r["ref_result"])
    print(f"→ saved {r['csv']}")


if __name__ == "__main__":
    main()
