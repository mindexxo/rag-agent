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
- context_precision  : top5 각 청크가 답변에 유용했는가 — Hit@1이 못 보는 '비-gold 청크 노이즈' 축 (기본 off, RAGAS_CTX_PRECISION=1)
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


from config import settings


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
        # 4 → 기본 8(#218): 경량 호출(입력 ~3.7k토큰)이라 토큰은 동시성과 무관. 제약은 로컬 메모리 —
        # claude -p 1개 최대 RSS 약 240MB, 16GB 기기에서 여유 약 3.4GB(서버·워커·IDE 상주) → 16은 스왑 위험.
        max_workers = int(os.getenv("CLAUDE_JUDGE_WORKERS", "8"))   # 20까지 실측 근거: 여유 6.3GB 기준 약 4.8GB
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

    # ── 묶음 채점 + 이어 돌리기 (#218) ──────────────────────────────────────────────
    # evaluate()는 전체가 끝나야 결과를 돌려준다 — 사용량 한도로 중단되면 완료분까지 사라졌다(10/10 A: 765/1974에서 중단).
    # RAGAS_BATCH문항씩 채점해 묶음마다 진행 파일에 덧붙이고, 재실행 시 지표가 다 찬 문항은 건너뛴다.
    # 결측(NaN)·미채점 칸만 다시 채점하므로 결측 메우기 스크립트가 따로 필요 없다.
    # 진행 파일 키 = 심판·모델·생성 결과 파일 해시 — 답변을 재생성하면 새 파일로 시작한다(옛 점수 재사용 방지).
    import hashlib
    import pandas as pd
    from datetime import datetime
    from pathlib import Path
    # context_precision은 기본 off(#218): 문항당 판정 5회(청크 5개 각각)로 호출의 절반 이상인데, 검색 노이즈 지표라
    # 검색축(Hit@1·R@5·MRR)과 겹치고 '생성기만 교체'하는 상용 비교에선 A·B가 거의 같게 나온다. run_all 이력에도 행이 없다.
    # 필요하면 RAGAS_CTX_PRECISION=1로 켠다.
    with_ctx_precision = os.getenv("RAGAS_CTX_PRECISION", "0") == "1"
    base_cols = ["faithfulness", "answer_relevancy"] + (["llm_context_precision_without_reference"] if with_ctx_precision else [])
    base_metrics = [faithfulness, answer_relevancy] + ([LLMContextPrecisionWithoutReference()] if with_ctx_precision else [])
    ref_cols = ["context_recall", "answer_correctness"]
    gen_hash = hashlib.sha1(Path("eval/results/generation_retrieved.jsonl").read_bytes()).hexdigest()[:10]
    model_tag = (os.getenv("CLAUDE_MODEL", "sonnet") if JUDGE == "claude" else
                 (settings.vllm_model if JUDGE == "vllm" else OPENAI_JUDGE_MODEL)).replace("/", "_")
    progress = Path(f"eval/results/ragas_progress_{JUDGE}_{model_tag}_{gen_hash}.csv")
    prog = pd.read_csv(progress).set_index("id") if progress.exists() else pd.DataFrame()

    def _missing(r: dict, cols: list[str]) -> bool:
        if r["id"] not in prog.index:
            return True
        return any(c not in prog.columns or pd.isna(prog.at[r["id"], c]) for c in cols)

    need_base = [r for r in rows if _missing(r, base_cols)]
    need_ref = [r for r in ref_rows if _missing(r, ref_cols)]
    print(f"진행 파일 {progress.name}: 3지표 남은 {len(need_base)}/{len(rows)} · 정답대조 남은 {len(need_ref)}/{len(ref_rows)}")
    batch = int(os.getenv("RAGAS_BATCH", "50"))

    def _run(todo: list[dict], metrics: list, cols: list[str], rename: dict | None = None) -> None:
        nonlocal prog
        for i in range(0, len(todo), batch):
            chunk = todo[i:i + batch]
            res = evaluate(dataset=EvaluationDataset.from_list([r["sample"] for r in chunk]), metrics=metrics,
                           llm=judge, embeddings=emb, run_config=run_config)
            part = res.to_pandas()
            part.insert(0, "id", [r["id"] for r in chunk])
            part = part.set_index("id")
            keep = [c for c in part.columns if c in cols or c in ("user_input", "response", "reference")]
            part = part[keep]
            for rid in part.index:                     # 칸 단위 갱신 — 다른 지표 칸은 보존
                for c in part.columns:
                    prog.loc[rid, c] = part.at[rid, c]
            prog.reset_index(names="id").to_csv(progress, index=False)   # 묶음마다 저장 — 여기까지는 중단돼도 남는다
            print(f"  [{'/'.join(cols)[:30]}] {min(i + batch, len(todo))}/{len(todo)} 저장")

    if need_base:
        _run(need_base, base_metrics, base_cols)
    if need_ref:
        _run(need_ref, [LLMContextRecall(), answer_correctness], ref_cols)

    df = prog.reindex([r["id"] for r in rows]).reset_index(names="id")
    for c in ref_cols:                                  # 모범답안 없는 행은 정답대조 빈값이 정상
        if c in df.columns:
            df.loc[~df["id"].isin({r["id"] for r in ref_rows}), c] = float("nan")
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    out = Path(f"eval/results/ragas_retrieved_{stamp}_n{len(rows)}.csv")   # 최종 스냅샷(시각·문항수) — 덮어쓰기 사고 방지(07-18)
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
        "n": len(rows),
        "n_ref": len(ref_rows),                          # 정답 대조 2축의 분모(#216)
        # 지표별 결측(NaN) 수 — 0이 아니면 같은 명령을 다시 돌리면 그 칸만 다시 채점한다(#218)
        "missing": {c: int(df[c].isna().sum()) - (len(rows) - len(ref_rows) if c in ref_cols else 0)
                    for c in base_cols + ref_cols if c in df.columns},
        "csv": str(out),
        "progress": str(progress),
    }


def main():
    print(f"judge={JUDGE}  |  SMOKE={SMOKE or '0(전체)'}")
    r = compute()
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k not in ("missing",)})
    if any(r["missing"].values()):
        print("⚠ 결측(판정 실패·타임아웃) — 같은 명령을 다시 돌리면 그 칸만 다시 채점:", {k: v for k, v in r["missing"].items() if v})
    print(f"→ saved {r['csv']}")


if __name__ == "__main__":
    main()
