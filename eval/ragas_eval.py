"""RAGAS 채점 러너.

judge 선택 (RAGAS_JUDGE 환경변수):
- settings (기본): rag/llm 공급자 설정(LLM_PROVIDER·LLM_MODEL)과 같은 모델 — 사내 vLLM이면 비용 0·데이터 사내 잔류.
  생성 모델과 동일 모델이라 self-judge 편향 있음 → **상대 비교(A/B·회귀 감시) 전용**.
  외부 강모델로 채점하려면 LLM_PROVIDER=openai 등으로 설정 자체를 바꾼다(별도 분기를 두지 않는다).
  절대값은 judge가 다르면 스케일이 달라 비교 불가.
- claude: Claude(claude -p 헤드리스) 교차채점 — 자기채점 다변화, API 키·과금 없음.
  콜=서브프로세스라 느림(전체 450 ≈ 19~38시간) → 스모크 규모로만. judge 다르면 절대값 비교 불가.
  주의: 스모크 표본이 앞쪽 쉬운 문항(단일 사실)만이면 어느 judge든 만점이라 무의미하다 —
  judge 견해차를 보려면 약점 축(paraphrase·multi_doc)이 표본에 들도록 골라야 한다.

embeddings = 로컬 BGE-M3 재활용 (answer_relevancy용, 외부 콜 절약)

metric — 한 번 실행에 두 집합을 채점한다:
  [생성축 전체] reference 불필요 3축
- faithfulness       : 답변 주장이 retrieved_contexts에 근거하는가 (환각)
- answer_relevancy   : 답변이 질문에 맞는가 (동문서답)
- context_precision  : top5 각 청크가 답변에 유용했는가 — Hit@1이 못 보는 '비-gold 청크 노이즈' 축 (기본 off, RAGAS_CTX_PRECISION=1)
  [모범답안(expected_answer) 보유 문항만] 정답 대조 2축 — RAGAS_REF=0이면 생략
- context_recall     : 모범답안의 내용을 검색 청크가 담고 있었나 (검색의 근거 누락)
- answer_correctness : 답이 모범답안과 사실이 일치하나 (정답률을 직접)
  정답 대조 2축만 보유 문항으로 줄이고 3축은 전체 문항을 채점한다. 모범답안은 기본 5유형에만 있다(고난도·trap 등은 규약상 없음).

실행: python -m eval.ragas_eval                        # 설정 공급자 judge(기본 사내 vLLM), SMOKE=3
      SMOKE=0 python -m eval.ragas_eval                # 전체
      LLM_PROVIDER=openai LLM_MODEL=gpt-5-mini LLM_TEMPERATURE=1.0 python -m eval.ragas_eval   # 외부 judge
      RAGAS_JUDGE=claude SMOKE=10 python -m eval.ragas_eval   # Claude 교차채점 스모크
      (settings judge의 실제 대상 = .env의 LLM_PROVIDER/LLM_BASE_URL — 기본 worker15:18888)
"""

import hashlib
import math
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from dotenv import load_dotenv

load_dotenv()

from langchain_core.embeddings import Embeddings  # noqa: E402 — .env를 먼저 읽어야 judge·임베딩 설정이 잡힌다
from ragas import EvaluationDataset, evaluate  # noqa: E402
from ragas.embeddings import LangchainEmbeddingsWrapper  # noqa: E402
from ragas.llms import LangchainLLMWrapper  # noqa: E402
from ragas.metrics import (  # noqa: E402
    LLMContextPrecisionWithoutReference,
    LLMContextRecall,
    answer_correctness,
    answer_relevancy,
    faithfulness,
)
from ragas.run_config import RunConfig  # noqa: E402

from config import settings  # noqa: E402
from eval.ragas_adapter import build_samples  # noqa: E402
from rag.search.embeddings import embed_query_sync, embed_texts_sync  # noqa: E402

if TYPE_CHECKING:
    import pandas as pd

JUDGE = os.getenv("RAGAS_JUDGE", "settings")  # settings=rag/llm 공급자 설정 | claude=CLI (docstring 참조)
SMOKE = int(os.getenv("SMOKE", "3"))  # 0이면 전체
WITH_REF = os.getenv("RAGAS_REF", "1") == "1"  # 기본 on: 정답 대조 2축을 보유 문항에 추가 채점. 0이면 생략


class LocalBGEEmbeddings(Embeddings):
    """BGE-M3 dense를 LangChain Embeddings 인터페이스로 래핑."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [e.dense for e in embed_texts_sync(texts)]

    def embed_query(self, text: str) -> list[float]:
        return embed_query_sync(text).dense


def _judge(kind: str) -> tuple[Any, int]:
    """(RAGAS judge 래퍼, max_workers). 분기는 여기뿐 — tests/test_eval_llm_adapter.py가 고정한다."""
    from eval.llm_langchain import LlmClientChatModel

    if kind == "settings":
        # thinking off·max_tokens·temperature는 공급자 구현(rag/llm)이 넣는다(요청 바디는 test_llm_providers가 고정).
        from rag.llm import create_llm

        # max_workers 32: 더 낮으면 클라이언트가 병목이라 vLLM이 논다. 상용 공급자면 벤더 rate limit이 상한이다.
        return LangchainLLMWrapper(LlmClientChatModel(client=create_llm()), bypass_temperature=True), 32
    if kind == "claude":
        # Claude 교차채점 — claude -p 헤드리스를 같은 어댑터로 감싼다. 콜=서브프로세스라 vLLM 연속배칭이 없다.
        # 전체 450은 19~38시간 추정이라 스모크 규모(SMOKE=10 내외)로만 쓴다.
        from eval.claude_client import ClaudeCliClient

        judge = LangchainLLMWrapper(
            LlmClientChatModel(client=ClaudeCliClient(model=os.getenv("CLAUDE_MODEL", "sonnet")))
        )
        # 기본 8: 경량 호출(입력 ~3.7k토큰)이라 토큰은 동시성과 무관. 제약은 로컬 메모리 —
        # claude -p 1개 최대 RSS 약 240MB, 16GB 기기에서 여유 약 3.4GB(서버·워커·IDE 상주) → 16은 스왑 위험.
        return judge, int(os.getenv("CLAUDE_JUDGE_WORKERS", "8"))  # 20이면 약 4.8GB — 여유 메모리가 그 이상일 때만
    raise SystemExit(f"RAGAS_JUDGE={kind} — settings|claude 만 지원")


def _progress_tag(kind: str) -> str:
    """진행 파일 키의 심판 부분 — 공급자·모델. settings judge가 vLLM이면 "vllm_<모델>"이라 기존 진행 파일과 이어진다."""
    judge, model = (
        ("claude", os.getenv("CLAUDE_MODEL", "sonnet"))
        if kind == "claude"
        else (settings.llm_provider, settings.llm_model)
    )
    return f"{judge}_{model.replace('/', '_')}"


def _base_metrics() -> tuple[list[str], list]:
    """(결과 칸 이름, 3축 지표) — reference가 필요 없는 지표 집합."""
    # context_precision은 기본 off: 문항당 판정 5회(청크 5개 각각)로 호출의 절반 이상인데, 검색 노이즈 지표라
    # 검색축(Hit@1·R@5·MRR)과 겹치고 '생성기만 교체'하는 상용 비교에선 A·B가 거의 같게 나온다. run_all 이력에도 행이 없다.
    # 필요하면 RAGAS_CTX_PRECISION=1로 켠다.
    with_ctx_precision = os.getenv("RAGAS_CTX_PRECISION", "0") == "1"
    base_cols = ["faithfulness", "answer_relevancy"] + (
        ["llm_context_precision_without_reference"] if with_ctx_precision else []
    )
    base_metrics = [faithfulness, answer_relevancy] + (
        [LLMContextPrecisionWithoutReference()] if with_ctx_precision else []
    )
    return base_cols, base_metrics


@dataclass
class _Progress:
    """진행 파일 — 문항(id)×지표 칸. 묶음마다 디스크에 덧붙여 중단돼도 완료분이 남는다."""

    path: Path
    frame: "pd.DataFrame"

    @classmethod
    def open(cls) -> "_Progress":
        # 진행 파일 키 = 심판·모델·생성 결과 파일 해시 — 답변을 재생성하면 새 파일로 시작한다(옛 점수 재사용 방지).
        import pandas as pd

        gen_hash = hashlib.sha1(Path("eval/results/generation_retrieved.jsonl").read_bytes()).hexdigest()[:10]
        path = Path(f"eval/results/ragas_progress_{_progress_tag(JUDGE)}_{gen_hash}.csv")
        return cls(path, pd.read_csv(path).set_index("id") if path.exists() else pd.DataFrame())

    def missing(self, r: dict, cols: list[str]) -> bool:
        import pandas as pd

        prog = self.frame
        if r["id"] not in prog.index:
            return True
        return any(c not in prog.columns or pd.isna(prog.at[r["id"], c]) for c in cols)

    def merge(self, part: "pd.DataFrame") -> None:
        for rid in part.index:  # 칸 단위 갱신 — 다른 지표 칸은 보존
            for c in part.columns:
                self.frame.loc[rid, c] = part.at[rid, c]
        self.frame.reset_index(names="id").to_csv(self.path, index=False)  # 묶음마다 저장 — 여기까지는 중단돼도 남는다


@dataclass
class _Scorer:
    """RAGAS evaluate() 호출 설정 — judge·임베딩·실행 설정·묶음 크기."""

    judge: Any
    emb: Any
    run_config: RunConfig
    batch: int

    def run(self, progress: _Progress, todo: list[dict], metrics: list, cols: list[str]) -> None:
        batch = self.batch
        for i in range(0, len(todo), batch):
            chunk = todo[i : i + batch]
            res = evaluate(
                dataset=EvaluationDataset.from_list([r["sample"] for r in chunk]),
                metrics=metrics,
                llm=self.judge,
                embeddings=self.emb,
                run_config=self.run_config,
            )
            part = res.to_pandas()  # type: ignore[union-attr]  # return_executor를 안 쓰면 EvaluationResult다
            part.insert(0, "id", [r["id"] for r in chunk])
            part = part.set_index("id")
            keep = [c for c in part.columns if c in cols or c in ("user_input", "response", "reference")]
            progress.merge(part[keep])
            print(f"  [{'/'.join(cols)[:30]}] {min(i + batch, len(todo))}/{len(todo)} 저장")


def _make_scorer() -> _Scorer:
    judge, max_workers = _judge(JUDGE)
    emb = LangchainEmbeddingsWrapper(LocalBGEEmbeddings())
    # timeout 600: 기본 180초면 14B judge의 긴 판정이 시간 초과로 조용히 NaN이 된다.
    # NaN은 평균에서 빠져 분모가 몰래 준다.
    run_config = RunConfig(max_workers=max_workers, max_retries=10, timeout=600)
    return _Scorer(judge, emb, run_config, 0)


def _snapshot(progress: _Progress, rows: list[dict], ref_rows: list[dict], ref_cols: list[str]) -> tuple:
    """채점 대상 행만 뽑은 최종 표를 시각·문항수가 박힌 CSV로 저장 → (표, 경로)."""
    df = progress.frame.reindex([r["id"] for r in rows]).reset_index(names="id")
    for c in ref_cols:  # 모범답안 없는 행은 정답대조 빈값이 정상
        if c in df.columns:
            df.loc[~df["id"].isin({r["id"] for r in ref_rows}), c] = float("nan")
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    out = Path(
        f"eval/results/ragas_retrieved_{stamp}_n{len(rows)}.csv"
    )  # 최종 스냅샷(시각·문항수) — 덮어쓰기 사고 방지(07-18)
    df.to_csv(out, index=False)
    return df, out


def _mean(df: "pd.DataFrame", col: str) -> float | None:
    v = df[col].mean() if col in df.columns else float("nan")
    return None if (v is None or math.isnan(v)) else float(v)


def compute(smoke: int | None = None) -> dict:
    """RAGAS 채점 실행 → 요약 반환 (per-sample CSV도 저장). 출력은 main이 담당.

    smoke: None이면 환경변수 SMOKE 사용. 반환: {'faithfulness', 'answer_relevancy', 'n', 'csv'}
    """
    rows = build_samples("retrieved")
    n_smoke = SMOKE if smoke is None else smoke
    if n_smoke:
        rows = rows[:n_smoke]
    ref_rows = [r for r in rows if r["has_ref"]] if WITH_REF else []
    EvaluationDataset.from_list([r["sample"] for r in rows])
    scorer = _make_scorer()

    # ── 묶음 채점 + 이어 돌리기 ──────────────────────────────────────────────
    # evaluate()는 전체가 끝나야 결과를 돌려준다 — 사용량 한도로 중단되면 완료분까지 사라진다.
    # RAGAS_BATCH문항씩 채점해 묶음마다 진행 파일에 덧붙이고, 재실행 시 지표가 다 찬 문항은 건너뛴다.
    # 결측(NaN)·미채점 칸만 다시 채점하므로 결측 메우기 스크립트가 따로 필요 없다.
    base_cols, base_metrics = _base_metrics()
    ref_cols = ["context_recall", "answer_correctness"]
    progress = _Progress.open()
    need_base = [r for r in rows if progress.missing(r, base_cols)]
    need_ref = [r for r in ref_rows if progress.missing(r, ref_cols)]
    print(
        f"진행 파일 {progress.path.name}: 3지표 남은 {len(need_base)}/{len(rows)} · 정답대조 남은 {len(need_ref)}/{len(ref_rows)}"
    )
    scorer.batch = int(os.getenv("RAGAS_BATCH", "50"))
    if need_base:
        scorer.run(progress, need_base, base_metrics, base_cols)
    if need_ref:
        scorer.run(progress, need_ref, [LLMContextRecall(), answer_correctness], ref_cols)

    df, out = _snapshot(progress, rows, ref_rows, ref_cols)
    return {
        "faithfulness": _mean(df, "faithfulness"),
        "answer_relevancy": _mean(df, "answer_relevancy"),
        "context_precision": _mean(df, "llm_context_precision_without_reference"),
        "context_recall": _mean(df, "context_recall"),
        "answer_correctness": _mean(df, "answer_correctness"),
        "n": len(rows),
        "n_ref": len(ref_rows),  # 정답 대조 2축의 분모
        # 지표별 결측(NaN) 수 — 0이 아니면 같은 명령을 다시 돌리면 그 칸만 다시 채점한다
        "missing": {
            c: int(df[c].isna().sum()) - (len(rows) - len(ref_rows) if c in ref_cols else 0)
            for c in base_cols + ref_cols
            if c in df.columns
        },
        "csv": str(out),
        "progress": str(progress.path),
    }


def main() -> None:
    print(f"judge={JUDGE}  |  SMOKE={SMOKE or '0(전체)'}")
    r = compute()
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k not in ("missing",)})
    if any(r["missing"].values()):
        print(
            "⚠ 결측(판정 실패·타임아웃) — 같은 명령을 다시 돌리면 그 칸만 다시 채점:",
            {k: v for k, v in r["missing"].items() if v},
        )
    print(f"→ saved {r['csv']}")


if __name__ == "__main__":
    main()
