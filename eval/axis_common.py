"""색인에서 질문셋을 만드는 검색축들의 공용부 — 조 번호 축(eval/retrieval_article)·키워드 축(eval/retrieval_keyword, #248).

두 축 모두 질문셋 파일이 없다 — 실행할 때마다 색인에서 만든다(재색인하면 고정 정답이 낡고, eval/gold_private에 두면
eval.generation.load_gold가 다른 축에 병합해 버린다). 그래서 "색인 조각 읽기"와 "결과 지문"이 같아야 두 축의 결과를
같은 눈금으로 읽을 수 있다 — 여기 한 곳에 둔다.
"""
import re
import subprocess

from config import settings
from rag.os_client import client
from rag.os_search import tenant_filter


async def load_chunks(tenant: str) -> list[dict]:
    """테넌트의 검색 가능 조각 전부(운영 검색과 같은 필터). FAQ 조각(document_id 없음)은 뺀다 — 조 번호 축은 FAQ에 조가
    없고, 키워드 축은 측정 테넌트에 FAQ가 0건이라(2026-10-18 실측) 넣을 이유가 아직 없다(#248 설계 3)."""
    resp = await client().search(index=settings.opensearch_index, body={
        "size": 10000, "track_total_hits": True,
        "_source": ["chunk_id", "document_id", "filename", "heading_path", "headings", "text", "table_text"],
        "query": tenant_filter(tenant),
    })
    total = resp["hits"]["total"]["value"]
    if total > 10000:
        raise RuntimeError(f"{tenant}: 조각 {total}개 — size 상한 초과, 페이지네이션 필요")
    return [{"chunk_id": int(s["chunk_id"]), "document_id": s["document_id"], "filename": s["filename"],
             "heading_path": list(s.get("heading_path") or []), "headings": list(s.get("headings") or []),
             "text": s.get("text") or "", "table_text": s.get("table_text")}   # 표 묶음이면 표 전체(#257) — chunk_boundary가 본다
            for h in resp["hits"]["hits"] if (s := h["_source"]).get("document_id") is not None]


def commit() -> str:
    """결과 지문의 커밋 — 측정 결과를 인용할 때 어느 커밋인지 함께 적는다(AGENTS.md)."""
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    except OSError:
        return "?"


def chunk_head(text: str, n: int = 120) -> str:
    """사람이 읽는 덤프용 조각 머리 — 줄바꿈을 ⏎로 접는다."""
    return text[:n].replace("\n", " ⏎ ")


def norm_text(s: str) -> str:
    """부분문자열 정답 매칭 정규화 — 공백 제거 + 대소문자 무시(키워드 축 #248)."""
    return re.sub(r"\s+", "", s).casefold()


def evidence_has(evidence, token: str) -> float:
    """모델이 실제로 받는 근거 **글**에 토큰이 있나(키워드 축 합격 기준, #248). 조각 id로 보면 안 된다: 표는 같은 시트 조각을
    가장 위 하나로 접고 시트 완성본으로 바꿔 넘기므로(rag.retriever.prepare_evidence), 정답 행이 든 형제 조각이 id로는 빠져도
    모델은 그 행을 본다(첫 실행 실측: R@5=1인데 id 기준 근거 밖 4건이 전부 이 꼴, 2026-10-18)."""
    n = norm_text(token)
    return float(any(n in norm_text(c.text) for c in evidence))
