"""#235 문서 규모가 커질 때 검색 품질 — 노이즈 문서 주입 곡선·근사 탐색 일치율·IDF 공유 영향.

질문: 검색 대상이 수백·수천 문서로 늘면 엉뚱한 조각이 올라와 품질이 떨어지는가(세미나 27장 지적). 모든 수치가
테넌트당 수백 조각에서 나왔고, 규모를 키워 잰 적이 없다.

방법(이슈 #235 + 10/23 보정):
  1. 노이즈 문서를 LLM(운영 14B)으로 생성한다 — 회사 규정 꼴(측정 테넌트에 **없는** 규정 종류·가상 회사명)과 다른 도메인
     (쇼핑몰·여행·약관류). 골드의 사실을 다른 값으로 적은 문서는 만들지 않는다(규모 실험이지 상충 실험이 아니다) —
     generate가 주제로 피하고 filter가 골드 스니펫·수치 겹침을 기계로 걸러낸다.
  2. 운영과 같은 청킹(chunk_file)·같은 색인 텍스트(build_index_text)·같은 임베딩(TEI)으로 조각을 만든다(chunks).
  3. 운영 인덱스를 **임시 인덱스**로 통째 복사한 뒤(#234 방식) 노이즈 조각을 측정 테넌트(inticube)로 넣는다 — 규모별
     (x0 복사만 / x5 / x10 / x30 = 측정 테넌트 조각 수의 배수). 노이즈 tenant_id가 측정 테넌트여야 하는 이유: 다른 테넌트로
     넣으면 kNN 필터가 선택적이라 Lucene이 정확 탐색으로 빠져 근사 열화가 숨는다. IDF 공유 실험(idf)은 같은 노이즈를
     **다른** tenant_id로 넣는다 — BM25 df·avgdl은 인덱스 전체라 측정 테넌트 점수가 움직이는지 본다.
     운영 인덱스는 읽기만 한다. 임시 인덱스 이름은 `<운영 인덱스>_exp235_<조건>`, drop은 그 이름만 지운다.
  4. 측정(measure) — 두 층. ⓐ dense 풀: 같은 질의 벡터로 근사(운영 knn_clause)와 정확(script_score knn_score) 상위 30을 뽑아
     겹침·정답 순위를 비교(리랭커 앞 단계 — 근사 오차는 여기에만 나타난다). ⓑ 최종: eval/retrieval_v2.compute를 측정
     테넌트만 돌려 Hit@1·R@5·R@20·MRR. 떨어진 문항은 상위 5를 원문과 나란히 덤프해 "엉뚱한 조각"의 유형을 사람이 분류한다.

사용(저장소 루트, 개발계 DB·OpenSearch·TEI·vLLM):
    python -m eval.scale_noise generate --n 1200      # 노이즈 문서 생성(이어 돌리기 가능) → eval/results/scale_noise/docs/
    python -m eval.scale_noise filter                 # 골드 겹침 검사 → filter.json
    python -m eval.scale_noise chunks                 # 청킹·임베딩 → chunks.jsonl
    python -m eval.scale_noise build [--scales x0,x5,x10,x30,idf]
    python -m eval.scale_noise measure                # → measure.json·report.md·regressions.md
    python -m eval.scale_noise drop
결과(eval/results/scale_noise/)는 gitignore — 생성 문서는 재현성이 없고(LLM 출력), 회귀 덤프에 실문서 조각이 들어간다.
"""
from pathlib import Path

from config import settings

OUT = Path(__file__).resolve().parents[2] / 'eval' / 'results' / 'scale_noise'   # cwd 무관(워크트리에서도 그 트리의 results)
DOCS = OUT / 'docs'
MANIFEST = OUT / 'manifest.jsonl'
FILTER = OUT / 'filter.json'
CHUNKS = OUT / 'chunks.jsonl'

TENANT = 'inticube'              # 측정 테넌트 — 실문서 골드가 있는 유일한 테넌트
NOISE_TENANT = 'noise235'        # IDF 공유 실험에서 노이즈가 받는 tenant_id
SCALES = {'x0': 0, 'x5': 5, 'x10': 10, 'x30': 30}   # 측정 테넌트 조각 수의 배수만큼 노이즈를 더한다
CONDITIONS = [*SCALES, 'idf']
NOISE_DOC_ID_BASE = 1_000_000_000   # 노이즈 document_id 시작 — PG 행이 없으므로 실제 id(수만 대)와 겹치지 않는 큰 값

_BASE_INDEX = settings.opensearch_index   # condition()이 settings를 바꿔 끼우므로 읽을 때 한 번 고정(#234와 같음)


def index_name(cond: str) -> str:
    return f'{_BASE_INDEX}_exp235_{cond}'
