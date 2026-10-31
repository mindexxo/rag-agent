"""#235 노이즈 문서의 골드 겹침 검사 — 정답을 흔들 수 있는 문서를 버린다(정의점).

규칙(10/23 사용자 결정: 규모만 늘리고 정답은 흔들지 않는다):
  0. 잘린 문서(finish_reason != stop)는 버린다 — 규정 전문 꼴이 아니다.
  1. 스니펫 포함: 골드 expected_chunks[].snippet(원문 발췌, 공백·대소문자 무시 15자 이상)이 노이즈 본문에 그대로 있으면 버린다.
  2. 사실 겹침: **측정 테넌트** 골드의 expected_points·expected_answer에서 뽑은 "숫자+단위" 토큰(14일·6,000원·30% …)이 **둘 이상**
     노이즈 본문에 있고(숫자 앞 경계를 봐서 '2일'이 '12일'에 걸리지 않게), 그 문항의 핵심어(3자 이상 한글 어절, 길이·사전순으로 고른
     다섯의 앞 3자 — 조사가 붙어도 걸리게)가 하나라도 같이 있으면 버린다. 숫자만으로 막지 않는 이유: '30일' 같은 값은 여러 규정이
     각자 갖고 있어 금지어로 쓰면 오탐이다(eval/upload_corpus_v2.py 선례).
스니펫 규칙은 load_gold() 전체(정본 + 비공개)를, 사실 규칙은 측정 테넌트 골드만 본다 — 노이즈는 측정 테넌트에만 들어가므로 모의
테넌트 질문의 후보는 바뀌지 않는다(테넌트 필터). 스모크 4편에서 모의 골드까지 사실 규칙을 걸자 다른 도메인 문서가 '배송비 3,000원'
같은 흔한 값으로 전부 버려졌다(10/23) — 그건 상충이 아니라 같은 업종의 흔한 숫자다.
산출: FILTER = {'kept': [...], 'dropped': [{'id','reason','gold_id','hit'}], 'stats': {...}}. 표본은 사람이 읽는다(docs/ 원문).
"""

import json
import re
from collections import Counter

from eval.axis_common import norm_text
from eval.generation import load_gold, row_tenant
from eval.scale_noise import DOCS, FILTER, MANIFEST, TENANT

_NUM = re.compile(r'\d[\d,.]*\s*(?:만\s*원|천\s*원|억\s*원|원|일|개월|년|주|시간|회|명|%|퍼센트|건)')
_WORD = re.compile(r'[가-힣]{3,}')


def _facts(g: dict) -> tuple[set[str], list[str]]:
    texts = [*(g.get('expected_points') or []), g.get('expected_answer') or '']
    nums = {norm_text(m.group()) for t in texts for m in _NUM.finditer(t)}
    words = sorted({w for t in texts for w in _WORD.findall(t)}, key=lambda w: (-len(w), w))[:5]
    return nums, [norm_text(w)[:3] for w in words]


def _has_num(body: str, tok: str) -> bool:
    return re.search(r'(?<![\d,.])' + re.escape(tok), body) is not None


def check(doc_text: str, gold: list[dict], tenant: str = TENANT) -> dict | None:
    """버릴 이유가 있으면 {'reason','gold_id','hit'}, 없으면 None."""
    body = norm_text(doc_text)
    for g in gold:
        for ec in g.get('expected_chunks') or []:
            s = norm_text(ec.get('snippet') or '')
            if len(s) >= 15 and s in body:
                return {'reason': 'snippet', 'gold_id': g['id'], 'hit': s[:40]}
    for g in gold:
        if row_tenant(g) != tenant:
            continue
        nums, keys = _facts(g)
        hit = [x for x in nums if _has_num(body, x)]
        if len(hit) >= 2 and any(k in body for k in keys):
            return {'reason': 'facts', 'gold_id': g['id'], 'hit': hit[:4]}
    return None


def run() -> None:
    gold = load_gold()
    rows = [json.loads(l) for l in MANIFEST.read_text(encoding='utf-8').splitlines() if l.strip()]
    kept, dropped = [], []
    for r in rows:
        if r.get('finish') != 'stop':
            dropped.append({'id': r['id'], 'reason': 'truncated', 'gold_id': None, 'hit': r.get('finish')})
            continue
        v = check((DOCS / f"{r['id']}.md").read_text(encoding='utf-8'), gold)
        (dropped if v else kept).append({'id': r['id'], **v} if v else r['id'])
    kept_set = set(kept)
    stats = {
        'total': len(rows),
        'kept': len(kept),
        'dropped': len(dropped),
        'reasons': dict(Counter(d['reason'] for d in dropped)),
        'by_kind': dict(Counter(r['kind'] for r in rows if r['id'] in kept_set)),
    }
    FILTER.write_text(
        json.dumps({'kept': kept, 'dropped': dropped, 'stats': stats}, ensure_ascii=False, indent=1), encoding='utf-8'
    )
    print(stats)
    for d in dropped[:10]:
        print('  버림', d)
