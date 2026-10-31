"""합성 표 이미지가 그림으로 분류된 경우의 캡션 대조 (#247 결정 4 — #246·#253 코멘트에서 넘어온 항목).

#246에서 작은 글씨(V12)·저품질 캡처(V23)·표 안의 표(V15) 이미지는 docling이 표가 아니라 그림으로 분류해 OCR 대신 VLM 캡션으로
갔다. 정답은 하네스 정의(eval/table_parsing/specs) 자체라 사람 판정 없이 자동으로 센다 — 합성이라 외부 전송 문제도 없다.
  - 칸 값 포함: 정의의 칸 글자(공백 무시)가 캡션에 있는가 / 숫자 칸 포함: 숫자·금액·날짜 칸만
  - 그림으로 분류됐는지(표로 잡혔으면 OCR 경로라 이 대조의 대상이 아니다 — 그 경우 '표 분류'로 남긴다)
관계(어느 값이 어느 항목인지)는 자동으로 못 센다 — 캡션 원문을 결과에 실어 사람이 본다.
"""
import re
from pathlib import Path

from eval.doc_fidelity.pictures import norm
from eval.table_parsing.expected import expected_cells, is_numeric, flatten_nested

TARGETS = ('V12_small', 'V23_lowq', 'V15_nested')


def spec_values(spec) -> list[str]:
    rows = spec['rows']
    if any(isinstance(c, dict) and c.get('nested') for r in rows for c in r):
        rows = flatten_nested(rows)
    seen, out = set(), []
    for cell in expected_cells(rows):
        t = cell.text.strip()
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def score(values: list[str], caption: str) -> dict:
    cap = norm(caption)
    hit = [v for v in values if norm(v) in cap]
    nums = [v for v in values if is_numeric(v)]
    return {'values': len(values), 'values_hit': len(hit), 'numeric': len(nums),
            'numeric_hit': sum(1 for v in nums if norm(v) in cap), 'missing': [v for v in values if v not in hit]}


def run(out: Path) -> list[dict]:
    from docling_core.types.doc import PictureItem, TableItem
    from eval.table_parsing.build import build_img_pdf
    from eval.table_parsing.specs import BY_ID
    from rag import chunking as ch

    out.mkdir(parents=True, exist_ok=True)
    (out / 'png').mkdir(exist_ok=True)
    rows = []
    for sid in TARGETS:
        spec = BY_ID[sid]
        pdf = build_img_pdf(spec, out, out / 'png')
        doc = ch.docling_convert(pdf, max_num_pages=100)
        captions = [ch.docling_elements.docling_picture_caption(it) for it, _ in doc.iterate_items() if isinstance(it, PictureItem)]
        n_tables = sum(1 for it, _ in doc.iterate_items() if isinstance(it, TableItem))
        caption = '\n'.join(c for c in captions if c)
        r = {'id': sid, 'classified': '그림' if captions else ('표' if n_tables else '없음'), 'pictures': len(captions),
             'tables': n_tables, 'caption': caption, **score(spec_values(spec), caption)}
        print(f"{sid}: {r['classified']} 칸 {r['values_hit']}/{r['values']} 숫자 {r['numeric_hit']}/{r['numeric']}", flush=True)
        rows.append(r)
    return rows


# ── 합성 그림(#247 결정 9) — picture_specs의 정답과 실제 캡션 대조 ─────────────────────────────────────────────

CAPTION_MATCH_RATIO = 0.5     # 모양으로 못 짝지었을 때 캡션 적중률 하한
MATCH_MAX_DIST = 14.0          # 24×24 회색 평균 픽셀 차이 상한(0~255). 실측(10/18): 같은 그림 0~10.7, 다른 그림 최소 16.5

def _pair_hit(caption: str, a: str, b: str, labels: set[str]) -> bool:
    """(항목, 값)이 짝지어졌나 — 같은 줄, 또는 빈 줄로 나뉜 블록의 **첫 줄이 그 항목이고 블록에 다른 항목이 없을 때** 블록 안.

    VLM은 묶음형 차트를 '상암⏎정규직 24⏎계약직 11'처럼 블록으로 적는다(누적 막대 실측) — 같은 줄만 보면 관계가 맞는데도 짝 0이
    나왔다. 그렇다고 블록 전체를 허용하면 빈 줄 없는 캡션('상암⏎수원⏎24⏎31')이 한 블록이 되어 모든 짝이 맞는 것으로 나왔다(테스트).
    """
    na, nb = norm(a), norm(b)
    text = caption or ''
    if any(na in norm(x) and nb in norm(x) for x in text.split('\n')):
        return True
    for block in re.split(r'\n\s*\n', text):
        lines = [norm(x) for x in block.split('\n') if x.strip()]
        if lines and na in lines[0] and any(nb in x for x in lines[1:]) \
                and not any(norm(o) in x for o in labels - {a} for x in lines):
            return True
    return False


def score_picture(spec: dict, caption: str) -> dict:
    """정답 정의 ↔ 캡션. texts·numbers는 포함, pairs는 같은 줄, sequence는 첫 등장 위치가 오름차순인지."""
    cap = norm(caption)
    texts, nums = spec.get('texts', []), spec.get('numbers', [])
    pairs, seq = spec.get('pairs', []), spec.get('sequence', [])
    pos = [cap.find(norm(t)) for t in seq]
    return {
        'texts': len(texts), 'texts_hit': sum(norm(t) in cap for t in texts),
        'numbers': len(nums), 'numbers_hit': sum(norm(n) in cap for n in nums),
        'pairs': len(pairs), 'pairs_hit': sum(_pair_hit(caption, a, b, {x for x, _ in pairs}) for a, b in pairs),
        'sequence_ok': (all(p >= 0 for p in pos) and pos == sorted(pos)) if seq else None,
        'missing_texts': [t for t in texts if norm(t) not in cap],
    }


def _thumb(image):
    """짝짓기용 지문 — 회색 24×24. 크롭은 쪽 렌더(PDF)·원본(DOCX)이라 크기·여백이 조금 달라도 모양은 같다."""
    return list(image.convert('L').resize((24, 24)).getdata())


def match_pictures(specs: list[tuple[str, dict]], records: list[dict], crops_dir) -> list[tuple[str, dict | None]]:
    """정의 ↔ 추출 그림 짝짓기 — **크롭 이미지와 정의 그림의 모양**으로 짓는다(캡션과 무관).

    캡션 글자로 짓던 첫 구현은 틀렸다(10/18): 적중 1개면 남의 그림을 가져오고('22:00'), 한 자리 숫자('2','1')는 어느 캡션에나
    있고, 캡션 없는 작은 상자는 구별할 수 없었다. 캡션이 틀리거나 없는 그림이 정작 보려는 대상이라 캡션에 기대면 안 된다.
    거리(평균 픽셀 차이)가 MATCH_MAX_DIST를 넘으면 짝이 없다 — 그림으로 검출되지 않았다는 뜻이고, 그 내용이 글자(OCR)·헤딩으로
    들어갔는지는 run.phase_synthetic_pictures가 색인에서 따로 본다.
    """
    from PIL import Image
    fps = {}
    for r in records:
        if r.get('crop'):
            fps[r['picture_id']] = _thumb(Image.open(crops_dir / r['crop']))
    pairs = []
    for sid, spec in specs:
        ref = _thumb(spec['draw']())
        for r in records:
            if r['picture_id'] in fps:
                dist = sum(abs(a - b) for a, b in zip(ref, fps[r['picture_id']])) / len(ref)
                pairs.append((dist, sid, r['picture_id']))
    used_s, used_r, got = set(), set(), {}
    for dist, sid, pid in sorted(pairs):
        if dist > MATCH_MAX_DIST or sid in used_s or pid in used_r:
            continue
        used_s.add(sid); used_r.add(pid); got[sid] = pid
    by_id = {r['picture_id']: r for r in records}
    # 모양으로 못 짝지은 정의는 캡션의 정답 적중률로 다시 찾는다 — docling이 그림 영역을 일부만 자르거나(산식 상자의 제목 띠가
    # 빠짐, 거리 24.4) 붙은 그림 둘을 하나로 합치면(4%·6% 상자 → 그림 1장) 모양이 정의와 달라진다(10/18 실측). 이 단계는 이미
    # 짝지어진 그림도 다시 쓸 수 있다 — 한 그림에 정의 둘이 걸리면 합쳐진 그림이다(run.phase_synthetic_pictures가 'merged'로 표시).
    for sid, spec in specs:
        if sid in got:
            continue
        best, best_ratio = None, CAPTION_MATCH_RATIO
        for r in records:
            ratio = _caption_ratio(spec, r.get('caption') or '')
            if ratio >= best_ratio:
                best, best_ratio = r['picture_id'], ratio
        if best:
            got[sid] = best
    return [(sid, by_id.get(got.get(sid))) for sid, _ in specs]


def _caption_ratio(spec: dict, caption: str) -> float:
    """정답 글자 + 두 자리 이상 숫자 중 캡션에 있는 몫. 한 자리 숫자('2','1')는 어느 캡션에나 있어 짝짓기 근거가 못 된다."""
    cap = norm(caption)
    keys = [t for t in spec.get('texts', [])] + [n for n in spec.get('numbers', []) if len(norm(n)) >= 2]
    return sum(norm(k) in cap for k in keys) / len(keys) if keys and cap else 0.0
