"""#244 실행기 — 배치 유형별 합성 PDF를 워커와 같은 변환기(docling)로 읽어 **읽기 순서**를 정답과 대조하고, 조각 단계에서 단이 섞였는지 센다.

채점(정의점):
  - 문단 보존(paragraph_fate): 문단 본문(specs.paragraph_bodies)이 docling 요소에 어떻게 남았나 — 세 갈래.
      intact = 한 요소 안에 연속으로 남음 / split = 둘 이상 요소로 나뉘었지만 **글자를 이어 붙이면 원문**(단 끝·쪽 끝에서 마지막 줄이 별도
      요소가 된 꼴 — 순서는 맞고 글자도 다 있다, 넓은 2단 실측) / **mixed = 글자를 이어도 원문이 안 됨**(다른 단의 줄이 끼어들거나 줄이 사라짐
      — 좁은 2단 실측: 두 단을 가로로 한 줄씩 이어 읽음). **단 섞임의 주 지표는 mixed**다. 첫 채점은 표식 순서만 봐서 1/4로 과소 집계했고
      (리뷰 지적 1), 둘째 채점(연속 문자열 여부)은 split을 mixed와 섞었다.
  - docling 순서: `docling_elements.docling_elements` 출력에서 문단 표식(`[P07]`)을 등장 순서대로 뽑아 정답(specs.expected_order)과 비교.
      inversions = 인접 쌍 가운데 정답 순서가 뒤집힌 수, displaced = 정답 위치에서 멀리(2칸 이상) 밀린 표식 수(블록이 통째로 밀린 L05 같은 꼴 —
      인접 쌍 역전은 1로만 센다, 리뷰 지적 5), lost = 정답에 있는데 출력에 없는 표식, dup = 두 번 나온 표식,
      merged = 한 요소에 표식이 둘 이상(이웃 문단이 한 덩어리로 합쳐짐 — 순서는 맞아도 문단 경계를 잃음)
  - 조각: `chunk_file`로 만든 조각 **본문**에서 표식이 정답 순서대로인지(제목 경로로 올라간 문단은 순서와 무관). 내림이 있으면 '섞임'(다른 단·다른 자리의 문단이 한
      조각에 들어옴 — 검색·인용이 오염되는 모양). 빈틈은 섞임이 아니다 — 표 캡션·헤딩으로 올라간 문단이 빈틈을 만든다(첫 측정의
      "연속 구간" 판정은 L04·L02에서 거짓 양성을 냈다).
  - 머리글·바닥글(L07): 요소 글에 furniture 문자열이 들어오면 '본문 오염'.
운영 코드 무변경. 변환은 `convert_process.convert_local`(인프로세스)로 한 번만 하고 그 문서를 chunk_file에 끼운다(표 축 #246과 같은 방식). OCR은 끈다(순서
판정과 무관, 원격 의존 제거 — 끝나면 복구). 그림 캡션 설정은 운영 그대로다 — L03의 그림이 캡션 대상이면 VLM이 호출된다(안 닿으면 변환 실패).
용법: python -m eval.pdf_layout.run [--out eval/results/pdf_layout] [--only L01_two_col_narrow ...]
산출물: <out>/docs/*.pdf, results.jsonl(건별 — 실제 순서·조각 글 포함), matrix.md(유형 × 지표).
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eval.pdf_layout.build import build_all          # noqa: E402
from eval.pdf_layout.specs import SPECS, expected_order, paragraph_bodies   # noqa: E402

TOKEN = re.compile(r'\[(P\d+)\]')


def tokens_in(text: str) -> list[str]:
    return TOKEN.findall(text or '')


def order_score(expected: list[str], got: list[str]) -> dict:
    pos = {t: i for i, t in enumerate(expected)}
    seq = [t for t in got if t in pos]
    inv = sum(1 for a, b in zip(seq, seq[1:]) if pos[a] > pos[b])
    seen, dup, first = set(), 0, []
    for t in seq:
        dup += t in seen
        if t not in seen:
            first.append(t)
        seen.add(t)
    displaced = sum(1 for i, t in enumerate(first) if abs(pos[t] - i) >= 2)   # 정답 자리에서 2칸 이상 떨어진 표식
    return {'inversions': inv, 'displaced': displaced, 'lost': [t for t in expected if t not in seen], 'dup': dup, 'n': len(expected)}


def _norm(s: str) -> str:
    """공백·줄바꿈·NBSP를 모두 지운다 — docling이 단 끝 문단의 마지막 줄을 별도 요소로 내면 조각엔 줄바꿈 하나만 끼고 글자 순서는
    그대로다(넓은 2단 실측). 그건 찢어짐이 아니다 — 글자 사이에 **다른 글자**가 끼어든 것만 찢어짐이다."""
    return re.sub(r'[\s\xa0]+', '', s or '')


def paragraph_fate(bodies: dict[str, str], texts: list[str]) -> dict[str, str]:
    """표식 → 'intact' | 'split' | 'mixed'. split 판정 = 모든 글(요소)을 순서대로 이어 붙인 한 문자열 안에 본문이 연속으로 있음(요소 경계에서만
    끊긴 것). 이어 붙여도 없으면 mixed — 다른 글자가 끼었거나 줄이 사라졌다."""
    blobs = [_norm(t) for t in texts]
    joined = ''.join(blobs)
    out = {}
    for tok, body in bodies.items():
        b = _norm(body)
        out[tok] = 'intact' if any(b in x for x in blobs) else 'split' if b in joined else 'mixed'
    return out


def chunk_mixing(expected: list[str], chunks: list[list[str]]) -> dict:
    """조각 안에서 표식이 정답 순서대로(오름차순)인지. 내림이 한 번이라도 있으면 그 조각은 '섞임' — 다른 단·다른 자리의 문단이 끼어든
    모양이다. 빈틈(표·캡션·헤딩으로 올라간 문단)은 섞임이 아니다. 표식 없는 조각은 제외."""
    pos = {t: i for i, t in enumerate(expected)}
    mixed, checked = [], 0
    for i, toks in enumerate(chunks):
        idx = [pos[t] for t in toks if t in pos]
        if not idx:
            continue
        checked += 1
        if any(a > b for a, b in zip(idx, idx[1:])):
            mixed.append({'chunk': i, 'tokens': toks})
    return {'chunks_checked': checked, 'mixed': mixed}


def evaluate(spec: dict, pdf: Path) -> dict:
    import rag.chunking as ch
    from config import settings
    ocr = settings.docling_do_ocr
    settings.docling_do_ocr = False            # 순서 판정과 무관 — 원격 OCR 의존을 뺀다(끝나면 복구)
    try:
        doc = ch.convert_process.convert_local(pdf, max_num_pages=settings.docling_max_num_pages)
    finally:
        settings.docling_do_ocr = ocr
    elements = list(ch.docling_elements.docling_elements(doc))
    expected, bodies = expected_order(spec), paragraph_bodies(spec)
    # 소제목(h) 블록은 제목 경로로 올라가 조각 본문에 없는 게 정상 — 조각 찢어짐 판정에서 뺀다(요소 단계에선 본문에 있으니 둔다)
    heading_toks = {tokens_in(b[1])[0] for b in spec['blocks'] if b[0] == 'h'}
    body_only = {k: v for k, v in bodies.items() if k not in heading_toks}
    got, merged = [], 0
    for label, text, _page, _rows in elements:
        toks = tokens_in(text)
        merged += len(toks) > 1
        got += toks
    labels = [(label, tokens_in(text), (text or '')[:60]) for label, text, _, _ in elements]   # 요소마다 표식 전부(합쳐짐 확인용)
    fate = paragraph_fate(bodies, [t for _, t, _, _ in elements])
    furniture = [f for f in spec.get('furniture', []) if f != '쪽' and any(f in (t or '') for _, t, _, _ in elements)]

    # 조각 — 변환을 다시 하지 않고 같은 문서를 끼운다(표 축 #246 run.py와 같은 방식)
    orig = ch.docling_convert
    ch.docling_convert = lambda *a, **k: doc
    try:
        chunks = ch.chunk_file(pdf)
    finally:
        ch.docling_convert = orig
    # 조각 본문의 표식만 본다 — 헤딩으로 올라간 문단(docling이 표 도입 문장을 section_header로 내기도 한다, L04)은 제목 경로에
    # 있어 순서와 무관하고, 본문 앞에 붙이면 거짓 내림이 난다
    chunk_toks = [tokens_in(c.text) for c in chunks]
    return {'id': spec['id'], 'layout': spec['layout'], 'pages': len(doc.pages), 'expected': expected, 'got': got,
            'order': order_score(expected, got), 'merged_elements': merged, 'furniture_in_body': furniture,
            'fate': fate, 'mixed': sorted(t for t, f in fate.items() if f == 'mixed'), 'split': sorted(t for t, f in fate.items() if f == 'split'),
            'chunk_fate': paragraph_fate(body_only, [c.text for c in chunks]),
            'chunks': chunk_mixing(expected, chunk_toks), 'n_chunks': len(chunks),
            'pictures': sum(1 for l, *_ in elements if l == 'picture'),
            'elements': labels, 'chunk_texts': [c.text for c in chunks]}


def matrix(rows: list[dict]) -> str:
    out = ['| 배치 | 쪽 | 문단 | **섞임(mixed)** | 분리(split) | 순서 뒤집힘 | 멀리 밀림 | 소실 | 합쳐짐 | 조각 수 | 순서 섞인 조각 | 머리글 오염 | 그림 |',
           '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for r in rows:
        o = r['order']
        out.append(f"| {r['id']} | {r['pages']} | {o['n']} | **{len(r['mixed'])}** | {len(r['split'])} | {o['inversions']} | {o['displaced']} | "
                   f"{len(o['lost'])} | {r['merged_elements']} | {r['n_chunks']} | {len(r['chunks']['mixed'])}/{r['chunks']['chunks_checked']} | "
                   f"{len(r['furniture_in_body'])} | {r['pictures']} |")
    return '\n'.join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='eval/results/pdf_layout')
    ap.add_argument('--only', nargs='*')
    args = ap.parse_args()
    out = Path(args.out)
    pdfs = {p.stem: p for p in build_all(out / 'docs')}
    specs = [s for s in SPECS if not args.only or s['id'] in args.only]
    rows = []
    with open(out / 'results.jsonl', 'w', encoding='utf-8') as fp:
        for spec in specs:
            r = evaluate(spec, pdfs[spec['id']])
            rows.append(r)
            fp.write(json.dumps(r, ensure_ascii=False) + '\n'); fp.flush()
            o = r['order']
            print(f"{r['id']:20} 섞임 {len(r['mixed']):2}/{o['n']:2}  분리 {len(r['split']):2}  뒤집힘 {o['inversions']} 밀림 {o['displaced']} 소실 {len(o['lost'])} 합쳐짐 {r['merged_elements']} 그림 {r['pictures']}", flush=True)
    from eval.axis_common import commit
    (out / 'matrix.md').write_text(f'# #244 배치 유형별 읽기 순서 (커밋 {commit()})\n\n{matrix(rows)}\n', encoding='utf-8')
    print('→', out / 'matrix.md')


if __name__ == '__main__':
    main()
