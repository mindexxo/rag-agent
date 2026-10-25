"""합성 스캔의 청킹 결과를 원본 디지털 PDF와 대조한다(#224) — 제목·헤딩 경로·문장·띄어쓰기.

정답은 같은 원본 디지털 PDF를 같은 청킹으로 돌린 결과다(글자 층을 그대로 읽으므로 원문과 같다 — #231 문장 포함률 PDF 0.98~).
변환은 이 프로세스에서 돌린다(자식 프로세스를 거치면 OCR 설정을 실험으로 바꿀 수 없다 — eval.scan_quality.ocr_variant).
쪽 전체 OCR 모드(OcrMode.FULL_PAGE)는 1차 실측에서 기각했다 — 제목은 고치나 본문 띄어쓰기 유지 0.64→0.16·글자 0.993→0.782(100dpi).

지표(문서마다):
  title_ok        첫 헤딩(heading_path[0], 문서 제목)이 원본과 같은가 — 공백 무시 글자 유사도 0.95 이상
  heading_recall  원본 헤딩 중 스캔 헤딩에 (공백 무시) 그대로 있는 비율 — 깨진 헤딩은 문서의 모든 청크 경로로 퍼진다
  sent_exact      원본 문장 중 스캔 본문에 글자 그대로(공백 포함) 있는 비율
  sent_nospace    공백을 무시하면 있는 비율 — exact와의 차가 띄어쓰기만 틀린 문장
  char_acc        공백 뺀 전체 글자 일치율(difflib ratio)
  space_kept      원본 띄어쓰기 중 스캔에 남은 비율(공백 뺀 글자 정렬 위에서 셈)

사용: python -m eval.scan_quality.compare [remote|local_min736|local_max|local_min400|vlm ...]  (ocr_variant.VARIANTS + vlm=#296 쪽 전체 VLM)   → eval/results/scan224/compare.json · compare.html
"""
import difflib
import html
import json
import re
import sys

from eval.scan_quality.scans import OUT, SPECS

_SENT = re.compile(r'(?<=[.!?。다요])\s+|\n')


def _ns(t: str) -> str:
    return re.sub(r'\s+', '', t)


def _chunk(path, variant: str):
    import rag.chunking as C
    from eval import vlm_engine
    from eval.scan_quality.ocr_variant import install
    vlm_engine.uninstall()                 # 앞 모드가 vlm이었으면 원복 — 디지털 기준·OCR 변형이 VLM을 타지 않게(#296)
    if variant == 'vlm':
        vlm_engine.install()               # 쪽 전체 VLM(VLM_CAPTION_* 설정의 벤더), #296
    else:
        install(variant)
    return C.chunk_file(path)


def _headings(chunks) -> list[str]:
    seen = []
    for c in chunks:
        for h in c.heading_path:
            if h not in seen:
                seen.append(h)
    return seen


def _sentences(chunks) -> list[str]:
    body = '\n'.join(c.text for c in chunks)
    return [s.strip() for s in _SENT.split(body) if len(_ns(s)) >= 8 and not s.strip().startswith('|')]


def _space_kept(ref: str, hyp: str) -> float:
    """공백을 뺀 두 글을 정렬하고, 원본에서 공백 바로 앞이던 글자 중 스캔에서도 공백 앞인 비율."""
    def strip(t):
        chars, before_space = [], []
        for i, ch in enumerate(t):
            if ch.isspace():
                if before_space:
                    before_space[-1] = True
                continue
            chars.append(ch)
            before_space.append(False)
        return ''.join(chars), before_space
    rc, rs = strip(ref)
    hc, hs = strip(hyp)
    kept = 0                                          # 정렬되지 않은 구간의 공백은 '남지 않음'으로 센다(보수적)
    for a, b, n in difflib.SequenceMatcher(None, rc, hc, autojunk=False).get_matching_blocks():
        for k in range(n):
            if rs[a + k]:
                kept += hs[b + k]
    total_all = sum(rs)
    return kept / total_all if total_all else 1.0


def measure(digital, scan) -> dict:
    dh, sh = _headings(digital), _headings(scan)
    ds = _sentences(digital)
    scan_body = '\n'.join(c.text for c in scan)
    scan_ns = _ns(scan_body)
    dtext, stext = '\n'.join(c.text for c in digital), scan_body
    return {
        'title': dh[0] if dh else None, 'scan_title': sh[0] if sh else None,
        # 공백 무시 글자 유사도 0.95 이상이면 같은 제목 — 'ㆍ'(한글 자모)를 OCR이 '·'로 읽는 정도는 깨짐이 아니다(harim_13 실측)
        'title_sim': round(difflib.SequenceMatcher(None, _ns(dh[0]), _ns(sh[0])).ratio(), 3) if dh and sh else 0.0,
        'title_ok': bool(dh and sh and difflib.SequenceMatcher(None, _ns(dh[0]), _ns(sh[0])).ratio() >= 0.95),
        'heading_recall': sum(_ns(h) in {_ns(x) for x in sh} for h in dh) / len(dh) if dh else 1.0,
        'bad_headings': [h for h in sh if _ns(h) not in {_ns(x) for x in dh}],
        'sentences': len(ds),
        'sent_exact': sum(s in scan_body for s in ds) / len(ds) if ds else 1.0,
        'sent_nospace': sum(_ns(s) in scan_ns for s in ds) / len(ds) if ds else 1.0,
        'char_acc': difflib.SequenceMatcher(None, _ns(dtext), scan_ns, autojunk=False).ratio(),
        'space_kept': _space_kept(dtext, stext),
        'chunks': len(scan), 'digital_chunks': len(digital),
        'missing_sentences': [s for s in ds if _ns(s) not in scan_ns][:15],
    }


def main(modes: list[str]) -> None:
    rows = []
    digital = {s.filename: _chunk(s.src, 'remote') for s in SPECS}
    for mode in modes:
        for s in SPECS:
            m = measure(digital[s.filename], _chunk(s.out, mode))
            rows.append({'file': s.filename, 'condition': s.note, 'mode': mode, **m})
            print(f"{mode:<10} {s.filename[:28]:<30} title {'✓' if m['title_ok'] else '✗'} head {m['heading_recall']:.2f} "
                  f"sent {m['sent_exact']:.2f}/{m['sent_nospace']:.2f} char {m['char_acc']:.3f} space {m['space_kept']:.2f}", flush=True)
    (OUT / 'compare.json').write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    e = html.escape
    body = ''.join(
        f"<section><h3>{e(r['file'])} · {e(r['condition'])} · <code>{r['mode']}</code></h3>"
        f"<p>제목: {e(r['title'] or '')} → <b>{e(r['scan_title'] or '')}</b> {'✓' if r['title_ok'] else '✗'}</p>"
        f"<p>헤딩 일치 {r['heading_recall']:.2f} · 문장(글자 그대로) {r['sent_exact']:.2f} · 문장(공백 무시) {r['sent_nospace']:.2f} · "
        f"글자 {r['char_acc']:.3f} · 띄어쓰기 유지 {r['space_kept']:.2f}</p>"
        f"<details><summary>원본에 없는 스캔 헤딩 {len(r['bad_headings'])}</summary><pre>{e(chr(10).join(r['bad_headings']))}</pre></details>"
        f"<details><summary>공백 무시해도 없는 원본 문장 {len(r['missing_sentences'])}</summary><pre>{e(chr(10).join(r['missing_sentences']))}</pre></details></section>"
        for r in rows)
    (OUT / 'compare.html').write_text(f"<!doctype html><meta charset=utf-8><title>#224 스캔 대조</title><style>body{{font-family:sans-serif;max-width:1100px;margin:20px auto;padding:0 16px}}section{{border-top:1px solid #ccc}}pre{{white-space:pre-wrap;background:#f5f5f5;padding:8px}}</style><h1>#224 합성 스캔 ↔ 원본 디지털 대조</h1>{body}")


if __name__ == '__main__':
    main(sys.argv[1:] or ['remote'])
