"""표 E2E 리뷰 — manifest·answers로 사람이 볼 review.html(정의·색인 청크·Q/A)과 summary.md(축 × 경로 집계)를 쓴다."""

from collections import defaultdict
from pathlib import Path
from typing import Any, cast

from eval.table_parsing.expected import expected_rows, expected_tables, norm
from eval.table_parsing.llm_judge import final_ok
from eval.table_parsing.questions import judge
from eval.table_parsing.render_image import cell_text
from eval.table_parsing.specs import BY_ID, PATH_LABEL, axis_name, axis_no

_SPECS = cast('dict[str, dict[str, Any]]', BY_ID)

PATHS = ('img', 'pdf', 'docx', 'xlsx')
_LLM_NAME = {'correct': '맞음', 'wrong': '틀림', 'ungrounded': '근거 없음', 'not_answered': '미답'}
_STYLE = (
    '<style>body{font-family:-apple-system,sans-serif;margin:24px;background:#fafafa}.c{background:#fff;border:1px solid #ddd;'
    'border-radius:8px;padding:12px;margin:16px 0}.row{display:flex;gap:16px}.col{flex:1;min-width:0}img{max-width:100%;border:1px dashed #ccc}'
    'pre{white-space:pre-wrap;font-size:12px;background:#f6f6f6;padding:6px;border-radius:4px}.chunk{margin-bottom:6px}.hp{color:#888;font-size:11px}'
    'table{border-collapse:collapse;font-size:13px;width:100%}td,th{border:1px solid #ddd;padding:4px 8px;vertical-align:top}'
    'tr.ok td{background:#f0fff0}tr.bad td{background:#fff0f0}tr.ung td{background:#fffbe6}td.y{color:#1a7f37;font-weight:600}td.n{color:#c0392b;font-weight:600}.why{color:#666;font-size:11px}'
    '.fail{color:#c0392b}.loss{font-size:12px;padding:4px 8px;border-radius:4px;margin-bottom:6px}.loss.ok{background:#eaf7ea;color:#1a7f37}.loss.bad{background:#fdecec;color:#c0392b}'
    'h3{font-size:15px;margin:0 0 8px}h4{margin:8px 0 4px;font-size:13px}</style>'
)
_LEGEND = (
    '<h1>#253 표 E2E 리뷰 — 정의 · 색인 청크 · 질문/답변</h1>'
    '<p>기계 판정 = 정답 값 문자열이 답변에 있나 · LLM 재판정 = 기계 실패분만 Claude가 질문·정답·답변·청크를 보고 맞음/틀림/근거 없음/미답 판정 · 생성(최종) = 둘 중 하나라도 맞음 · 인용 = 근거로 댄 문서가 이 문서인가. 초록 = 최종 성공+인용 성공, 노랑 = 근거 없는 정답(청크에 값이 없는데 맞게 답함), 빨강 = 실패. 청크는 OpenSearch 조회 그대로, 답변은 /kms/query 실제 응답.</p>'
)


def _loose(t: str) -> str:
    return norm(t).replace(' ', '')


def chunk_loss(spec: dict, path: str, chunks: list[dict]) -> list[str]:
    """정의의 칸 글자 중 실제 색인 청크에 없는 것(띄어쓰기 무시) — '청킹이 누락 없이 됐는가'의 근거."""
    text = _loose(' '.join(c['text'] for c in chunks))
    cells = [cell_text(c) for t in expected_tables(spec, path) for row in t for c in row if c is not None]
    return [c for c in dict.fromkeys(cells) if _loose(c) and _loose(c) not in text]


def diagnose(row: dict, chunks: list[dict]) -> str:
    """최종 실패한 답의 원인 추정 — 답변 모양·인용·청크에 정답이 있는지로 가른다(LLM 판정 이유와 나란히 보여 준다)."""
    if row['final_ok']:
        return ''
    in_chunk = _loose(row['expected']) in _loose(' '.join(c['text'] for c in chunks))
    refused = ('확인할 수 없' in row['answer']) or row.get('llm_label') == 'not_answered'
    if refused and not row['cited']:
        return (
            '검색 단계에서 후보 0건(근거 없음 게이트) — 정답이 청크에 있는데 질문 어휘가 표 어휘와 달라 못 찾음'
            if in_chunk
            else '검색 후보 0건 + 정답이 청크에도 없음 — 파싱 단계 손실이 원인'
        )
    if row.get('llm_label') == 'ungrounded':
        return '청크에 근거가 없는데 답함 — 파싱 손실(OCR 기호·병합) 위에 모델이 추정'
    if not in_chunk:
        return '청크 자체에 정답이 없거나 틀림 — 파싱 손실(OCR 오독·캡션 오독·렌더 누락)'
    if row['cited'] and not row['cited_ok']:
        return '다른 문서를 근거로 답함 — 검색이 엉뚱한 문서를 올림'
    return '청크엔 정답이 있는데 답이 다름 — 생성 모델 오류'


def _esc(s: object) -> str:
    return str(s).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def _llm_cell(x: dict) -> str:
    if x.get('llm_label') is None:
        return '—' if x['value_ok'] else ('오류' if x.get('llm_error') else '—')
    return f"<b>{_LLM_NAME[x['llm_label']]}</b><br><span class=\"why\">{_esc(x.get('llm_reason') or '')}</span>"


def _qa_html(qa: list[dict], chunks: list[dict]) -> str:
    return ''.join(
        f"<tr class=\"{'ok' if x['final_ok'] and x['cited_ok'] else ('ung' if x.get('llm_label') == 'ungrounded' else 'bad')}\">"
        f"<td>{_esc(x['question'])}</td><td>{_esc(x['expected'])}</td><td>{_esc(x['answer'])}</td>"
        f"<td class=\"{'y' if x['value_ok'] else 'n'}\">{'성공' if x['value_ok'] else '실패'}</td>"
        f"<td>{_llm_cell(x)}</td>"
        f"<td class=\"{'y' if x['final_ok'] else 'n'}\">{'성공' if x['final_ok'] else '실패'}</td>"
        f"<td class=\"{'y' if x['cited_ok'] else 'n'}\">{'성공' if x['cited_ok'] else '실패'}</td>"
        f"<td class=\"why\">{_esc(diagnose(x, chunks))}</td></tr>"
        for x in qa
    )


def _loss_html(spec: dict, m: dict) -> str:
    lost = chunk_loss(spec, m['path'], m.get('chunks') or []) if m.get('status') == 'ready' else []
    if not lost:
        return '<div class="loss ok">청킹 누락 없음 — 정의의 모든 칸 글자가 청크에 있음(띄어쓰기 무시)</div>'
    return f'<div class="loss bad">청킹 누락 {len(lost)}칸: {_esc(", ".join(lost[:8]))}{" …" if len(lost) > 8 else ""}</div>'


def _card(out: Path, m: dict, spec: dict, qa: list[dict]) -> str:
    """문서(정의 × 경로) 하나의 카드 — 정의 이미지·기대 행 | 실제 청크, 그 아래 Q/A 표."""
    exp = expected_rows(spec, expected_tables(spec, m['path'])[0], m['path'])
    exp_md = '\n'.join('| ' + ' | '.join(r) + ' |' for r in exp)
    chunks_html = ''.join(
        f'<div class="chunk"><div class="hp">#{i} {_esc(" › ".join(c.get("heading_path") or []))}</div>'
        f'<pre>{_esc(c["text"])}</pre></div>'
        for i, c in enumerate(m.get('chunks') or [])
    )
    png = out / 'docs' / 'png' / f"{m['id']}.png"
    img = f'<img src="docs/png/{png.name}">' if png.exists() else ''
    failed = (
        ''
        if m.get('status') == 'ready'
        else f" <span class=\"fail\">[실패: {_esc(m.get('status_reason') or m.get('error') or m.get('status'))}]</span>"
    )
    return (
        f'<div class="c" id="{m["id"]}-{m["path"]}"><h3>[{axis_no(spec)}] [{_esc(axis_name(spec))}] [{PATH_LABEL[m["path"]]}]{failed}</h3>'
        f'<div class="row"><div class="col">{img}<h4>기대 행(설계 규칙)</h4><pre>{_esc(exp_md)}</pre></div>'
        f'<div class="col"><h4>실제 색인 청크 {len(m.get("chunks") or [])}개 (OpenSearch 조회)</h4>{_loss_html(spec, m)}{chunks_html}</div></div>'
        f'<table><tr><th>질문</th><th>정답</th><th>답변(14B)</th><th>기계 판정</th><th>LLM 재판정</th><th>생성(최종)</th><th>인용</th><th>실패 원인(추정)</th></tr>{_qa_html(qa, m.get("chunks") or [])}</table></div>'
    )


def _tally(a: dict[str, int], m: dict, qa: list[dict]) -> None:
    a['idx'] += 1
    a['ready'] += m.get('status') == 'ready'
    for x in qa:
        a['q'] += 1
        a['v'] += x['value_ok']
        a['f'] += x['final_ok']
        a['c'] += x['cited_ok']
        a['man'] += x['manual']
        a['ung'] += x.get('llm_label') == 'ungrounded'


def _summary(agg: dict[tuple[str, str], dict[str, int]], answers: list[dict]) -> str:
    axes = sorted({k[0] for k in agg})
    lines = ['| 축 | ' + ' | '.join(PATH_LABEL[p] for p in PATHS) + ' |', '|---|' + '---|' * len(PATHS)]
    for ax in axes:
        cells = []
        for p in PATHS:
            a = agg.get((ax, p))
            if not a:
                cells.append('—')
                continue
            cells.append(
                f"색인 {a['ready']}/{a['idx']} · 생성 {a['f']}/{a['q']}(기계 {a['v']}) · 인용 {a['c']}/{a['q']}"
                + (f" · 근거없음 {a['ung']}" if a['ung'] else '')
            )
        lines.append(f'| {ax} | ' + ' | '.join(cells) + ' |')
    tot = {k: sum(a[k] for a in agg.values()) for k in ('q', 'v', 'f', 'c', 'ung')}
    judges = sorted({a['judge'] for a in answers if a.get('judge')})
    return (
        '# #253 E2E 요약 — 실제 업로드·색인·질문\n\n'
        + '\n'.join(lines)
        + f"\n\n합계: 질문 {tot['q']} · 생성 성공(최종) {tot['f']} (기계 판정만 {tot['v']}) · 인용 성공 {tot['c']}"
        f" · 근거 없는 정답 {tot['ung']}\n재판정자: {', '.join(judges) or '없음'} (기계 판정 실패분만)\n"
    )


def phase_report(out: Path, manifest: list[dict], answers: list[dict]) -> None:
    by_doc: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for a in answers:
        a['value_ok'] = judge(a['answer'], a['expected'])  # 판정 규칙이 바뀌면 리뷰만 다시 만들어도 반영되게
        a['final_ok'] = final_ok(a)
        by_doc[(a['id'], a['path'])].append(a)
    agg: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: dict(q=0, v=0, f=0, c=0, man=0, idx=0, ready=0, ung=0)
    )
    cards = []
    for m in sorted(
        manifest, key=lambda x: (int(axis_no(_SPECS[x['id']])), x['id'], PATHS.index(x['path']))
    ):  # 축 번호 순(1→23), 경로 순
        spec = _SPECS[m['id']]
        qa = by_doc.get((m['id'], m['path']), [])
        _tally(agg[(spec['axis'], m['path'])], m, qa)
        cards.append(_card(out, m, spec, qa))
    (out / 'review.html').write_text('<meta charset="utf-8">' + _STYLE + _LEGEND + ''.join(cards))
    (out / 'summary.md').write_text(_summary(agg, answers))
