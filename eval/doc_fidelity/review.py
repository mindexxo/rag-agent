"""리뷰 HTML·사람 재판정·집계 (#247 결정 1·7).

재판정 대상(select_for_human): ALL_BELOW장 이하면 **전부**(결정 8). 그보다 많으면 Claude가 오류·위험·애매·판정 실패로 본 그림 **전부** +
정확·경미로 본 그림 중 **20%**(시드 고정).
정상 표본까지 사람이 보므로 '판정자가 놓친 비율'이 숫자로 나온다. 캡션이 색인과 다른 그림(in_index=False)도 전부 넣는다(결정 5).
사진·장식은 판정 제외라 표본에 넣지 않는다.

HTML 카드: 원본 크롭 | 캡션 | Claude 판정(개수·근거) | 사람 등급 선택·메모. 입력은 브라우저 localStorage에 임시 저장되고
'판정 내보내기'가 human.json을 내려받게 한다 — summarize가 그 파일을 읽는다(결정 7 = a).
"""
import html
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from eval.doc_fidelity.rubric import EXCLUDED_TYPE, GRADES

SAMPLE_RATE = 0.20
SEED = 247
FLAG_GRADES = {'오류', '위험'}


def is_flagged(r: dict) -> bool:
    v = r.get('verdict') or {}
    return (v.get('grade') in FLAG_GRADES or bool(v.get('ambiguous')) or bool(v.get('error'))
            or bool(v.get('grade_mismatch')) or r.get('in_index') is False)


ALL_BELOW = 60                # 이 장수 이하면 사람이 전부 본다(결정 8 — 실문서 29장, 20% 표본은 2~3장이라 의미가 없었다)


def select_for_human(records: list[dict]) -> list[dict]:
    if len(records) <= ALL_BELOW:            # 사진·장식도 넣는다 — 판정자의 '제외'가 맞는지 사람이 확인한다
        return [{**r, 'review_reason': '플래그' if is_flagged(r) else ('장식 확인' if (r.get('verdict') or {}).get('type') == EXCLUDED_TYPE
                                                                       else '전수')} for r in records]
    judged = [r for r in records if (r.get('verdict') or {}).get('type') != EXCLUDED_TYPE]
    flagged = [r for r in judged if is_flagged(r)]
    ok = [r for r in judged if not is_flagged(r)]
    k = round(len(ok) * SAMPLE_RATE)
    sample = random.Random(SEED).sample(ok, k) if k else []
    picked = {r['picture_id'] for r in flagged + sample}
    return [{**r, 'review_reason': '플래그' if r in flagged else '정상 표본'} for r in records if r['picture_id'] in picked]


def _chunk_block(v: dict) -> str:
    where = ' > '.join(v['heading_path']) + (f" · {v['page']}쪽" if v.get('page') else '') + (f" · {v['note']}" if v.get('note') else '')
    return f'<p class="small">{html.escape(where)}</p><pre class="chunk">{html.escape(v["text"])}</pre>'


def _card(r: dict, crops_rel: str) -> str:
    v = r.get('verdict') or {}
    counts = ' · '.join(f'{k} {v.get(k)}' for k in ('text_units', 'missing', 'misread', 'number_errors', 'relation_errors',
                                                    'hallucinations', 'chatter') if k in v)
    ev = ''.join(f'<li>{html.escape(e)}</li>' for e in v.get('evidence') or [])
    opts = ''.join(f'<option>{g}</option>' for g in ('', *GRADES, '제외(사진·장식)'))
    img = f'<img src="{crops_rel}/{html.escape(r["crop"])}">' if r.get('crop') else '<p>(크롭 없음)</p>'
    where = f'{html.escape(r["filename"])} · {r["fmt"].upper()}' + (f' · {r["page"]}쪽' if r.get('page') else '') + \
            (f' · 면적 {r["area_ratio"]:.1%}' if r.get('area_ratio') is not None else '') + (' · 표 칸 안' if r.get('in_table') else '')
    idx = '' if r.get('in_index') is not False else '<p class="warn">⚠️ 이 캡션이 색인 청크에 없다 — 색인 캡션은 아래 "가장 비슷한 청크"</p>'
    near = f'<details><summary>가장 비슷한 색인 청크</summary><pre>{html.escape(r.get("nearest_chunk", ""))}</pre></details>' \
        if r.get('nearest_chunk') else ''
    cap = html.escape(r.get('caption') or f'(캡션 없음 — {r.get("skip_reason")})')
    views = r.get('chunk_view')
    if views is None:
        chunks_html = ''
    elif not views:
        chunks_html = '<h4>실제 색인 청크</h4><p class="warn">⚠️ 이 그림의 캡션·자리표시가 든 청크를 찾지 못했다</p>'
    else:
        split = ' — 캡션이 청크 경계에서 나뉘었다' if len(views) > 1 and r.get('caption') else ''
        chunks_html = f'<h4>실제 색인 청크 ({len(views)}개{split})</h4>' + ''.join(_chunk_block(v) for v in views)
    return f'''<section class="card" data-id="{r['picture_id']}">
<h3>[{r['picture_id']}] {where} <span class="tag">{r.get('review_reason', '')}</span></h3>
<div class="row"><div class="img">{img}</div>
<div class="txt"><h4>VLM 캡션</h4><pre>{cap}</pre>{idx}{near}{chunks_html}
<h4>Claude 판정: {html.escape(str(v.get('type', '')))} → <b>{html.escape(str(v.get('grade') or v.get('error') or ''))}</b>{' (애매)' if v.get('ambiguous') else ''}</h4>
<p class="small">{counts}</p><ul>{ev}</ul>
<label>사람 등급 <select class="g">{opts}</select></label> <input class="m" placeholder="메모(틀린 곳)" size="60">
</div></div></section>'''


def write_html(records: list[dict], out: Path, crops_rel: str = 'crops', title: str = '그림 캡션 재판정') -> Path:
    cards = '\n'.join(_card(r, crops_rel) for r in records)
    page = f'''<!doctype html><html lang="ko"><head><meta charset="utf-8"><title>{title}</title>
<style>body{{font-family:-apple-system,sans-serif;margin:16px;background:#fafafa}} .card{{background:#fff;border:1px solid #ddd;
border-radius:8px;padding:12px;margin:12px 0}} .row{{display:flex;gap:16px;flex-wrap:wrap}} .img img{{max-width:520px;
border:1px solid #ccc}} .txt{{flex:1;min-width:320px}} pre{{white-space:pre-wrap;background:#f4f4f4;padding:8px;font-size:13px}}
.tag{{font-size:12px;background:#eef;padding:2px 6px;border-radius:4px}} .warn{{color:#b00}} .small{{color:#555;font-size:13px}}
pre.chunk{{background:#eef6ee;border-left:3px solid #5a5}} #bar{{position:sticky;top:0;background:#fff;padding:8px;border-bottom:1px solid #ccc}}</style></head><body>
<div id="bar"><b>{title}</b> — {len(records)}장 · <span id="done">0</span>장 입력 ·
<button onclick="exp()">판정 내보내기(human.json)</button></div>
<p>등급 기준(rubric.py): 위험=숫자 오류·환각 1개 이상 / 오류=누락+오독이 글자 덩어리의 10% 초과 또는 관계 오류 / 경미=10% 이하 /
정확=모두 0(잡담 허용). 캡션 없음 = 글자 전부 누락.</p>
{cards}
<script>
const KEY='doc_fidelity_human_v1';
function load(){{try{{return JSON.parse(localStorage.getItem(KEY)||'{{}}')}}catch(e){{return {{}}}}}}
function save(s){{try{{localStorage.setItem(KEY,JSON.stringify(s))}}catch(e){{}}}}
const st=load();
function count(){{document.getElementById('done').textContent=Object.values(st).filter(x=>x.grade).length}}
document.querySelectorAll('.card').forEach(c=>{{const id=c.dataset.id,g=c.querySelector('.g'),m=c.querySelector('.m');
 if(st[id]){{g.value=st[id].grade||'';m.value=st[id].memo||''}}
 const up=()=>{{st[id]={{grade:g.value,memo:m.value}};save(st);count()}}; g.onchange=up; m.oninput=up;}});
count();
function exp(){{const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(st,null,1)],
 {{type:'application/json'}}));a.download='human.json';a.click()}}
</script></body></html>'''
    out.write_text(page)
    return out


def summarize(records: list[dict], human: dict | None = None) -> dict:
    """그림 판정 집계. human = {picture_id: {grade, memo}} — 있으면 사람 등급이 최종(재판정한 그림만), 일치율도 계산."""
    human = {k: v for k, v in (human or {}).items() if v.get('grade')}
    rows = []
    for r in records:
        v = r.get('verdict') or {}
        final = human.get(r['picture_id'], {}).get('grade') or v.get('grade')
        if final == '제외(사진·장식)' or v.get('type') == EXCLUDED_TYPE and r['picture_id'] not in human:
            final = None
        rows.append({**r, 'final': final})
    judged = [r for r in rows if r['final']]
    # 판정 실패(사람도 안 본 그림)는 '제외'와 따로 센다 — 같은 칸에 넣으면 판정자 장애가 판정 수·오류율을 조용히 깎는다(리뷰 지적)
    errors = [r for r in rows if not r['final'] and (r.get('verdict') or {}).get('error') and r['picture_id'] not in human]
    by_type = defaultdict(Counter)
    for r in judged:
        by_type[(r.get('verdict') or {}).get('type', '?')][r['final']] += 1
    agree = [(r['verdict'].get('grade'), human[r['picture_id']]['grade']) for r in rows
             if r['picture_id'] in human and (r.get('verdict') or {}).get('grade')]
    n = len(judged) or 1
    return {
        'pictures': len(rows), 'excluded': sum(1 for r in rows if not r['final']) - len(errors), 'judge_errors': len(errors),
        'judged': len(judged), 'grades': dict(Counter(r['final'] for r in judged)),
        # 환각·숫자 오류는 최종 등급이 '위험'으로 남은 그림만 센다 — 사람이 판정자의 환각 지적을 뒤집으면 빠져야 한다
        'hallucination_rate': round(sum(1 for r in judged if r['final'] == '위험'
                                        and (r.get('verdict') or {}).get('hallucinations')) / n, 4),
        'number_error_rate': round(sum(1 for r in judged if r['final'] == '위험'
                                       and (r.get('verdict') or {}).get('number_errors')) / n, 4),
        'uncaptioned_with_text': sum(1 for r in judged if not r['captioned']),
        'in_index_mismatch': sum(1 for r in rows if r.get('in_index') is False),
        'by_type': {t: dict(c) for t, c in by_type.items()},
        'human_reviewed': len(human), 'judge_human_agree': round(sum(a == b for a, b in agree) / len(agree), 4) if agree else None,
        'judge_missed': sum(1 for a, b in agree if a in ('정확', '경미') and b in FLAG_GRADES),
    }


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


COVERAGE_LABELS = ('유실', '변형', '잡음')


def write_coverage_html(units: list[dict], chunks_by_doc: dict, out: Path) -> Path:
    """문장 포함률 확인 페이지(#231) — Claude가 '유실'·'판정 실패'로 본 원문 단위를 사람이 유실/변형/잡음으로 다시 가른다.
    카드: 원문 단위 | 가장 비슷한 색인 청크 | Claude 판정·메모 | 사람 선택. 내보내기 = human_coverage.json."""
    cards = []
    for i, u in enumerate(units):
        key = f"{u['document_id']}:{i}"
        near = chunks_by_doc.get(u['document_id'], [])
        chunk = near[u['best_chunk']]['text'][:1200] if u.get('best_chunk', -1) >= 0 and u['best_chunk'] < len(near) else '(없음)'
        opts = ''.join(f'<option>{x}</option>' for x in ('', *COVERAGE_LABELS))
        cards.append(f'''<section class="card" data-id="{key}"><h3>{html.escape(u["filename"])} · {u["fmt"].upper()} · {u["element"]}
{f'· {u["page"]}쪽' if u.get('page') else ''} <span class="tag">Claude: {html.escape(str(u.get("loss_label")))}</span></h3>
<div class="row"><div class="txt"><h4>원문 단위</h4><pre>{html.escape(u["text"])}</pre><p class="small">{html.escape(u.get("loss_note") or "")}</p>
<label>사람 판정 <select class="g">{opts}</select></label> <input class="m" placeholder="메모" size="50"></div>
<div class="txt"><h4>가장 비슷한 색인 청크 (겹침 {u.get("overlap")})</h4><pre>{html.escape(chunk)}</pre></div></div></section>''')
    page = write_html([], out, title='문장 포함률 — 유실 확인').read_text()
    page = page.replace("doc_fidelity_human_v1", "doc_fidelity_coverage_v1").replace("a.download='human.json'", "a.download='human_coverage.json'")
    page = page.replace('<script>', '\n'.join(cards) + '\n<script>', 1).replace(' — 0장 ·', f' — {len(units)}건 ·')
    page = page.replace('등급 기준(rubric.py)', '유실=정보가 색인에 없다 / 변형=표현만 다르고 정보는 있다 / 잡음=인쇄 머리글·바닥글·포털 제목 등.<br>(그림 등급 기준: rubric.py)')
    out.write_text(page)
    return out


def write_docview_html(docs: list[tuple[str, list[dict]]], captions: dict[str, list[str]], out: Path) -> Path:
    """문서 전체 청킹 보기(10/18 사용자 요청) — 그림·표가 앞뒤 문맥과 같은 청크로 묶였는지, 캡션만 떨어졌는지, 제목 경로가
    무엇인지를 한눈에 본다. 생성은 청크에 함께 실린 문맥에 좌우된다.
    docs = [(파일명, 청크들(OpenSearch 순서))], captions = {파일명: [그 문서 그림 캡션들]} — 캡션에서 온 줄은 파랗게,
    자리표시(캡션 대상에서 빠진 그림)는 빨갛게, 그림·산식·표 제목 줄('[그림 N]' 등)은 회색으로 칠한다."""
    import re as _re
    from eval.doc_fidelity.pictures import PLACEHOLDER, norm
    title_re = _re.compile(r'^\[(그림|산식|표)\s*\d+\]')
    sections = []
    for name, chunks in docs:
        cap_lines = {n for c in captions.get(name, []) for n in (norm(x) for x in c.split('\n')) if len(n) >= 2}
        blocks = []
        for i, c in enumerate(chunks):
            lines = []
            for line in c['text'].split('\n'):
                esc = html.escape(line)
                if PLACEHOLDER in line:
                    lines.append(f'<span class="ph">{esc} ← 캡션 대상에서 빠진 그림</span>')
                elif title_re.match(line.strip()):
                    lines.append(f'<span class="tt">{esc}</span>')
                elif line.strip() and norm(line) in cap_lines:
                    lines.append(f'<span class="cap">{esc}</span>')
                else:
                    lines.append(esc)
            where = ' > '.join(c.get('heading_path') or []) + (f" · {c['page']}쪽" if c.get('page') else '')
            blocks.append(f'<div class="chunk"><div class="hd">#{i} · {html.escape(where)} · {len(c["text"])}자</div>'
                          f'<pre>{"<br>".join(lines)}</pre></div>')
        sections.append(f'<h2>{html.escape(name)} — 청크 {len(chunks)}개</h2>' + ''.join(blocks))
    page = f'''<!doctype html><html lang="ko"><head><meta charset="utf-8"><title>문서 청킹 보기</title>
<style>body{{font-family:-apple-system,sans-serif;margin:16px;background:#fafafa;max-width:1100px}} .chunk{{background:#fff;border:1px solid #ddd;
border-radius:6px;margin:10px 0}} .hd{{background:#eef2f7;padding:6px 10px;font-size:13px;color:#335}} pre{{white-space:pre-wrap;margin:0;
padding:10px;font-size:13px}} .cap{{background:#e3efff}} .ph{{background:#ffe3e3;color:#a00}} .tt{{color:#777;font-style:italic}}</style></head><body>
<h1>문서 청킹 보기</h1><p>실제 색인 청크(OpenSearch 조회) 순서대로. <span class="cap">파란 줄</span>=그림 캡션(VLM)에서 온 글,
<span class="ph">빨간 줄</span>=캡션 없이 빠진 그림, <span class="tt">회색</span>=그림·산식·표 제목 줄. 머리글의 제목 경로가 그 청크가 속한 절이다.</p>
{''.join(sections)}</body></html>'''
    out.write_text(page)
    return out


def write_ab_html(items: list[dict], out: Path, key_out: Path, crops_rel: str = 'crops', seed: int = 247) -> Path:
    """프롬프트 A/B 블라인드 비교(#247 결정 15) — 그림마다 두 캡션의 좌우를 시드 고정으로 섞는다. 어느 쪽이 현행인지는
    key_out(JSON)에만 적고 페이지에는 쓰지 않는다. items = [{picture_id, crop, filename, caption_a_src, caption_b_src, base, cand}]
    선택: A가 낫다 / B가 낫다 / 비슷하다 + 메모 → ab_human.json 내려받기."""
    rng = random.Random(seed)
    key, cards = {}, []
    for it in items:
        flip = rng.random() < 0.5
        left, right = (it['cand'], it['base']) if flip else (it['base'], it['cand'])
        key[it['picture_id']] = {'A': 'cand' if flip else 'base', 'B': 'base' if flip else 'cand'}
        opts = ''.join(f'<option>{x}</option>' for x in ('', 'A가 낫다', 'B가 낫다', '비슷하다'))
        img = f'<img src="{crops_rel}/{html.escape(it["crop"])}">'
        cards.append(f'''<section class="card" data-id="{it['picture_id']}"><h3>[{it['picture_id']}] {html.escape(it['filename'])}</h3>
<div class="row"><div class="img">{img}</div></div>
<div class="row"><div class="txt"><h4>A</h4><pre>{html.escape(left)}</pre></div><div class="txt"><h4>B</h4><pre>{html.escape(right)}</pre></div></div>
<label>어느 캡션이 그림을 더 정확히 옮겼나 <select class="g">{opts}</select></label> <input class="m" placeholder="메모(틀린 곳)" size="60"></section>''')
    key_out.write_text(json.dumps(key, ensure_ascii=False, indent=1))
    page = write_html([], out, title='캡션 프롬프트 A/B 블라인드 비교').read_text()
    tag = out.parent.name
    page = page.replace('doc_fidelity_human_v1', f'doc_fidelity_ab_{tag}_v1').replace("a.download='human.json'", f"a.download='{tag}_ab_human.json'")
    page = page.replace('<script>', '\n'.join(cards) + '\n<script>', 1).replace(' — 0장 ·', f' — {len(items)}장 ·')
    page = page.replace('등급 기준(rubric.py)', '그림을 보고, 글자 누락·오독·숫자·관계(화살표·항목-값 짝)·지어낸 내용이 더 적은 쪽을 고른다. '
                        '어느 쪽이 현행 프롬프트인지는 숨겨져 있다.<br>(참고: 등급 기준 rubric.py)')
    out.write_text(page)
    return out
