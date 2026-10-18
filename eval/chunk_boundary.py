"""청크 경계 진단 — **경계 분류와 "맥락 손실"의 정의점** (#245).

#245 측정의 결론: 경계는 **글자를 잃지 않는다**(분할기 오버랩·표 행 단위 묶음 — 소실은 eval/chunk_invariants가 정의한다).
잃는 것은 **맥락**이다 — 뒤 조각만 근거로 받으면 무엇에 대한 사실인지 모른다(합성 13문항 중 6문항이 사실 조각만으로 모호·오답·거절,
eval/report_chunk_boundary.md). 그래서 두 가지를 센다.

1. 경계 분류(classify_boundary) — 같은 섹션(제목 경로가 같음) 안 이웃 조각 사이, 즉 분할기·표 묶음이 만든 경계가 어디서 끊겼나.
2. 맥락 손실(context_losses) — 뒤 조각에 없어서 뒤 조각만으로는 뜻이 안 서는 앞 글:
   - list_head  : 목록 머리('다음 각 호의 … 한다')가 앞 조각에만 — 뒤 항목은 귀결(조치)을 잃는다(T6)
   - list_label : 목록의 소제목 줄('[상황 6] …'·'■ …')이 앞 조각에만 — 뒤 항목은 어느 상황의 단계인지 잃는다.
                  TXT는 헤딩이 없어 제목 경로도 이를 메우지 못한다
   - reference  : 뒤 조각이 앞 글을 가리키는 말('이 경우'·'단,'·'다만'…)로 시작한다 — 가리키는 대상이 앞 조각에만(T5)
   - table_lead : 표 앞 설명 줄(최대 3줄)이 어느 묶음에 없다(B1 여러 줄 중 마지막만 붙음·B2 150자 넘어 안 붙음)
   - table_note : 표 뒤 단위·주석 줄('※ …')이 어느 묶음에 없다(B3 — 마지막 묶음이나 표 밖에만)
   - table_header: 뒤 묶음 첫 표 줄이 첫 묶음의 머리 행과 다르다 — MD/TXT 표는 글로 분할기를 타 머리 행을 반복하지 않는다(B6)

판정은 규칙(정규식)이라 후보다 — 리포트의 수치는 "후보 건수"이고 사람이 예시를 읽어 확정했다. 입력은 조각 dict
{'heading_path': list, 'text': str}의 문서 순 목록 하나(한 문서). 색인(eval.axis_common.load_chunks)과 chunk_file 둘 다 받는다.

사용:
    python -m eval.chunk_boundary                       # 개발계 색인 전 테넌트(V2_TENANTS) — OpenSearch 필요
    python -m eval.chunk_boundary --files sample_docs/corpus_v2/goodpeople/goodpeople_18_해외봉사단복무지침.txt
결과 JSON은 eval/results/chunk_boundary_<시각>/ 에 남긴다(gitignore — 실문서 원문 포함). 화면에는 건수와 파일명만.
"""
import argparse
import json
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.chunk_invariants import _stitch, norm   # noqa: E402 — 비교 정규화·이어 붙이기의 정의점
from rag.identifiers import is_heading_line   # noqa: E402 — 조 제목 판정의 정의점(#233)

# 한국어 문장 끝 — 종결 어미(+마침표) 또는 문장 부호, 닫는 따옴표·괄호 허용
SENTENCE_END = re.compile(r'(다|요|음|함|됨|임|것|니다|까|죠)[.!?]?["”’)\]]*\s*$|[.!?。]["”’)\]]*\s*$')
# 번호 뒤에 숫자가 오면 항목이 아니다('3.5kg'·'2026.10.18'·'1.1 소제목'). 한글 번호는 가나다 열거만 — 범위 [가-하]는 한글 음절 전부다
ITEM = re.compile(r'^\s*(\d+[.)](?!\d)|\(\d+\)|[①-⑳]|[가나다라마바사아자차카타파하][.)]|[-•·]\s)')
LABEL = re.compile(r'^\s*(\[[^\]]{1,40}\]|[■□◆●▶]|\d+(\.\d+)+\s|<[^>]{1,30}>)')
LIST_HEAD = re.compile(r'다음 각 호|다음과 같[다이]|아래와 같[다이]|[:：]\s*$')
NOTE = re.compile(r'^\s*(※|\*\s|주\)|\(?단위)')
REFERENCE = re.compile(r'^\s*(이 경우|이때|이 때|단,|다만|그러나|위의|앞의)')   # '해당 …'은 대상을 새로 지시하는 독립 문장이 많아 뺐다
_MIN_OVERLAP = 5   # 이보다 짧은 일치는 꼬리·머리 글자가 우연히 같은 것(진짜 잘림을 가린다). 오버랩은 마지막 분할 조각을 통째로 싣는 거라 짧을 수 있다 — 실문서에 7자(쉼표 조각 하나)가 있었다(10/18)


def _lines(text: str) -> list[str]:
    return [l for l in text.split('\n') if l.strip()]


def _is_row(line: str) -> bool:
    return line.lstrip().startswith('|')


def _is_row_part(line: str) -> bool:
    """표 행이거나 분할기가 행 중간을 잘라 남긴 행 조각('| AQ-5012 | … | 90.' / '1 | 14 | 표본 검사 |') — MD·TXT 표(B6)."""
    return _is_row(line) or line.rstrip().endswith('|')


def overlap_chars(a: str, b: str, cap: int = 400) -> int:
    """앞 조각 꼬리 = 뒤 조각 머리인 가장 긴 길이(정규화 글자) — 분할기 오버랩의 실제 크기."""
    a, b = norm(a), norm(b)
    return next((k for k in range(min(len(a), len(b), cap), 0, -1) if a.endswith(b[:k])), 0)


def classify_boundary(a: str, b: str) -> str:
    """같은 섹션 이웃 조각 경계의 종류. 앞에서부터 처음 맞는 것:
    table_mid_row(표 행 한가운데 — MD·TXT 표를 분할기가 자름) / table_row(표 행 사이) / table_text(표↔글) / sentence(앞 조각이 문장 끝) / next_item(뒤 조각이 항목·소제목·조 제목으로 시작) /
    mid_overlap(문장 중간이지만 오버랩이 끊긴 문장을 뒤 조각에 다시 실음) / mid_cut(문장 중간·오버랩 없음 — 진짜 잘림)."""
    la, lb = _lines(a), _lines(b)
    if not la or not lb:
        return 'sentence'
    tail, head = la[-1], lb[0]
    if _is_row(tail) and not tail.rstrip().endswith('|') and _is_row_part(head) and not _is_row(head):
        return 'table_mid_row'
    if _is_row(tail) and _is_row(head):
        return 'table_row'
    if _is_row(tail) or _is_row(head):
        return 'table_text'
    if SENTENCE_END.search(tail):
        return 'sentence'
    # PDF는 짧은 조끼리 한 섹션으로 묶여(제목 경로가 같다) 뒤 조각이 '제 36 조(휴일)'로 시작할 수 있다 — 앞 항목이 마침표 없이
    # 끝나도 잘림이 아니다(실문서 후보 5건 중 4건이 이 꼴, 10/18)
    if ITEM.match(head) or LABEL.match(head) or is_heading_line(head):
        return 'next_item'
    # 오버랩(앞 조각 꼬리를 뒤 조각 머리에 다시 실음)이 있으면 끊긴 문장의 앞부분이 뒤 조각에도 있다
    return 'mid_overlap' if overlap_chars(a, b) >= _MIN_OVERLAP else 'mid_cut'


def _list_context(prev_lines: list[str]) -> str | None:
    """목록 항목 줄들 위로 올라가 처음 만나는 항목 아닌 줄 — 그 목록의 머리나 소제목."""
    for line in reversed(prev_lines):
        if LABEL.match(line) or not ITEM.match(line) and not _is_row(line):
            return line
    return None


def context_losses(chunks: list[dict], tables: bool = True) -> list[dict]:
    """한 문서의 조각 목록(문서 순)에서 맥락 손실 후보. 각 항목 {'kind', 'at'(뒤 조각 순번), 'line'(잃은 줄)}."""
    out = []
    for i in range(1, len(chunks)):
        a, b = chunks[i - 1], chunks[i]
        if a['heading_path'] != b['heading_path']:
            continue
        lb = _lines(b['text'])
        if not lb:
            continue
        nb = norm(b['text'])
        if REFERENCE.match(lb[0]):
            out.append({'kind': 'reference', 'at': i, 'line': lb[0]})
        # 목록 — 뒤 조각이 항목으로 시작하고 앞 조각이 항목으로 끝나거나 목록 머리로 끝난다
        if ITEM.match(lb[0]):
            section = [l for c in chunks[:i] if c['heading_path'] == b['heading_path'] for l in _lines(c['text'])]
            ctx = _list_context(section)
            if ctx and norm(ctx) not in nb and (ITEM.match(section[-1]) or ctx == section[-1]):
                kind = 'list_head' if LIST_HEAD.search(ctx) else 'list_label' if LABEL.match(ctx) else None
                if kind:
                    out.append({'kind': kind, 'at': i, 'line': ctx})
    if tables:
        out.extend(_table_losses(chunks))
    return sorted(out, key=lambda x: x['at'])


def _has_own_header(lines: list[str]) -> bool:
    """머리 행 + 구분 행('| --- |')으로 시작 — 제 머리 행을 가진 새 표다(MD 뒤 조각은 구분 행이 없다)."""
    return len(lines) > 1 and _is_row(lines[0]) and set(lines[1].replace('|', '').split()) <= {'---'}


def _table_runs(chunks: list[dict]) -> list[list[int]]:
    """한 표가 여러 조각에 걸친 무리 — 같은 섹션에서 표 행으로 끝나는 조각 다음에, 그 표의 머리 행을 다시 단 묶음(PDF·DOCX 행 묶음)
    이나 표 행으로 바로 시작하는 조각(MD·TXT 표를 분할기가 자른 뒤 조각)이 이어진다. 머리 행이 같은지만으로 묶으면 MD 표의 뒤
    조각(머리 행 없음)이 빠지고, 표 행이 든 조각이면 다 묶으면 이웃한 다른 표끼리 묶인다(실문서 부칙·별지 오탐, 10/18). 그래서
    다른 머리 행 + 구분 행으로 시작하는 조각은 새 표로 본다."""
    runs, cur, header = [], [], None
    for i, c in enumerate(chunks):
        lines = _lines(c['text'])
        rows = [l for l in lines if _is_row(l)]
        prev = chunks[cur[-1]] if cur else None
        if prev and rows and prev['heading_path'] == c['heading_path'] and _is_row_part(_lines(prev['text'])[-1]) \
                and (rows[0] == header or _is_row_part(lines[0]) and not _has_own_header(lines)):
            cur.append(i)
            continue
        if len(cur) >= 2:
            runs.append(cur)
        cur, header = ([i], rows[0]) if rows else ([], None)
    if len(cur) >= 2:
        runs.append(cur)
    return runs


def _table_losses(chunks: list[dict]) -> list[dict]:
    """XLSX는 보지 않는다 — 시트 조각은 근거 조립에서 시트 완성본으로 바뀌어(rag.retriever.prepare_evidence) 묶음 경계가 모델에
    닿지 않는다. 호출부(census)가 확장자로 거른다."""
    out = []
    for run in _table_runs(chunks):
        first = _lines(chunks[run[0]]['text'])
        k = next(j for j, l in enumerate(first) if _is_row(l))
        # 표 앞 설명 — 첫 묶음의 표 위 줄 + 같은 섹션 앞 조각이 글로 끝나면 그 꼬리(설명이 앞 조각에 따로 남는 B1 모양)
        lead = first[:k]
        prev = chunks[run[0] - 1] if run[0] > 0 else None
        if prev and prev['heading_path'] == chunks[run[0]]['heading_path']:
            pl = _lines(prev['text'])
            tail = pl[max((j for j, l in enumerate(pl) if _is_row(l)), default=-1) + 1:]
            lead = tail + [l for l in lead if l not in tail]
        lead = lead[-3:]
        header = first[k]
        last = _lines(chunks[run[-1]]['text'])
        notes = [l for l in last[max(j for j, l in enumerate(last) if _is_row(l)) + 1:] if NOTE.match(l)]
        if run[-1] + 1 < len(chunks) and chunks[run[-1] + 1]['heading_path'] == chunks[run[-1]]['heading_path']:
            notes += [l for l in _lines(chunks[run[-1] + 1]['text'])[:2] if NOTE.match(l)]
        for i in run:
            n = norm(chunks[i]['text'])
            out += [{'kind': 'table_lead', 'at': i, 'line': l} for l in lead if norm(l) not in n]
            out += [{'kind': 'table_note', 'at': i, 'line': l} for l in notes if norm(l) not in n]
            if next(l for l in _lines(chunks[i]['text']) if _is_row(l)) != header:
                out.append({'kind': 'table_header', 'at': i, 'line': header})
    return out


def census(docs: dict[str, list[dict]]) -> dict:
    """문서별 조각 목록 → 경계 분류 건수·맥락 손실 후보(예시 포함). XLSX는 둘 다 세지 않는다(_table_losses docstring과 같은 이유). 맥락 손실은 (문서, 종류, 잃은 줄) 하나를 한 건으로 센다 —
    표 하나의 설명 한 줄이 묶음 다섯에서 빠져도 한 건이다(묶음 수만큼 세면 큰 표 하나가 수치를 지배한다)."""
    bounds, losses, examples = Counter(), Counter(), []
    for name, chunks in docs.items():
        sheet = name.lower().endswith('.xlsx')
        for a, b in zip(chunks, chunks[1:]):
            if sheet:
                break
            if a['heading_path'] == b['heading_path']:
                kind = classify_boundary(a['text'], b['text'])
                bounds[kind] += 1
                if kind == 'mid_cut':
                    examples.append({'file': name, 'kind': 'mid_cut', 'line': _lines(a['text'])[-1][-80:],
                                     'overlap': overlap_chars(a['text'], b['text'])})
        seen = set()
        for x in context_losses(chunks, tables=not sheet):
            if (x['kind'], x['line']) in seen:
                continue
            seen.add((x['kind'], x['line']))
            losses[x['kind']] += 1
            examples.append({'file': name, **x, 'line': x['line'][:80]})
    return {'boundaries': dict(bounds), 'context_losses': dict(losses), 'examples': examples}


def gold_spans(docs: dict[str, list[dict]], gold: list[dict]) -> dict:
    """골드 정답 스니펫이 조각 하나 안에 있나 / 이웃 조각을 이어야만 있나(경계에 걸림) / 아예 없나. docs 키는 '<테넌트>/<파일명>'."""
    from eval.generation import row_tenant
    out, spans = Counter(), []
    for g in gold:
        for e in g.get('expected_chunks') or []:
            sn = norm(e['snippet'])
            chunks = docs.get(f"{row_tenant(g)}/{e['filename']}")
            if not sn or chunks is None:
                continue
            texts = [norm(c['text']) for c in chunks]
            kind = 'inside' if any(sn in t for t in texts) else 'spans' if sn in _stitch(texts) else 'missing'
            out[kind] += 1
            if kind == 'spans':
                spans.append(g['id'])
    return {**out, 'span_ids': spans}


async def _from_index() -> dict[str, list[dict]]:
    from eval.axis_common import load_chunks
    from eval.generation import V2_TENANTS
    from rag.os_client import close_client
    docs = {}
    try:
        for t in sorted(V2_TENANTS):
            for c in sorted(await load_chunks(t), key=lambda c: c['chunk_id']):
                docs.setdefault(f"{t}/{c['filename']}", []).append(c)
    finally:
        await close_client()
    return docs


def _from_files(paths: list[str]) -> dict[str, list[dict]]:
    from rag.chunking import chunk_file
    files = sorted({p for t in paths for p in ([Path(t)] if Path(t).is_file() else Path(t).rglob('*.*'))})
    return {str(p): [{'heading_path': c.heading_path, 'text': c.text} for c in chunk_file(p)] for p in files}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--files', nargs='+', metavar='PATH', help='색인 대신 chunk_file로 조각을 만들 파일·디렉터리')
    args = ap.parse_args()
    if args.files:
        docs = _from_files(args.files)
        res = census(docs)
    else:
        import asyncio
        from eval.generation import load_gold
        docs = asyncio.run(_from_index())
        res = census(docs)
        res['gold_snippets'] = gold_spans(docs, load_gold())
    from eval.axis_common import commit
    res['commit'], res['docs'] = commit(), len(docs)
    out = Path(__file__).resolve().parent / 'results' / f"chunk_boundary_{datetime.now():%Y%m%d_%H%M}"
    out.mkdir(parents=True, exist_ok=True)
    (out / 'census.json').write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f"문서 {res['docs']} · 커밋 {res['commit']}")
    print('경계 분류:', res['boundaries'])
    print('맥락 손실 후보:', res['context_losses'])
    if 'gold_snippets' in res:
        print('골드 스니펫:', {k: v for k, v in res['gold_snippets'].items() if k != 'span_ids'}, res['gold_snippets']['span_ids'][:10])
    print('파일별:', dict(Counter((x['file'].split('/')[-1], x['kind']) for x in res['examples'])))
    print(f'예시 → {out}/census.json')


if __name__ == '__main__':
    main()
