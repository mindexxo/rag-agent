"""청크 머리말이 잘린 맥락을 메우는가 — 머리말 없음 / 현행 머리말 / Contextual Retrieval / 규칙 맥락 줄 비교(#234).

머리말(rag/search/index_text.build_index_text — '파일명 > 헤딩')은 임베딩·어휘 필드(Nori·bigram)·리랭커 세 곳에 같은 모양으로 들어간다.
조건마다 그 세 곳의 텍스트만 바꾼 **임시 인덱스**(kms_chunks_exp234_<조건>)를 만들고, 운영 검색(retrieve_candidates)을 인덱스 이름만
바꿔 그대로 돌린다 — 측정용 재구현이 틀릴 위험(#128)을 없애려고(#234 결정 3). 리랭커 입력은 같은 변형 텍스트로 바꿔 끼운다.

조건(#234 결정 2)
  none   : 본문만
  header : 현행 — 파일명 > 헤딩 + 본문
  cr     : LLM이 쓴 맥락 문장 + 현행 머리말 + 본문. 원 방법(Anthropic Contextual Retrieval)대로 문서와 조각을 주고 "검색이 잘 되게
           이 조각의 맥락을 짧게" 쓰게 한다. 생성기는 운영 vLLM(결정 4) — 문서가 길면 그 조각 앞뒤 CR_WINDOW_TOKENS만 준다
  rule   : 현행 머리말 + 잘린 맥락 줄(eval/chunk_boundary.context_losses의 목록 머리·소제목, '단,' 등은 앞 조각 마지막 줄) + 본문
           — #266 a안을 측정용으로만 흉내 낸 것
측정 테넌트(V2_TENANTS)의 문서 조각만 바꾸고, 다른 테넌트·FAQ 조각은 그대로 복사한다 — BM25 IDF가 인덱스 전체로 정해지므로
모집단을 운영과 같게 둔다. 임시 인덱스는 `drop`으로 지운다(운영 인덱스는 읽기만 한다).

채점(결정 5): 검색 — 전체 골드 검색축(eval/retrieval_v2.compute) 그대로, 맥락 셋(id에 _cx·_cb0)은 따로 묶는다. 답변 — 맥락 셋만,
상위 5 근거로 운영 생성 → EPCov(규칙+심판), 회차를 나눠 돈다(TEI·생성 흔들림).

사용:
    python -m eval.contextual_header contexts           # CR 맥락 문장 생성(이어 돌리기 가능) → eval/results/ctx234/cr_contexts.json
    python -m eval.contextual_header build              # 임시 인덱스 4개
    python -m eval.contextual_header measure [--runs 2] # 검색 + 답변 → eval/results/ctx234/measure_<시각>.json
    python -m eval.contextual_header drop               # 임시 인덱스 삭제
결과(eval/results/ctx234/)는 gitignore — 실문서 원문·생성 맥락 문장이 들어 있다.
"""

import argparse
import asyncio
import json
import sys
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings  # noqa: E402
from eval.chunk_boundary import context_losses  # noqa: E402
from eval.generation import V2_TENANTS  # noqa: E402
from rag.search.index_text import build_index_text  # noqa: E402
from rag.search.os_client import LEX_BIGRAM_FIELD, MAPPING, NORI_FIELD, client  # noqa: E402

CONDITIONS = ('none', 'header', 'cr', 'rule')
OUT = Path(__file__).resolve().parent / 'results' / 'ctx234'
CR_FILE = OUT / 'cr_contexts.json'
CR_WINDOW_TOKENS = 20000  # 문서가 이보다 길면 조각 앞뒤만 — vLLM 컨텍스트 30K에서 지시문·출력 몫을 뺀 값
CR_CONCURRENCY = 8  # 사내 vLLM 공유 장비 — 생성 평가(eval/generation)보다 보수적으로
CR_PROMPT = """<문서>
{doc}
</문서>
다음은 위 문서 안의 한 조각이다.
<조각>
{chunk}
</조각>
검색이 잘 되도록, 이 조각이 문서 전체에서 어떤 맥락에 있는지 짧고 간결하게(한두 문장) 설명하라. 설명만 답하고 다른 말은 하지 마라."""


_BASE_INDEX = (
    settings.opensearch_index
)  # 운영 인덱스 이름 — condition()이 settings 값을 바꿔 끼우므로 읽을 때 한 번 고정한다


def index_name(cond: str) -> str:
    return f'{_BASE_INDEX}_exp234_{cond}'


def is_context_row(gold_id: str) -> bool:
    return '_cx' in gold_id or '_cb0' in gold_id


# ── 원본 조각 읽기 ────────────────────────────────────────────────────────────


async def _all_docs(index: str) -> list[dict]:
    resp = await client().search(
        index=index, body={'size': 10000, 'track_total_hits': True, 'query': {'match_all': {}}}
    )
    if resp['hits']['total']['value'] > 10000:
        raise RuntimeError('조각 10,000개 초과 — scroll 필요')
    return [{'_id': h['_id'], **h['_source']} for h in resp['hits']['hits']]


def _target(doc: dict) -> bool:
    """조건마다 텍스트를 바꿀 조각 — 측정 테넌트의 문서 조각(FAQ 제외)."""
    return doc.get('tenant_id') in V2_TENANTS and doc.get('document_id') is not None


def _by_document(docs: list[dict]) -> dict[int, list[dict]]:
    out = defaultdict(list)
    for d in docs:
        if _target(d):
            out[d['document_id']].append(d)
    return {k: sorted(v, key=lambda d: d['chunk_id']) for k, v in out.items()}


# ── 조건별 텍스트 ─────────────────────────────────────────────────────────────


def rule_lines(chunks: list[dict]) -> dict[int, str]:
    """chunk_id → 잃은 맥락 줄. 목록 머리·소제목은 그 줄, '단,'·'이 경우'는 앞 조각의 마지막 글줄(가리키는 대상).
    표 손실(table_*)은 넣지 않는다 — #257이 표 전체를 근거로 넘긴다."""
    out = {}
    for x in context_losses([{'heading_path': c['heading_path'], 'text': c['text']} for c in chunks]):
        cid = chunks[x['at']]['chunk_id']
        if x['kind'] in ('list_head', 'list_label'):
            out[cid] = x['line']
        elif x['kind'] == 'reference':
            prev = [line for line in chunks[x['at'] - 1]['text'].split('\n') if line.strip()]
            if prev:
                out[cid] = prev[-1]
    return out


def variant_text(cond: str, doc: dict, cr: dict[str, str], rule: dict[int, str]) -> str:
    """임베딩·어휘 필드가 볼 텍스트(폴더 없음 — 운영 lex_text와 같은 자리)."""
    text, fn, hp = doc['text'], doc['filename'], list(doc.get('heading_path') or [])
    if cond == 'none':
        return text
    header = build_index_text(text, fn, hp)
    if cond == 'header':
        return header
    if cond == 'cr':
        ctx = cr.get(str(doc['chunk_id']), '').strip()
        return f'{ctx}\n{header}' if ctx else header
    line = rule.get(doc['chunk_id'])
    if not line:
        return header
    if header == text:  # 머리말이 없는 조각(파일명·헤딩이 본문 첫 줄과 같음)
        return f'{line}\n{text}'
    head, body = header.split('\n', 1)
    return f'{head}\n{line}\n{body}'


# ── ① CR 맥락 문장 생성 ───────────────────────────────────────────────────────


def _window(chunks: list[dict], i: int) -> str:
    from rag.tokens import estimate_tokens_conservative

    texts = [c['text'] for c in chunks]
    if estimate_tokens_conservative('\n'.join(texts)) <= CR_WINDOW_TOKENS:
        return '\n'.join(texts)
    lo = hi = i
    while True:  # 앞뒤로 한 조각씩 넓힌다
        grown = False
        for j in (lo - 1, hi + 1):
            if 0 <= j < len(texts):
                cand = texts[min(lo, j) : max(hi, j) + 1]
                if estimate_tokens_conservative('\n'.join(cand)) <= CR_WINDOW_TOKENS:
                    lo, hi, grown = min(lo, j), max(hi, j), True
        if not grown:
            return '\n'.join(texts[lo : hi + 1])


async def make_contexts() -> None:
    from rag.llm import create_llm

    settings.llm_temperature = 0.0  # 결정적 — 재생성해도 같은 문장(운영 생성 온도와 별개)
    OUT.mkdir(parents=True, exist_ok=True)
    done = json.loads(CR_FILE.read_text(encoding='utf-8')) if CR_FILE.exists() else {}
    docs = _by_document(await _all_docs(settings.opensearch_index))
    llm, sem = create_llm(), asyncio.Semaphore(CR_CONCURRENCY)
    jobs = [
        (chunks, i) for chunks in docs.values() for i in range(len(chunks)) if str(chunks[i]['chunk_id']) not in done
    ]
    print(f'조각 {sum(len(v) for v in docs.values())} · 이미 생성 {len(done)} · 남음 {len(jobs)}', flush=True)

    async def one(chunks, i):
        prompt = CR_PROMPT.format(doc=_window(chunks, i), chunk=chunks[i]['text'])
        async with sem:
            text = await llm.acomplete([{'role': 'user', 'content': prompt}])
        return str(chunks[i]['chunk_id']), ' '.join(text.split())

    for k in range(0, len(jobs), 64):  # 64개마다 저장 — 끊겨도 이어 돈다
        for cid, ctx in await asyncio.gather(*(one(c, i) for c, i in jobs[k : k + 64])):
            done[cid] = ctx
        CR_FILE.write_text(json.dumps(done, ensure_ascii=False, indent=0), encoding='utf-8')
        print(f'  {min(k + 64, len(jobs))}/{len(jobs)}', flush=True)


# ── ② 임시 인덱스 ─────────────────────────────────────────────────────────────


async def build() -> None:
    from opensearchpy.helpers import async_bulk

    from rag.search.embeddings import embed_texts
    from rag.search.lexical import bigrams

    cr = json.loads(CR_FILE.read_text(encoding='utf-8'))
    docs = await _all_docs(settings.opensearch_index)
    by_doc = _by_document(docs)
    rule = {cid: line for chunks in by_doc.values() for cid, line in rule_lines(chunks).items()}
    targets = [d for d in docs if _target(d)]
    missing = [d['chunk_id'] for d in targets if str(d['chunk_id']) not in cr]
    if missing:
        raise SystemExit(f'CR 맥락 없는 조각 {len(missing)}개 — contexts를 먼저 끝낼 것(예: {missing[:3]})')
    print(f'전체 {len(docs)} · 바꿀 조각 {len(targets)} · 규칙 맥락 줄 {len(rule)}', flush=True)
    for cond in CONDITIONS:
        name = index_name(cond)
        if await client().indices.exists(name):
            await client().indices.delete(name)
        await client().indices.create(name, body=MAPPING)
        texts = [variant_text(cond, d, cr, rule) for d in targets]
        vectors = await embed_texts(texts)
        changed = {d['_id']: (t, v) for d, t, v in zip(targets, texts, vectors)}
        actions = []
        for d in docs:
            src = {k: v for k, v in d.items() if k != '_id'}
            if d['_id'] in changed:
                t, v = changed[d['_id']]
                src.update({'dense': list(v.dense), NORI_FIELD: t, LEX_BIGRAM_FIELD: ' '.join(bigrams(t))})
            actions.append({'_index': name, '_id': d['_id'], '_source': src})
        await async_bulk(client(), actions, refresh=True)
        print(f'  {name}: {(await client().count(index=name))["count"]}건', flush=True)


async def drop() -> None:
    for cond in CONDITIONS:
        name = index_name(cond)
        if await client().indices.exists(name):
            await client().indices.delete(name)
            print('삭제', name)


# ── ③ 측정 ────────────────────────────────────────────────────────────────────


@contextmanager
def condition(cond: str, rerank_texts: dict[int, str]):
    """검색이 이 조건의 임시 인덱스를 보고, 리랭커가 이 조건의 텍스트를 보게 한다(운영 코드는 그대로)."""
    import rag.search.reranker as rr

    orig_index, orig_rerank = settings.opensearch_index, rr._rerank_text
    target_docs = {
        cid >> 20 for cid in rerank_texts
    }  # 문서 조각 id = (문서 id << 20) | 순번(rag/search/os_index.chunk_os_id)

    def rerank_text(chunk):
        v = rerank_texts.get(chunk.chunk_id)
        if v is None:
            if chunk.faq_id is None and getattr(chunk, 'document_id', None) in target_docs:
                raise RuntimeError(f'변형 텍스트 없는 측정 조각 {chunk.chunk_id} — 스냅숏과 어긋났다')
            return orig_rerank(chunk)
        folder = ' — '.join(x for x in (chunk.folder_name, chunk.folder_description) if x)
        if not folder:
            return v
        # 근사 — 운영 build_index_text(folder=)는 폴더를 머리말 조각 목록 맨 앞에 넣고 ' > '로 잇는다. 여기는 완성된 텍스트 앞에
        # 붙여, 머리말 조각이 없는 header·cr 조각에서 모양이 조금 다르다(폴더 있는 테넌트만, #234 리뷰 Minor)
        return f'{folder}\n{v}' if cond == 'none' else f'{folder} > {v}'

    settings.opensearch_index, rr._rerank_text = index_name(cond), rerank_text
    try:
        yield
    finally:
        settings.opensearch_index, rr._rerank_text = orig_index, orig_rerank


async def _answers(cond: str, rows: list[dict], gold: dict, run: int) -> list[dict]:
    """맥락 셋 문항 — 운영 retrieve(근거 준비 포함) → 운영 생성 → EPCov."""
    from database import AsyncSessionLocal
    from eval import epcov_judge
    from eval.generation import _epcov_scores, expected_points_hits, row_tenant
    from rag.llm import create_llm
    from rag.search.retriever import retrieve
    from rag.turn.prompts import build_knowledge_generation_prompt

    llm, judge, sem, jsem = create_llm(), epcov_judge.judge_llm(), asyncio.Semaphore(8), asyncio.Semaphore(4)
    out = []
    async with AsyncSessionLocal() as session:
        evid = {}
        for r in rows:  # 세션은 직렬(AsyncSession 동시 실행 불가)
            g = gold[r['id']]
            evid[g['id']] = (await retrieve(session, row_tenant(g), g['query'])).chunks

    async def one(gid):
        g = gold[gid]
        async with sem:
            ans = await llm.acomplete(
                build_knowledge_generation_prompt(g['query'], evid[gid], standalone_query=g['query'])
            )
        pts = g.get('expected_points') or []
        sc = await _epcov_scores(judge, jsem, ans, pts, expected_points_hits(ans, pts))
        return {
            'id': gid,
            'cond': cond,
            'run': run,
            'epcov': sc['expected_points_coverage'],
            'judge_errors': sc['expected_points_judge_errors'],
            'answer': ans,
            'evidence': [c.filename + ' :: ' + c.text[:160] for c in evid[gid]],
        }

    return list(await asyncio.gather(*(one(r['id']) for r in rows)))


async def measure(runs: int) -> None:
    from eval.generation import load_gold
    from eval.retrieval_v2 import compute

    cr = json.loads(CR_FILE.read_text(encoding='utf-8'))
    # 조각은 build가 만든 스냅숏(header 임시 인덱스)에서 읽는다 — 운영 인덱스는 공유라 build 뒤에도 다른 세션의 업로드로 바뀐다.
    # 운영에서 읽으면 바뀐 조각이 변형 텍스트 표에 없어 리랭커 입력이 조용히 운영 머리말로 돌아간다(#234 리뷰 지적 1)
    docs = await _all_docs(index_name('header'))
    rule = {cid: line for chunks in _by_document(docs).values() for cid, line in rule_lines(chunks).items()}
    gold = {g['id']: g for g in load_gold()}
    # (조건, 회차)마다 저장하고 이미 끝난 몫은 건너뛴다 — TEI·세션이 끊겨도 이어 돈다(첫 실행이 cr 1회차에서 끊겨 답변 채점이 날아갔다)
    path = OUT / 'measure.json'
    result = (
        json.loads(path.read_text(encoding='utf-8'))
        if path.exists()
        else {
            'commit': _commit(),
            'index': settings.opensearch_index,
            'started': datetime.now().isoformat(timespec='seconds'),
            'retrieval': {},
            'answers': [],
        }
    )
    OUT.mkdir(parents=True, exist_ok=True)
    for cond in CONDITIONS:
        texts = {d['chunk_id']: variant_text(cond, d, cr, rule) for d in docs if _target(d)}
        with condition(cond, texts):
            for run in range(1, runs + 1):
                key = f'{cond}#{run}'
                if key in result['retrieval']:
                    continue
                rows = (await compute())['rows']
                ctx_rows = [r for r in rows if is_context_row(r['id'])]
                answers = await _answers(cond, ctx_rows, gold, run)
                result['retrieval'][key] = rows
                result['answers'] += answers
                path.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding='utf-8')
                print(cond, run, _summary(rows), '| 맥락 셋', _summary(ctx_rows), flush=True)
    print('→', path)


def _summary(rows: list[dict]) -> dict:
    if not rows:
        return {}
    keys = rows[0]['scores'].keys()
    return {
        'n': len(rows),
        **{
            k: round(sum(r['scores'][k] for r in rows) / len(rows), 3)
            for k in keys
            if isinstance(rows[0]['scores'][k], (int, float))
        },
    }


def _commit() -> str:
    from eval.axis_common import commit

    return commit()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['contexts', 'build', 'measure', 'drop'])
    ap.add_argument('--runs', type=int, default=2)
    args = ap.parse_args()
    from rag.search.os_client import close_client

    async def run():
        try:
            if args.cmd == 'contexts':
                await make_contexts()
            elif args.cmd == 'build':
                await build()
            elif args.cmd == 'measure':
                await measure(args.runs)
            else:
                await drop()
        finally:
            await close_client()

    asyncio.run(run())


if __name__ == '__main__':
    main()
