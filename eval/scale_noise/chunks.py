"""#235 노이즈 문서 → 운영과 같은 조각·같은 벡터.

청킹은 워커와 같은 rag/chunking.chunk_file(.md는 docling MD 백엔드, #267)을 자식 프로세스 풀로 돌리고, 임베딩 입력은 운영 색인과
같은 build_index_text('파일명 > 헤딩' 머리말)로 조립해 TEI에 보낸다 — 노이즈 조각의 길이·벡터 분포가 실문서 조각과 같아야
규모 실험이 된다. 파일명은 '<회사> <주제>.md'(id가 아니라 사람이 읽는 이름 — 머리말이 임베딩에 들어가므로).
산출: CHUNKS(한 줄 = 조각: doc·filename·chunk_index·text·heading_path·page·meta·dense). 이미 있는 doc은 건너뛴다.
"""

import json
from concurrent.futures import ProcessPoolExecutor

from eval.scale_noise import CHUNKS, DOCS, FILTER, MANIFEST
from rag.search.embeddings import embed_texts
from rag.search.index_text import build_index_text

WORKERS = 4


def _chunk_one(args: tuple[str, str]) -> tuple[str, list[dict]]:
    doc_id, path = args
    from rag.chunking import chunk_file

    out = []
    for c in chunk_file(path):
        out.append(
            {
                'chunk_index': c.chunk_index,
                'text': c.text,
                'heading_path': list(c.heading_path or []),
                'page': c.page,
                'meta': {k: v for k, v in (c.meta or {}).items() if k != 'table_text' or v},
            }
        )
    return doc_id, out


def _done() -> set[str]:
    if not CHUNKS.exists():
        return set()
    return {json.loads(l)['doc'] for l in CHUNKS.read_text(encoding='utf-8').splitlines() if l.strip()}


async def run() -> None:
    kept = set(json.loads(FILTER.read_text(encoding='utf-8'))['kept'])
    meta = {r['id']: r for r in (json.loads(l) for l in MANIFEST.read_text(encoding='utf-8').splitlines() if l.strip())}
    todo = [d for d in sorted(kept) if d not in _done()]
    print(f'유지 {len(kept)} · 남은 {len(todo)}', flush=True)
    with ProcessPoolExecutor(WORKERS) as pool, CHUNKS.open('a', encoding='utf-8') as fp:
        n_chunks = 0
        for i, (doc_id, chunks) in enumerate(pool.map(_chunk_one, [(d, str(DOCS / f'{d}.md')) for d in todo]), 1):
            fname = f"{meta[doc_id]['company']} {meta[doc_id]['topic']}.md"
            vecs = await embed_texts([build_index_text(c['text'], fname, c['heading_path']) for c in chunks])
            # 문서 한 편의 줄을 한 번에 쓴다 — 도중에 죽어 일부 조각만 남은 문서가 '완료'로 잡히지 않게(리뷰 지적 7)
            fp.write(
                ''.join(
                    json.dumps(
                        {'doc': doc_id, 'filename': fname, **c, 'dense': [float(x) for x in v.dense]},
                        ensure_ascii=False,
                    )
                    + '\n'
                    for c, v in zip(chunks, vecs)
                )
            )
            fp.flush()
            n_chunks += len(chunks)
            if i % 25 == 0:
                print(f'  {i}/{len(todo)} 문서 · 조각 {n_chunks} (문서당 {n_chunks / i:.1f})', flush=True)
    print('완료', flush=True)
