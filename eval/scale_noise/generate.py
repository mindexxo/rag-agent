"""#235 노이즈 문서 생성 — 운영 vLLM(14B)에 직접 요청, 이어 돌리기 가능.

운영 rag/llm 공급자를 쓰지 않는 이유: 온도 0.2·max_tokens 3,000이 고정이라 수백 편이 비슷해지고 긴 규정이 잘린다.
eval/ragas_eval.py처럼 AsyncOpenAI를 직접 만들어 온도·상한을 넘긴다(thinking off는 동일).
동시성은 NOISE_CONCURRENCY(기본 12) — worker15 공유 장비, 14B 안전선 동시 24·32 붕괴(#101·#114).
산출: DOCS/<id>.md + MANIFEST(한 줄 = 문서 하나: id·kind·topic·company·chars·sec). 이미 manifest에 있는 id는 건너뛴다.
"""

import asyncio
import json
import os
import re
import time

import httpx
from openai import AsyncOpenAI

from config import settings
from eval.scale_noise import DOCS, MANIFEST
from eval.scale_noise.topics import jobs, prompt

CONCURRENCY = int(os.getenv('NOISE_CONCURRENCY', '12'))
MAX_TOKENS = 6000
_FENCE = re.compile(r'^```[a-z]*\n|\n```\s*$')


def _done() -> set[str]:
    if not MANIFEST.exists():
        return set()
    return {json.loads(l)['id'] for l in MANIFEST.read_text(encoding='utf-8').splitlines() if l.strip()}


async def generate(n: int) -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    done = _done()
    todo = [j for j in jobs(n) if j['id'] not in done]
    print(f'작업 {n} · 남은 {len(todo)} · 동시 {CONCURRENCY}', flush=True)
    cli = AsyncOpenAI(base_url=settings.llm_base_url, api_key='EMPTY', timeout=httpx.Timeout(600.0, connect=5.0))
    sem = asyncio.Semaphore(CONCURRENCY)
    t0, n_done, n_tok = time.time(), 0, 0

    async def one(job: dict) -> None:
        nonlocal n_done, n_tok
        async with sem:
            t = time.time()
            r = await cli.chat.completions.create(
                model=settings.llm_model,
                messages=[{'role': 'user', 'content': prompt(job)}],
                max_tokens=MAX_TOKENS,
                temperature=0.8,
                extra_body={'chat_template_kwargs': {'enable_thinking': False}},
            )
            text = _FENCE.sub('', (r.choices[0].message.content or '').strip()).strip() + '\n'
            (DOCS / f"{job['id']}.md").write_text(text, encoding='utf-8')
            rec = {
                **job,
                'chars': len(text),
                'tokens': r.usage.completion_tokens if r.usage else None,
                'finish': r.choices[0].finish_reason,
                'sec': round(time.time() - t, 1),
            }
            with MANIFEST.open('a', encoding='utf-8') as fp:  # 한 작업씩 바로 기록 — 끊겨도 이어 돈다
                fp.write(json.dumps(rec, ensure_ascii=False) + '\n')
            n_done += 1
            n_tok += rec['tokens'] or 0
            if n_done % 20 == 0:
                el = time.time() - t0
                print(
                    f'  {n_done}/{len(todo)} {el / 60:.1f}분 · {n_tok / el:.0f} tok/s · 평균 {rec["chars"]}자',
                    flush=True,
                )

    errors = [
        r for r in await asyncio.gather(*(one(j) for j in todo), return_exceptions=True) if isinstance(r, Exception)
    ]
    print(
        f'완료 {n_done}편 {(time.time() - t0) / 60:.1f}분 · 실패 {len(errors)}'
        + (
            f' (예: {type(errors[0]).__name__}: {str(errors[0])[:120]}) — 다시 돌리면 실패분만 이어 생성'
            if errors
            else ''
        ),
        flush=True,
    )
