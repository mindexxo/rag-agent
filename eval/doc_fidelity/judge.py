"""그림 캡션 1차 판정 — Claude가 원본 크롭과 캡션을 보고 rubric(결정 6)대로 센다(#247 결정 1 = b-1).

등급은 rubric.grade_of가 개수에서 다시 계산한다. 판정자가 매긴 등급과 다르면 `grade_mismatch`로 남기고 코드 쪽을 쓴다 —
같은 기준을 사람 재판정에도 쓰므로 등급의 정의는 한 곳(코드)에 있어야 한다.
JSON 파싱·검증 실패는 1회 재시도, 그래도 실패하면 `error`로 남겨 사람 재판정 대상에 넣는다(조용히 '정확'으로 두지 않는다).
"""
import asyncio
import json
import re

from pydantic import ValidationError

from eval.doc_fidelity.rubric import PROMPT, PROMPT_VERSION, Verdict, grade_of

DEFAULT_MODEL = 'claude-sonnet-4-5-20250929'     # #253 재판정자와 같은 모델(eval/table_parsing/llm_judge.py)
_JSON = re.compile(r'\{.*\}', re.S)


def parse(text: str) -> Verdict:
    m = _JSON.search(text or '')
    if not m:
        raise ValueError(f'JSON 없음: {text[:120]!r}')
    return Verdict.model_validate(json.loads(m.group(0)))


async def judge_one(client, image_png: bytes, caption: str) -> dict:
    prompt = PROMPT + (caption if caption else '(캡션 없음 — 이 그림은 색인에 글이 들어가지 않았다)')
    last = ''
    for _ in range(2):
        raw = await client.acomplete_image(prompt, image_png)
        try:
            v = parse(raw)
        except (ValueError, ValidationError) as e:
            last = f'{type(e).__name__}: {str(e)[:160]}'
            continue
        code = grade_of(v)
        return {**v.model_dump(), 'judge_grade': v.grade, 'grade': code, 'grade_mismatch': code is not None and code != v.grade,
                'judge': f'claude:{client.model}:{PROMPT_VERSION}', 'error': ''}
    return {'grade': None, 'error': last, 'judge': f'claude:{client.model}:{PROMPT_VERSION}'}


async def judge_all(records: list[dict], crops_dir, concurrency: int = 3, model: str = DEFAULT_MODEL,
                    sink=None) -> list[dict]:
    """records(pictures 기록) → 판정 붙인 기록. 크롭이 없는 그림은 판정하지 않는다(error='크롭 없음').

    호출 실패(한도·로그인 만료·시간 초과 — ClaudeCliClient가 RuntimeError로 올린다)도 그 그림의 error로 남기고 나머지를 계속한다.
    전에는 예외가 gather 밖으로 새어 실행 전체가 멈추고 끝까지 모은 판정이 통째로 사라졌다(리뷰 지적, #247). sink(Path)가 있으면
    그림 하나가 끝날 때마다 JSONL로 이어 쓴다 — 프로세스가 죽어도 진행분이 남고 --retry-errors가 실패분만 다시 돌린다.
    """
    import json
    from eval.claude_client import ClaudeCliClient
    client = ClaudeCliClient(model=model)
    sem = asyncio.Semaphore(concurrency)
    fh = sink.open('a') if sink else None

    async def one(r):
        if not r.get('crop'):
            v = {'grade': None, 'error': '크롭 없음'}
        else:
            async with sem:
                try:
                    v = await judge_one(client, (crops_dir / r['crop']).read_bytes(), r.get('caption') or '')
                except RuntimeError as e:
                    v = {'grade': None, 'error': f'호출 실패: {str(e)[:200]}', 'judge': f'claude:{client.model}:{PROMPT_VERSION}'}
        row = {**r, 'verdict': v}
        print(f"  {r['picture_id']} {v.get('type', '?')} → {v.get('grade')} {v.get('error', '')}", flush=True)
        if fh:
            fh.write(json.dumps(row, ensure_ascii=False) + '\n')
            fh.flush()
        return row

    try:
        return await asyncio.gather(*(one(r) for r in records))
    finally:
        if fh:
            fh.close()
