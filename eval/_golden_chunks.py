"""청킹 골든 — 리팩터링 전/후 청크 출력을 통째로 비교한다 (#42).

청킹은 결정적이고 DB·TEI가 필요 없다. 그래서 검색을 거치지 않고 **청크 그 자체**를
전수 캡처해 대조할 수 있다 — 2번 축의 검색 순서 골든보다 직접적이다.

`chunk_file` 하나만 부른다 — 형식 분기는 그 안에 한 곳뿐이다(#42). 여기서 확장자를
다시 보면, 이 스크립트가 잡아야 할 바로 그 유형(분기가 두 곳이라 갈리는 버그)을 놓친다.

TEI·DB 불필요. 건수·소요 시간은 적지 않는다 — 문서가 늘 때마다 낡는다(docling 변환이 들어간 뒤로는 분 단위다).

사용:
    python -m eval._golden_chunks > before.json               # 변경 전
    python -m eval._golden_chunks > after.json                # 변경 후
    python -m eval._golden_chunks --diff before.json after.json   # 형식별 변화량 요약 + 파일별 사라진 줄
    python -m eval._golden_chunks --files ../consult-agent/docs/inticube > before.json   # 대상 지정(파일·디렉터리)

`--files`는 워크트리에 없는 실문서(gitignore된 docs/)를 **읽기만** 하려고 둔다 — 복사하지 않는다(#233 설계 5).
"사라진 줄"은 전 청크의 줄 중 후 청크의 제목 경로·본문 어디에도 없는 것이다. 판정은 eval/chunk_invariants
(원문 소실 0의 정의점)와 같은 함수다 — 테스트와 이 도구가 "소실"을 다르게 세면 PR 수치를 테스트로 재현할 수 없다.
"""
import argparse
import json
import sys
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.chunk_invariants import missing_lines   # noqa: E402 — "소실"의 정의점(위 docstring)


ROOTS = ('sample_docs', 'docs')
EXTS = ('.pdf', '.docx', '.md', '.txt', '.xlsx')


def chunks_for(path: Path):
    """운영과 같은 진입점 — chunk_file이 5형식을 전부 처리한다."""
    from rag.chunking import chunk_file
    return chunk_file(path)


def capture(targets: list[str] | None = None) -> dict:
    base = Path(__file__).resolve().parent.parent
    out = {}
    roots = [Path(t).resolve() for t in targets] if targets else [base / root for root in ROOTS]
    files = sorted({
        p for root in roots for p in ([root] if root.is_file() else root.rglob('*'))
        if p.suffix.lower() in EXTS and p.is_file()
    })
    for p in files:
        rel = str(p.relative_to(base)) if p.is_relative_to(base) else str(p)
        try:
            cs = chunks_for(p)
            out[rel] = [
                {'i': c.chunk_index, 'h': c.heading_path, 'p': c.page,
                 'm': c.meta or {}, 't': c.text}
                for c in cs
            ]
        except Exception as e:                    # 파싱 실패도 골든의 일부 — 전후가 같아야 한다
            out[rel] = {'error': f'{type(e).__name__}: {e}'}
    # 전부 실패했으면 골든이 아니라 환경 문제다. 조용히 넘기면 오류 목록이 데이터처럼
    # 보이고 diff가 '변화 없음'으로 나와 회귀를 놓친다 (실제로 한 번 겪었다).
    if out and not any(isinstance(v, list) for v in out.values()):
        raise SystemExit(f'전 파일 파싱 실패 — 환경을 확인할 것. 예: {next(iter(out.values()))}')
    return out


def show_diff(before_path: str, after_path: str) -> None:
    b = json.load(open(before_path, encoding='utf-8'))
    a = json.load(open(after_path, encoding='utf-8'))
    by_ext: dict[str, list[int]] = {}             # [파일수, 변한 파일수, 전 청크, 후 청크]
    changed_files = []
    for rel in sorted(set(b) | set(a)):
        ext = Path(rel).suffix.lower()
        s = by_ext.setdefault(ext, [0, 0, 0, 0])
        s[0] += 1
        bv, av = b.get(rel), a.get(rel)
        s[2] += len(bv) if isinstance(bv, list) else 0
        s[3] += len(av) if isinstance(av, list) else 0
        if bv != av:
            s[1] += 1
            changed_files.append(rel)

    print(f"{'형식':<8}{'파일':>6}{'변한 파일':>10}{'전 청크':>9}{'후 청크':>9}")
    for ext, (n, ch, bc, ac) in sorted(by_ext.items()):
        mark = '  ← 변화' if ch else ''
        print(f'  {ext:<8}{n:>5}{ch:>9}{bc:>10}{ac:>9}{mark}')

    if not changed_files:
        print('\n✅ 전 형식 바이트 동일')
        return

    print(f'\n변한 파일 {len(changed_files)}개 — 청크 수 변화 / 사라진 줄(전 청크에 있고 후 청크에 없음):')
    total_lost = 0
    for rel in changed_files:
        bv, av = b.get(rel), a.get(rel)
        if bv is None or av is None:
            print(f'  {rel}: {"새 파일" if bv is None else "없어진 파일"}')
            continue
        if not (isinstance(bv, list) and isinstance(av, list)):
            print(f'  {rel}: 파싱 오류 {bv if isinstance(bv, dict) else "정상"} → {av if isinstance(av, dict) else "정상"}')
            continue
        lines = [x for c in bv for x in (*c['h'], *c['t'].split('\n'))]
        lost = missing_lines(lines, [h for c in av for h in c['h']], [c['t'] for c in av])
        total_lost += len(lost)
        print(f'  {rel}: 청크 {len(bv)} → {len(av)}, 사라진 줄 {len(lost)}')
        for t in lost[:5]:
            print(f'    − {t[:80]!r}')
    print(f'\n사라진 줄 합계 {total_lost}')


ap = argparse.ArgumentParser()
ap.add_argument('--diff', nargs=2, metavar=('BEFORE', 'AFTER'))
ap.add_argument('--files', nargs='+', metavar='PATH', help='캡처 대상 파일·디렉터리(기본: sample_docs·docs)')
args = ap.parse_args()
if args.diff:
    show_diff(*args.diff)
else:
    print(json.dumps(capture(args.files), ensure_ascii=False, indent=1, sort_keys=True))
