"""#235 실행기 — 패키지 docstring(eval/scale_noise/__init__.py)의 사용법 참조."""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eval.scale_noise import CONDITIONS  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(prog='python -m eval.scale_noise')
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('generate').add_argument('--n', type=int, default=1200)
    sub.add_parser('filter')
    sub.add_parser('chunks')
    sub.add_parser('build').add_argument('--scales', default=','.join(CONDITIONS))
    sub.add_parser('measure').add_argument('--scales', default=','.join(CONDITIONS))
    sub.add_parser('drop')
    a = ap.parse_args()
    if a.cmd == 'generate':
        from eval.scale_noise.generate import generate

        asyncio.run(generate(a.n))
    elif a.cmd == 'filter':
        from eval.scale_noise.filter import run

        run()
    elif a.cmd == 'chunks':
        from eval.scale_noise.chunks import run

        asyncio.run(run())
    elif a.cmd == 'build':
        from eval.scale_noise.build import build

        asyncio.run(build(a.scales.split(',')))
    elif a.cmd == 'measure':
        from eval.scale_noise.measure import measure

        asyncio.run(measure(a.scales.split(',')))
    elif a.cmd == 'drop':
        from eval.scale_noise.build import drop

        asyncio.run(drop())


if __name__ == '__main__':
    main()
