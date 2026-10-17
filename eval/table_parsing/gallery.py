"""표 형식 23축 샘플 갤러리 — "이런 표 케이스를 검증했다"를 공유하고, 새 케이스를 더할 때 눈으로 확인하는 용도.

용법: python -m eval.table_parsing.gallery <out_dir>   → <out_dir>/gallery.html + png/<id>.png
축마다 짝(a·b) 중 첫 정의 하나를 표 이미지로 그려 축 이름·경로와 함께 한 화면에 놓는다(측정 결과는 싣지 않는다 —
그건 run.py의 matrix_*.md·mismatches.md). 케이스를 더하면 specs.py에 정의를 넣고 이 명령으로 다시 만든다.
"""
import sys
from pathlib import Path

from eval.table_parsing.render_image import render_spec
from eval.table_parsing.specs import SPECS, paths_of

_STYLE = ('<style>body{font-family:-apple-system,sans-serif;margin:24px;background:#fafafa}'
          '.c{display:inline-block;vertical-align:top;margin:12px;padding:12px;background:#fff;border:1px solid #ddd;'
          'border-radius:8px;max-width:640px}img{max-width:600px;max-height:400px;border:1px dashed #ccc;display:block}'
          'h3{margin:0 0 4px;font-size:15px}.m{color:#888;font-size:12px;margin-bottom:8px}</style>')


def build_gallery(out: Path) -> Path:
    (out / 'png').mkdir(parents=True, exist_ok=True)
    cards, seen = [], set()
    for spec in SPECS:
        if spec['axis'] in seen:            # 축당 첫 정의(a)만 — 짝(b)은 값만 다르다
            continue
        seen.add(spec['axis'])
        png = out / 'png' / f"{spec['id']}.png"
        render_spec(spec).save(png)
        cards.append(f'<div class="c"><h3>{spec["axis"]}</h3><div class="m">{spec["id"]} · 경로: '
                     f'{" · ".join(paths_of(spec))}</div><img src="png/{png.name}"></div>')
    page = out / 'gallery.html'
    page.write_text('<meta charset="utf-8">' + _STYLE + f'<h1>표 파싱 검증 — 표 형식 {len(cards)}축 (#246)</h1>'
                    '<p>축마다 값만 다른 표 2개를 만들어 표 이미지(OCR)·텍스트 PDF·DOCX·XLSX 네 경로로 검증했다. '
                    '그림은 표 이미지 경로의 입력이고, 다른 경로에는 같은 값이 진짜 표로 들어간다. '
                    '케이스 추가는 <code>eval/table_parsing/specs.py</code>.</p>' + ''.join(cards))
    return page


if __name__ == '__main__':
    print(build_gallery(Path(sys.argv[1])))
