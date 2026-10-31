"""주석·docstring만 바꾼 변경이 코드를 건드리지 않았음을 증명한다 — 두 리비전의 docstring 뺀 AST를 대조한다.

주석은 원래 AST에 없고, docstring은 양쪽에서 똑같이 걷어낸다(본문이 docstring뿐이던 함수는 `pass`로 맞춘다).
그 외 한 글자라도 다르면 실패다. docstring이 실행에 쓰이는 곳(pydantic 모델 설명이 LLM 스키마에 들어가는 등)은
이 도구가 못 보니 `tests/test_prompt_bytes.py`가 맡는다.

    python -m tests.same_code origin/main rag/turn/service.py rag/turn/streaming.py ...
"""

import ast
import subprocess
import sys
from collections.abc import Sequence

_SCOPES = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)


def _is_docstring(node: ast.stmt) -> bool:
    return isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)


def stripped_dump(source: str) -> str:
    """docstring을 뺀 AST의 위치 정보 없는 덤프."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, _SCOPES) and node.body and _is_docstring(node.body[0]):
            node.body = node.body[1:] or ([] if isinstance(node, ast.Module) else [ast.Pass()])
    return ast.dump(tree, include_attributes=False)


def _at(ref: str, path: str) -> str:
    return subprocess.run(['git', 'show', f'{ref}:{path}'], capture_output=True, text=True, check=True).stdout


def changed_code(ref: str, paths: Sequence[str]) -> list[str]:
    """ref와 작업 트리의 docstring 뺀 AST가 다른 파일."""
    out = []
    for path in paths:
        with open(path, encoding='utf-8') as f:
            if stripped_dump(_at(ref, path)) != stripped_dump(f.read()):
                out.append(path)
    return out


def main(argv: Sequence[str]) -> int:
    ref, *paths = argv
    diff = changed_code(ref, paths)
    for path in diff:
        print(f'코드 변경: {path}')
    print(f'{len(paths) - len(diff)}/{len(paths)} 파일 코드 동일')
    return 1 if diff else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
