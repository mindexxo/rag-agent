"""eval의 condense 입력이 운영과 같은 예산으로 잘리는지.

운영은 trim_messages_for_condense를 거쳐 condense를 부른다 — 긴 이력을 그대로 넘기면 이전 답변의
수치가 재작성 질의에 주입된다. **소스를 AST로 검사한다**: 트리밍이 무는 문항이 드물어 실행 테스트로는
호출 여부를 안정적으로 못 잡고, 문자열 검색은 호출을 지워도 import 줄의 이름 때문에 통과한다.
"""

import ast  # noqa: F401  (각 테스트가 지역 import하지 않도록 모듈 상단에 둔다)


class TestCondenseTrimParity:
    """eval 세 축(condense·generation·retrieval_mt)이 운영과 같은 condense 트리밍을 태운다."""

    _AXES = ('eval/condense.py', 'eval/generation.py', 'eval/retrieval_mt.py')

    def _src(self, rel: str) -> str:
        from pathlib import Path

        return (Path(__file__).resolve().parent.parent / rel).read_text()

    def _calls(self, rel: str) -> set[str]:
        """이 파일에서 실제로 **호출**되는 이름 집합.

        문자열 검색으로는 부족하다 — 호출을 지워도 import 줄에 이름이 남아 통과한다
        (이 테스트를 만들 때 실제로 그렇게 새는 것을 확인했다).
        """
        tree = ast.parse(self._src(rel))
        return {n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}

    def test_세_축이_모두_트리밍을_태운다(self):
        for rel in self._AXES:
            assert 'trim_messages_for_condense' in self._calls(rel), f'{rel}: condense 트리밍 호출 누락 (#81)'

    def test_예산_상수를_인자로_넘긴다(self):
        """숫자를 하드코딩하면 운영이 예산을 바꿀 때 조용히 갈라진다.

        문자열 검색이면 주석에만 있어도 통과한다 — 트리밍 호출의 **실제 인자**인지 본다.
        """
        for rel in self._AXES:
            found = False
            for n in ast.walk(ast.parse(self._src(rel))):
                if not (
                    isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Name)
                    and n.func.id == 'trim_messages_for_condense'
                ):
                    continue
                found = any(isinstance(a, ast.Attribute) and a.attr == 'condense_history_budget_tokens' for a in n.args)
                if found:
                    break
            assert found, f'{rel}: 트리밍 인자로 settings.condense_history_budget_tokens 미사용'

    def test_생성_맥락은_트리밍하지_않는다(self):
        """운영은 build_prior_turns에 자르지 않은 원본을 넘긴다(자체 예산 2000으로 트리밍한다).
        여기까지 600으로 자르면 생성 맥락이 운영보다 좁아진다 — 예산이 둘인 이유가 사라진다.

        소스 문자열을 그대로 비교하면 변수명 변경·줄바꿈에 거짓 실패한다. 인자의 **모양**만 본다:
        첫 인자가 트리밍 호출이 아니고, 둘째가 생성용 예산이면 된다.
        """
        calls = [
            n
            for n in ast.walk(ast.parse(self._src('eval/generation.py')))
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'build_prior_turns'
        ]
        assert calls, 'eval/generation.py: build_prior_turns 호출이 없다'
        for c in calls:
            first = c.args[0]
            assert not (
                isinstance(first, ast.Call)
                and isinstance(first.func, ast.Name)
                and first.func.id == 'trim_messages_for_condense'
            ), '생성 맥락에 condense 예산을 적용하면 안 된다'
            assert isinstance(c.args[1], ast.Attribute) and c.args[1].attr == 'history_budget_tokens', (
                '생성용 예산 상수를 쓸 것'
            )

    def test_운영과_같은_함수를_쓴다(self):
        """eval이 자체 트리밍을 구현하면 운영과 조용히 갈라진다 — rag/의 것을 import해 쓴다."""
        for rel in self._AXES:
            tree = ast.parse(self._src(rel))
            imported = {
                a.name
                for n in ast.walk(tree)
                if isinstance(n, ast.ImportFrom) and n.module == 'rag.turn.conversation'
                for a in n.names
            }
            assert 'trim_messages_for_condense' in imported, f'{rel}: rag/의 함수를 안 쓴다'
