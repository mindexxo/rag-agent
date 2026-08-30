"""Claude CLI 생성 클라이언트 (#96 — 프론티어 비교군 (a) 생성기 교체).

OpenAI 호환 API가 아니라 `claude -p`(헤드리스 CLI)를 서브프로세스로 부른다. API 키·종량
과금 없이 좌석 사용량 안에서 돌고, 이미 개발에 쓰는 승인 채널이라 외부 전송이 신규가
아니라는 것이 채택 이유다(이슈 #96 코멘트 2026-08-29). eval 전용 — 운영 rag/에는 물리지
않는다(rag→eval import 금지 규칙과 별개로, 운영 생성기는 vLLM 하나여야 한다).

LlmClient(rag/llm.py)와 acomplete 시그니처를 맞춰 eval/generation.py에 주입한다.
다른 점 — 전부 결과 각주에 명시할 것:
- extra_body(출처 꼬리 structural_tag 제약)를 **받되 무시한다**. CLI에 그 배관이 없다.
  꼬리는 프롬프트 서술(rag/prompt_texts.py의 TAIL_EXAMPLE 규칙)만으로 유도되고, 채점기는
  꼬리 부재를 인용 0으로 세므로 미준수는 수치에 정직하게 반영된다.
- astream 없음 — eval generate()는 acomplete만 쓴다. 필요해지면 그때 추가.
- temperature·max_tokens 통제 불가(CLI 미노출).

호출 위생 (전부 실측 근거):
- 프롬프트는 stdin으로 넘긴다 — 롱컨텍스트 모드는 수십만 자라 argv 길이 제한에 걸린다.
- **stdout만** 답변으로 취급한다 — 리포 안에서 돌리면 stderr에 설정 경고가 섞여 나왔다.
- cwd를 빈 임시 디렉터리로 둔다 — 리포에서 돌리면 CLAUDE.md 등 프로젝트 컨텍스트가
  주입돼 "순수 모델" 비교가 오염된다.
- --tools "" 로 도구를 차단한다(파일 읽기 등 에이전트 행동 없는 순수 생성 콜).

스모크: python -m eval.claude_client   (LLM 서버·DB 불필요, claude CLI 로그인만 필요)
"""
import asyncio
import tempfile
from pathlib import Path

# 콜당 상한. 생성 프롬프트는 1~2분이면 족하지만 롱컨텍스트(수십만 자)는 읽기만으로도
# 오래 걸린다 — vLLM 쪽 300s(rag/llm.py)보다 넉넉히 잡되 무한 대기는 막는다.
TIMEOUT_SECONDS = 600


class ClaudeCliClient:
    """`claude -p` 래퍼. eval 하네스의 LlmClient 대체 주입용."""

    def __init__(self, model: str = "sonnet"):
        self.model = model
        # 프로젝트 컨텍스트 오염 방지용 중립 cwd — 클라이언트 수명 동안 유지
        self._workdir = tempfile.mkdtemp(prefix="claude-eval-")

    async def acomplete(self, messages: list[dict], extra_body: dict | None = None) -> str:
        """비스트리밍 1콜. extra_body는 시그니처 호환용 — 무시한다 (모듈 docstring)."""
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        # 운영 생성 프롬프트는 system+user 구성(rag/prompts.py)이라 보통 user 1개가 남는다.
        # assistant 턴이 섞인 형태가 오면 순서대로 이어붙인다 — 정보 소실보다 낫다.
        prompt = "\n\n".join(m["content"] for m in messages if m["role"] != "system")

        cmd = ["claude", "-p", "--model", self.model, "--tools", "",
               "--output-format", "text"]
        if system:
            cmd += ["--system-prompt", system]

        proc = await asyncio.create_subprocess_exec(
            *cmd, cwd=self._workdir,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            out, err = await asyncio.wait_for(
                proc.communicate(prompt.encode()), timeout=TIMEOUT_SECONDS)
        except asyncio.TimeoutError:
            proc.kill()
            raise RuntimeError(f"claude -p 타임아웃 ({TIMEOUT_SECONDS}s, model={self.model})")
        if proc.returncode != 0:
            # stderr 앞부분만 — 로그인 만료·한도 도달이 여기로 온다. 조용히 삼키면
            # 빈 답변이 채점돼 수치가 오염되므로 시끄럽게 실패한다.
            raise RuntimeError(
                f"claude -p 실패 (exit {proc.returncode}): {err.decode()[:300]}")
        return out.decode().strip()


async def _smoke():
    client = ClaudeCliClient()
    answer = await client.acomplete([
        {"role": "system", "content": "설명 없이 값만 출력한다."},
        {"role": "user", "content": "1+1은?"},
    ])
    print(f"model={client.model} → {answer!r}")
    assert answer, "빈 응답"


if __name__ == "__main__":
    asyncio.run(_smoke())
