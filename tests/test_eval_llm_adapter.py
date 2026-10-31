"""eval 판정기·캡션 키 계약 (#290) — DB·서버 불필요.

고정하는 것:
1. 캡션 헤더: 키 비면 {}(사내, 전과 같은 요청), 있으면 Authorization: Bearer(결정 1)
2. RAGAS 어댑터는 어떤 LlmClient든 감싸 acomplete(role dict)로 부른다 — Claude CLI 전용이 아니다(결정 2)
3. RAGAS_JUDGE=settings는 create_llm()(= LLM_PROVIDER)을, claude는 CLI 클라이언트를 감싼다. 모르는 값은 즉시 실패
4. settings judge가 vLLM에 보내는 요청 바디 = 예전 ChatOpenAI 직접 생성 때 손으로 박던 값(enable_thinking False·
   max_tokens·temperature) — RAGAS 이력 연속성의 근거
"""
import importlib.util

import pytest

from config import settings
from rag.chunking.runtime import caption_headers


class TestCaptionHeaders:
    def test_빈_키는_헤더_없음(self, monkeypatch):
        monkeypatch.setattr(settings, "vlm_caption_api_key", "")
        assert caption_headers() == {}

    def test_키가_있으면_bearer(self, monkeypatch):
        monkeypatch.setattr(settings, "vlm_caption_api_key", "k-test")
        assert caption_headers() == {"Authorization": "Bearer k-test"}

    def test_키가_docling_옵션까지_닿는다(self, monkeypatch):
        """헬퍼가 아니라 파이프라인 옵션에 실리는지 — 호출부가 headers= 를 빠뜨리면 사내에선 티가 안 난다(리뷰 4)."""
        from docling.datamodel.base_models import InputFormat
        import rag.chunking as ch
        monkeypatch.setattr(settings, "vlm_caption_api_key", "k-test")
        monkeypatch.setattr(ch.runtime, "_docling_converter", None)   # 프로세스당 1회 캐시 — 키 바꾼 뒤 다시 만들게 (운영은 기동 시 설정 고정)
        conv = ch.runtime.docling_runtime()
        for fmt in (InputFormat.PDF, InputFormat.DOCX, InputFormat.XLSX):
            assert conv.format_to_options[fmt].pipeline_options.picture_description_options.headers == {"Authorization": "Bearer k-test"}


class _FakeLlm:
    def __init__(self): self.calls = []
    async def acomplete(self, messages, **kw):
        self.calls.append(messages)
        return "판정"


class TestAdapter:
    pytestmark = pytest.mark.skipif(importlib.util.find_spec("ragas") is None, reason="ragas 미설치")   # 캡션 헤더 테스트는 ragas 없이도 돈다(리뷰 1)

    @pytest.mark.anyio
    async def test_어떤_LlmClient든_감싸_role_dict로_부른다(self):
        from langchain_core.messages import HumanMessage, SystemMessage
        from eval.llm_langchain import LlmClientChatModel
        fake = _FakeLlm()
        result = await LlmClientChatModel(client=fake)._agenerate([SystemMessage(content="규칙"), HumanMessage(content="질문")])
        assert result.generations[0].message.content == "판정"
        assert fake.calls == [[{"role": "system", "content": "규칙"}, {"role": "user", "content": "질문"}]]


class TestJudgeSelection:
    pytestmark = pytest.mark.skipif(importlib.util.find_spec("ragas") is None, reason="ragas 미설치")

    def test_settings는_create_llm을_감싼다(self, monkeypatch):
        import eval.ragas_eval as re_
        import rag.llm
        fake = _FakeLlm()
        monkeypatch.setattr(rag.llm, "create_llm", lambda: fake)
        judge, workers = re_._judge("settings")
        assert judge.langchain_llm.client is fake and workers == 32
        assert judge.bypass_temperature is True      # RAGAS의 temperature 강제 주입 차단 — 온도는 설정(LLM_TEMPERATURE) 몫

    def test_claude는_CLI_클라이언트(self, monkeypatch):
        import eval.ragas_eval as re_
        from eval.claude_client import ClaudeCliClient
        import eval.claude_client as cc
        monkeypatch.setattr(cc.shutil, "which", lambda _: "/usr/bin/claude")   # CLI 없는 기기·CI에서도 돈다(리뷰 2)
        monkeypatch.setenv("CLAUDE_MODEL", "sonnet")
        judge, workers = re_._judge("claude")
        assert isinstance(judge.langchain_llm.client, ClaudeCliClient) and workers == 8

    @pytest.mark.parametrize("old", ["vllm", "openai"])   # #290 전 값 — 조용히 기본으로 붙지 않고 바로 알려 준다
    def test_옛_값은_즉시_실패(self, old):
        import eval.ragas_eval as re_
        with pytest.raises(SystemExit, match=f"RAGAS_JUDGE={old}"):
            re_._judge(old)

    def test_진행파일_태그는_사내면_예전_이름_그대로(self, monkeypatch):
        """settings judge + vLLM = 'vllm_<모델>' — #290 전 진행 파일과 같은 키라 이어 돌리기가 유지된다."""
        import eval.ragas_eval as re_
        monkeypatch.setattr(settings, "llm_provider", "vllm"); monkeypatch.setattr(settings, "llm_model", "Qwen/Qwen3-14B")
        assert re_._progress_tag("settings") == "vllm_Qwen_Qwen3-14B"
        monkeypatch.setenv("CLAUDE_MODEL", "opus")
        assert re_._progress_tag("claude") == "claude_opus"

    def test_vllm_judge_요청바디는_예전_수동값과_같다(self, monkeypatch):
        """#216에서 ChatOpenAI에 손으로 넣던 값: temperature 0.2·max_tokens=generation_reserve_tokens·enable_thinking False."""
        from rag.llm.vllm import VllmClient
        monkeypatch.setattr(settings, "llm_reasoning_effort", None)
        monkeypatch.setattr(settings, "llm_temperature", 0.2)       # 환경(.env)이 아니라 값으로 고정(리뷰 3)
        kw = VllmClient()._request_kwargs(None, None)
        assert kw["temperature"] == 0.2
        assert kw["max_tokens"] == settings.generation_reserve_tokens
        assert kw["extra_body"]["chat_template_kwargs"] == {"enable_thinking": False}
