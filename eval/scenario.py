"""종단 멀티턴 시나리오 러너 (#99) — 발표 시연 각본을 자동 재생하고 턴별로 판정한다.

기존 eval은 전부 '한 턴 조각'(검색/생성/분류 각각)이다. 실사용은 5~10턴 흐름이고 그
검증은 수동 E2E뿐이었다. 이 러너는 시연 각본과 같은 시나리오를 자동 재생해 ①통과율
②변경마다 도는 회귀 ③시연 리허설을 겸한다.

한 시나리오 = 대화 하나. 같은 conversation_id로 턴을 순서대로 재생하므로 이력·대명사
해소가 실제로 작동한다(refusal.py처럼 prepare→generate를 직접 태운다 — 운영 요청당 세션과
같게 턴마다 새 세션을 연다).

**두 종류의 턴** (turn의 seed_status 유무로 구분):
- 실행 턴: prepare→generate→finalize를 실제로 태워 route·인용·포인트를 판정.
- 시드 턴(seed_status): 취소·차단은 SSE 스트리밍+연결 끊김 기반이라 ASGI 직접 호출로는
  진짜 재현이 안 된다(tests/test_cancellation.py:289 실측). 그 턴만 DB에 해당 상태로 심고
  (실행 없이), 다음 턴이 그 상태를 어떻게 처리하는지 검증한다 — 취소 뒤 '다시'가 질문을
  복원하는가(#59), 차단 턴 첨부가 다음 턴에 안 새는가(#63). RETRY/첨부격리 디스패처는
  DB 컬럼값만 읽으므로 시드는 우회가 아니라 정석이다(리포 retry/attachment 테스트가 같은 패턴).

**maybe_cache는 일부러 호출하지 않는다** — 캐시 시나리오는 이 세트에서 제외했고(cache_eval이
커버), 리허설 재실행이 semantic 캐시 테이블을 매번 적재하지 않게 한다. 운영과 다른 유일한 지점.

의존: DB + LLM + TEI. 실행: python -m eval.scenario
"""
import asyncio
import json
import re
import time
from pathlib import Path

from database import AsyncSessionLocal
from eval._turn_cleanup import discard_turn
from eval.generation import _citation_match, _contains_point
from eval.refusal import _citations
from rag.conversation import ensure_conversation
from rag.models import Message
from rag.service import PreparedRag, RagService
from schemas.kms import QueryAttachment

GOLD = Path(__file__).resolve().parent / "scenario_set_v1.jsonl"
RESULT = Path(__file__).resolve().parent / "results" / "scenario.jsonl"
# eval.generation.citation_accuracy의 확장자 스트립과 같은 규칙 — export 안 된 세부라 한 줄 재작성.
_EXT_RE = re.compile(r"\.(pdf|docx|xlsx|txt|md)$")


def _to_attachments(attach: list[dict] | None) -> list[QueryAttachment] | None:
    if not attach:
        return None
    return [QueryAttachment(filename=a["filename"], text=a["text"]) for a in attach]


def _display_route(prepared: PreparedRag) -> str:
    """route(3값) + attachment_grounded를 합친 표시용 라벨(소문자, 저장 안 함).
    intent_label(대문자 DB 저장용)과 다른 파생점 — 시나리오 판정은 이 4값과 비교한다."""
    return "attachment_only" if prepared.attachment_grounded else prepared.route


async def _seed_turn(tenant_id: str, conversation_id: int, turn: dict) -> None:
    """cancelled/blocked 턴을 파이프라인 없이 DB에 직접 심는다 (기존 conversation_id에 이어붙임).

    tests/conftest.seed_turn과 같은 모양이지만 여기 따로 둔다 — eval은 pytest에 의존하지
    않는다(#99 설계). blocked 시드 + seed_attach면 다음 턴의 _load_history_attachments가
    blocked_turn 서브쿼리로 자동 제외한다(#63) — 별도 처리 불필요.
    """
    status = turn["seed_status"]
    if status == "cancelled" and not turn.get("seed_standalone"):
        # 실제 취소 턴은 finalize_turn이 standalone_query를 백필한다(turn_state docstring).
        # 안 채우면 RETRY 재실행이 prev_user.content 폴백(service.py:307)을 타 원문(대명사
        # 포함)을 그대로 검색해 실제 동작과 어긋난다 — 시드가 실상태를 정확히 모사하도록 강제.
        raise ValueError(f"cancelled 시드는 seed_standalone 필수: {turn.get('say')!r}")
    async with AsyncSessionLocal() as session:
        user = Message(tenant_id=tenant_id, conversation_id=conversation_id, role="user",
                       content=turn["say"], standalone_query=turn.get("seed_standalone"),
                       attachments=turn.get("seed_attach"))
        session.add(user)
        await session.flush()
        session.add(Message(tenant_id=tenant_id, conversation_id=conversation_id, role="assistant",
                            content=turn.get("seed_answer", ""), status=status,
                            question_message_id=user.id, intent=turn.get("seed_intent")))
        await session.commit()


async def _run_turn(tenant_id: str, conversation_id: int, turn: dict) -> dict:
    """실행 턴 — prepare→generate→finalize를 태우고 관측값을 수집 (턴마다 새 세션)."""
    t0 = time.monotonic()
    async with AsyncSessionLocal() as session:
        svc = RagService(tenant_id=tenant_id, session=session)
        try:
            prepared = await svc.prepare(turn["say"], conversation_id=conversation_id,
                                         attachments=_to_attachments(turn.get("attach")))
            answer = "".join([tok async for tok in svc.generate(prepared)])
            citations = _citations(prepared, answer)
            # terminal_status 사용 — 인젝션 턴은 실시간 blocked라 DB에도 blocked로 저장돼야
            # 다음 턴의 이력 격리(#22)가 성립한다. DONE 하드코딩 금지.
            await svc.finalize(prepared, answer, citations,
                               status=prepared.terminal_status,
                               latency_ms=int((time.monotonic() - t0) * 1000))
            await session.commit()
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}
    return {"route": _display_route(prepared),
            "citations": [c.filename for c in citations],
            # 프롬프트에 실제로 주입된 첨부 파일명 — 차단 첨부 격리(#63)를 코드 레벨로
            # 검증하기 위해서다. 답변 텍스트만 보면 규칙1(인젝션 방어)이 누출을 막아
            # 격리 서브쿼리 고장을 못 잡는다(리뷰 발견) — 주입 목록을 직접 관측한다.
            "attachment_files": [a["filename"] for a in (prepared.attachments or [])],
            "answer": answer, "cache_hit": prepared.is_cache_hit}


def _check(turn: dict, obs: dict) -> list[str]:
    """턴 기대 대 관측 — 어긋난 항목 목록(빈 목록=통과). 전부 결정적 신호."""
    if obs.get("error"):
        return [f"예외 {obs['error']}"]
    fails = []
    if (want := turn.get("expect_route")) is not None and obs["route"] != want:
        fails.append(f"route {obs['route']!r} != {want!r}")
    if (want := turn.get("expect_refuse")) is not None:
        refused = not obs["citations"]        # 실인용 0건 = 거절 (eval.refusal과 같은 정의)
        if refused != want:
            fails.append(f"refuse {refused} != {want}")
    if want := turn.get("expect_docs"):
        cores = [_EXT_RE.sub("", c) for c in obs["citations"]]
        groups = [[d] if isinstance(d, str) else d for d in want]
        missing = [g for g in groups
                   if not any(_citation_match(c, _EXT_RE.sub("", s)) for s in g for c in cores)]
        if missing:
            fails.append(f"인용 누락: {missing} (실제 {obs['citations']})")
    # _contains_point 재사용 — 공백만 지우는 substring은 "5만원"이 "15만원"에 오탐된다
    # (리뷰 발견). 숫자 경계+콤마 정규화를 그대로 가져와 수치 판정을 정확히 한다.
    norm_answer = obs["answer"].replace(" ", "")
    for s in turn.get("expect_contains", []):
        if not _contains_point(norm_answer, s.replace(" ", "")):
            fails.append(f"문구 누락: {s!r}")
    for s in turn.get("expect_not_contains", []):
        if _contains_point(norm_answer, s.replace(" ", "")):
            fails.append(f"금지 문구 존재: {s!r}")
    for fn in turn.get("expect_not_attached", []):     # 차단 첨부 격리 코드 레벨 검증 (#63)
        if any(fn in af for af in obs.get("attachment_files", [])):
            fails.append(f"차단 첨부 주입됨: {fn!r} (실제 {obs.get('attachment_files')})")
    return fails


async def run_scenario(sc: dict) -> dict:
    """시나리오 하나 = 대화 하나. 턴을 순서대로 재생하고 판정. 정리는 끝에 1회."""
    tenant = sc["tenant"]
    async with AsyncSessionLocal() as session:
        # 턴 0 전에 대화를 미리 만들어 정리할 id를 확보 (refusal.py와 같은 #72 근거)
        conversation = await ensure_conversation(session, tenant, None)
        await session.commit()
        conv_id = conversation.id

    turn_results = []
    try:
        for i, turn in enumerate(sc["turns"]):
            if turn.get("seed_status"):
                await _seed_turn(tenant, conv_id, turn)
                turn_results.append({"turn": i, "seeded": turn["seed_status"], "fails": []})
                continue
            obs = await _run_turn(tenant, conv_id, turn)
            turn_results.append({"turn": i, "route": obs.get("route"),
                                 "cache_hit": obs.get("cache_hit", False),
                                 "answer": obs.get("answer", "")[:120],
                                 "fails": _check(turn, obs)})
    finally:
        async with AsyncSessionLocal() as session:
            await discard_turn(session, tenant, conv_id)

    passed = all(not t["fails"] for t in turn_results)
    return {"id": sc["id"], "name": sc["name"], "passed": passed, "turns": turn_results}


async def compute() -> list[dict]:
    scenarios = [json.loads(l) for l in GOLD.read_text().splitlines() if l.strip()]
    results = []
    for sc in scenarios:      # 순차 — 멀티턴 순서를 로그로 따라가야 하는 리허설 도구라 순서 고정 우선
        try:
            results.append(await run_scenario(sc))
        except Exception as exc:
            results.append({"id": sc["id"], "name": sc.get("name", ""), "passed": False,
                            "turns": [], "error": f"{type(exc).__name__}: {exc}"})
    return results


def _print_report(results: list[dict]) -> None:
    n_ok = sum(r["passed"] for r in results)
    print(f"[멀티턴 시나리오]  {n_ok}/{len(results)} 통과\n")
    for r in results:
        print(f"[{'OK' if r['passed'] else 'FAIL'}] {r['id']} — {r['name']}")
        if r.get("error"):
            print(f"    시나리오 예외: {r['error']}")
        for t in r.get("turns", []):
            if "seeded" in t:
                print(f"    턴{t['turn']} (시드:{t['seeded']})")
                continue
            cache = " [cache]" if t.get("cache_hit") else ""
            print(f"    턴{t['turn']} route={t.get('route')}{cache}  {t.get('answer', '')[:60]!r}")
            for f in t["fails"]:
                print(f"      ✗ {f}")


async def main() -> None:
    results = await compute()
    RESULT.parent.mkdir(exist_ok=True)
    RESULT.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in results) + "\n")
    _print_report(results)
    print(f"\n→ {RESULT}")


if __name__ == "__main__":
    asyncio.run(main())
