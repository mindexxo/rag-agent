"""eval 전용 — gold의 대화 이력(list[dict])을 rag.conversation이 소비하는 Message로 (#77).

**계약.** rag/conversation.py의 이력 함수들이 이력 객체에서 읽는 것:

    속성       읽는 곳                                     없으면
    role      _history_content · build_prior_turns        user↔assistant 짝짓기 실패(턴 전부 소실)
    content   _history_content · estimate_tokens          TypeError
    status    _history_content (role=='assistant'일 때만)  ← #77이 여기서 터졌다

이 표의 근거(ground truth)는 rag/conversation.py 코드 자체이고 옮길 수 없다. 여기는 그
계약을 **eval 쪽에서 만족시키는 단일 지점**이다 — 소비처(condense·generation·retrieval_mt)는
이 함수만 부르고 필드를 직접 조립하지 않는다.

**왜 SimpleNamespace를 버렸나.** 예전엔 세 소비처가 각자 SimpleNamespace로 흉내냈다.
SimpleNamespace는 **넘긴 키만** 속성으로 갖기 때문에, 소비 함수가 새 속성을 읽기 시작하면
그 키를 안 넘긴 쪽이 AttributeError로 죽는다. 실제로 #59가 status를 도입했을 때 세 곳 중
condense.py만 고쳐졌고, 나머지 둘은 그때부터 크래시 상태였다 — #72가 failed 분기를 더하며
같은 자리가 또 걸렸고, 그 사이 생성축·멀티턴검색축을 한 번도 못 돌렸다(두 번 다 사람이
우연히 발견했다). 진짜 Message(rag/models.py, SQLAlchemy declarative)는 매핑 컬럼 전부가
인스턴스 속성으로 존재해 미설정이면 None일 뿐이라, **이 버그 클래스가 원리적으로 사라진다.**
DB·세션 없이 생성 가능하다(tests/test_condense_multiquery.py에 선례).

**status를 명시로 채우는 이유.** Message.status의 컬럼 기본값(mapped_column(default='done'))은
flush(INSERT) 시점에만 적용된다 — 여기서 만드는 transient 객체는 안 채우면 None으로 남는다.
지금은 소비처가 전부 ==/in 비교라 None도 'done'과 동치로 동작하지만, 그건 우연이다.
비교가 `!= 'done'` 형태로 바뀌는 순간 gold 이력 전부가 "비정상 턴"으로 렌더된다 — 크래시가
아니라 조용한 오측정이다. 그리고 운영 DB에는 status가 NULL인 행이 존재하지 않으므로
(컬럼 기본값이 항상 채운다) 'done'을 넣는 것은 추정이 아니라 **정확한 모사**다.
gold가 값을 실었으면 그대로 통과시킨다 — 어댑터를 안 고치고도 gold가 취소·실패 턴을
표현할 수 있다(현재 gold에 status는 0건).

**Message(**m)이 아니라 키를 명시로 꺼내는 이유.** gold는 사람이 손으로 편집하는 파일이다.
여분 키(오타·실험 필드)가 섞였을 때 TypeError로 죽는 것보다 무시하는 쪽이 낫다.
대가: "gold에 새 키를 넣었는데 어댑터가 안 읽는다"는 여기서 못 잡는다.

**이 설계가 못 막는 것.** rag/conversation.py에 새 컬럼이 생기고 _history_content가 그 값으로
분기하기 시작하면, 여기는 그 필드를 안 채우니 항상 None이다 — eval은 영원히 한쪽 분기만
타면서 통과한다. 크래시가 없어 회귀 테스트도 신호를 안 준다. 막으려면 gold 스키마 쪽
래칫이 필요하다(eval/validate_gold_v2.py의 REQUIRED_KEYS를 conversation 내부 키까지 확장).
#77 스콥에서는 제외했다 — 다시 마주치면 그 훅부터 볼 것.

실행: python -m eval._gold_history        # 계약 스모크 (DB·LLM 불필요)
"""
from rag.models import Message

DEFAULT_STATUS = 'done'   # 운영 DB의 컬럼 기본값과 같다 (rag/models.py) — 위 docstring 참조


def messages_from_conversation(conversation: list[dict] | None) -> list[Message]:
    """gold의 conversation을 condense_query·build_prior_turns가 받는 Message 리스트로.

    None·빈 리스트는 빈 리스트를 돌려준다 — 소비처의 "이력 없음" 경로와 같은 신호다.
    """
    if not conversation:
        return []
    return [Message(role=m['role'], content=m['content'],
                    status=m.get('status', DEFAULT_STATUS))
            for m in conversation]


if __name__ == '__main__':
    # 스모크 — 계약만 확인한다 (eval/ragas_adapter.py 선례)
    msgs = messages_from_conversation([{'role': 'user', 'content': '질문'},
                        {'role': 'assistant', 'content': '답변'}])
    assert [m.role for m in msgs] == ['user', 'assistant']
    assert msgs[1].status == DEFAULT_STATUS
    assert all(isinstance(m, Message) for m in msgs)
    assert messages_from_conversation([{'role': 'assistant', 'content': '', 'status': 'cancelled'}])[0].status \
        == 'cancelled', 'gold가 실은 status는 그대로 통과해야 한다'
    assert messages_from_conversation(None) == [] and messages_from_conversation([]) == []
    print(f'ok — {len(msgs)}건, status={msgs[1].status!r}')
