import uuid
import pytest
import pytest_asyncio
from sqlalchemy import delete, select

from database import AsyncSessionLocal
from rag.models import Document

"""
    1. async def - 코루틴 함수
    async def로 정의된 함수는 호출해도 즉시 실행되지 않음. 대신 '코루틴 객체'를 반환.
    -> lazy 하게 함수를 사용
    Q. 어떻게 실행시키나?
        A-1. 최상위에서 한번(보통 main())
            asyncio.run(코루틴객체)
        A-2. 다른 async def 안에서 await:
            async def fun():
                result = await 코루틴객체 (<- 이 때 바로 실행됨)
    but. 만약 async def + yield가 있다면.. 그건 제너레이터 함수가 된다. 그리고 코루틴이 아니라 제너레이터 객체 반환

    ┌───────────────┬──────────────────────────────────┬──────────────────────────────────────┐
    │               │      async def (yield 없음)       │        async def (yield 있음)         │
    ├───────────────┼──────────────────────────────────┼──────────────────────────────────────┤
    │ 종류           │ 코루틴 함수                         │ async 제너레이터 함수                    │
    ├───────────────┼──────────────────────────────────┼──────────────────────────────────────┤
    │ 호출 결과       │ 코루틴 객체                         │ async 제너레이터 객체                    │
    ├───────────────┼──────────────────────────────────┼──────────────────────────────────────┤
    │ 풀어내는 방법    │ await gen (한 번만, 단일 반환값)      │ await gen.__anext__() 또는 async for  │
    └───────────────┴──────────────────────────────────┴──────────────────────────────────────┘

    asyncio.run(fun()) <- 근데 결국 이렇게 해야 실행됨.
    
    2. await - 코루틴 풀어내기
    이 비동기 작업이 끝날 때까지 기다림(자바의 vitualThread 유사). 단 그 동안 이벤트 루프는 다른 일 할 수 있음( like 멀티플렉싱)

    3. yield - 재너레이터, 일시정지
    함수 실행을 중간에 멈추고 값을 내준 다음, 다시 호출(next())하면 그 자리에서 재개.


아래 테스트코드의 async 는 라이브러리에서 알아서 asyncio.run 호출해줌.
ex)
    async def runner():
        g = session()
        session = await g.__anext__()
        await test(session)
        try:
            await g.__anext__()
        except: StopAsyncIteration:
            pass
    
    asyncio.run(runner()) 
"""



# @pytest_asyncio.fixture: async fixture 등록 데코레이터.
#                         (일반 @pytest.fixture는 동기용이라 async에는 따로 필요)
# fixture: 테스트가 의존하는 리소스를 만들어주는 함수.
#          파라미터 이름(session)이 fixture 이름이랑 매칭되면 pytest가 자동 주입.
@pytest_asyncio.fixture
async def session():
    """요청당 세션 1개. 테스트 종료 시 자동 닫힘."""
    # async with: 컨텍스트 매니저의 async 버전.
    #             AsyncSessionLocal()이 enter/exit를 await로 처리.
    async with AsyncSessionLocal() as session:
        yield session
        # yield 이전 = setup, 이후 = teardown.
        # 여기서는 yield로 session을 테스트에 넘겨주고 함수가 일시정지.
        # 테스트 끝나면 재개되어 async with 블록 종료 -> 세션 close (풀로 반환).


# @pytest.mark.asyncio: 이 테스트가 async 함수임을 pytest에 알림.
#                      strict 모드에선 이 마커 없으면 async 테스트가 실행 안 됨.
@pytest.mark.asyncio
async def test(session):
    """A로 insert -> B로 조회 시 빈 결과, A로 조회 시 자기 row가 보여야 한다."""
    # uuid.uuid4(): 충돌 거의 불가능한 랜덤 식별자. 테스트 간 데이터 안 섞이게.
    tenant_a = str(uuid.uuid4())
    tenant_b = str(uuid.uuid4())

    # try/finally: 테스트가 assert 실패해도 finally의 cleanup은 무조건 실행됨.
    #              다음 테스트가 깨끗한 상태에서 시작하도록 보장.
    try:
        # A로 INSERT
        # session.add(model_instance): ORM 모델을 세션에 추가 (아직 DB 반영 X).
        session.add(Document(
            tenant_id=tenant_a,
            filename='iso_test.pdf',
            mime='application/pdf',
            blob_path='blob://test/iso_test.pdf',
        ))
        # await session.commit(): 트랜잭션 커밋. 이 시점에 실제 INSERT가 DB에 반영됨.
        #                         await로 I/O 끝까지 대기.
        await session.commit()

        # B 조회 — 격리 규약대로 tenant_id WHERE를 명시 (rag/models.py 규약 참조).
        # await session.execute(stmt): SELECT 실행, Result 객체 반환.
        result_b = await session.execute(
            select(Document).where(Document.tenant_id == tenant_b)
        )
        # result.scalars(): row 전체 대신 첫 컬럼(여기선 Document 객체)만 추출.
        # .all(): 결과를 리스트로 (지연된 iterator를 즉시 소비).
        rows_b = result_b.scalars().all()
        # assert: 조건이 False면 AssertionError 발생 -> pytest가 FAIL로 잡음.
        assert rows_b == [], '테넌트 B는 없어야하는데 있네?'

        # A 조회 — positive control.
        # SELECT 빌더는 .where()로 추가 조건 체이닝 가능.
        result_a = await session.execute(
            select(Document)
            .where(Document.tenant_id == tenant_a)
            .where(Document.filename == 'iso_test.pdf')
        )
        rows_a = result_a.scalars().all()
        assert len(rows_a) == 1, f"테넌트 A로 조회했는데 row 못 봄: {rows_a}"
    finally:
        # 정리 — 다음 테스트가 깨끗한 상태에서 시작하도록.
        # delete(Model).where(...): SELECT가 아닌 DELETE 문 빌더.
        await session.execute(delete(Document).where(Document.tenant_id == tenant_a))
        await session.commit()
