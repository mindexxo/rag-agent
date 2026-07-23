import asyncio
from sqlalchemy import select
from database import AsyncSessionLocal
from rag.models import Chunk, Document

# trap별 의심 키워드 — 코퍼스에 이게 잡히면 "사실은 근거 있음 → trap 오류" 의심
TRAPS = {
    'tr001 첫구매할인': ['첫 구매', '신규 가입', '첫구매', '웰컴'],
    'tr002 리뷰포인트': ['리뷰', '후기', '작성 시'],
    'tr003 VIP환불빠름': ['VIP', '등급', '우선 처리', '빠르게'],
    'tr004 설치기사': ['설치', '방문 서비스', '기사 방문'],
    'tr005 선물포장': ['선물', '포장'],
    'tr006 학생할인': ['학생', '할인'],
    'tr007 쿠폰양도': ['쿠폰', '양도'],
    'tr008 포인트선물': ['포인트', '가족', '선물', '이전'],
    'tr009 업그레이드교환': ['차액', '대체 상품', '동급', '업그레이드'],
    'tr010 전화주문': ['전화', '주문 접수'],
    'tr011 배송일지정': ['배송 날짜', '지정', '예정일'],
    'tr012 당일배송': ['당일', '익일', '배송'],
    'tr013 냉장반품': ['냉장', '신선', '식품', '위생'],
    'tr014 1+1반품': ['1+1', '행사', '묶음', '증정'],
    'tr015 등급양도': ['등급', '양도', '이전'],
    'tr016 탈퇴위로금': ['위로금', '보상', '탈퇴'],
    'tr017 앱설치포인트': ['앱', '설치', '가입'],
    'tr018 후기등급': ['후기', '등급', '상향'],
    'tr019 생체인증': ['지문', '얼굴', '생체', 'OTP'],
    'tr020 단골면제': ['단골', '면제', '무료', '배송비'],
}

async def main():
    async with AsyncSessionLocal() as s:
        rows = (await s.execute(
            select(Chunk.text, Document.filename).join(Document, Chunk.document_id == Document.id)
            .where(Chunk.tenant_id == 'demo', Document.is_active.is_(True))
        )).all()
    for label, kws in TRAPS.items():
        hits = []
        for kw in kws:
            for text, fn in rows:
                if kw in text:
                    # 키워드가 등장한 문맥 한 줄
                    idx = text.find(kw)
                    ctx = text[max(0, idx-15):idx+25].replace('\n', ' ')
                    hits.append(f'[{kw}] {fn}: ...{ctx}...')
                    break
        print(f'\n● {label}')
        if hits:
            for h in hits[:4]:
                print('   ', h)
        else:
            print('    (키워드 무출현 → 명확한 부재)')

asyncio.run(main())
