"""코퍼스 문서를 폴더로 분류하고 '참조 설명'을 채운다 (2026-08-05).

폴더 설명은 리랭커 입력에 붙어 **폴더 사이 순위**에 개입한다(rag/index_text 참조).
그래서 설명은 짧고 변별적이어야 한다 — 길면 후보 수만큼 누적돼 지연이 늘고,
뭉뚱그리면 폴더를 나눈 의미가 없다.

분류 기준은 **파일명 + 문서 제목**(첫 청크의 heading_path[0]) 키워드다.
파일명만 보면 놓친다 — goodpeople은 파일명에 '정책/규정'이 없는데 제목은 전부 "... 규정"이라
안내 폴더로 쏠렸다. 코퍼스가 규칙적이라 가능한 방식이고, 실문서에는 그대로 못 쓴다
(그때는 업로더가 폴더를 고른다).

특히 **상담 스크립트를 별도 폴더로 분리**하는 것이 목적 중 하나다 —
구어체 질의와 문체가 닮아 상위로 올라오지만 정책 근거가 아닌 문서라,
설명에 그 사실을 적어 리랭커가 구분할 수 있게 한다.

실행:
    python -m eval.assign_folders                 # dry-run (기본)
    python -m eval.assign_folders --apply
    python -m eval.assign_folders --apply --tenant summers
    python -m eval.assign_folders --clear         # 배정·폴더 원복
"""
import argparse
import asyncio
from collections import defaultdict

from sqlalchemy import delete, select, update

from database import AsyncSessionLocal
from rag.models import Chunk, Document, Folder

V2_TENANTS = ["adererror", "aromanica", "goodpeople", "harim", "homeplus", "summers"]

# (폴더명, 참조 설명) — 설명은 30~50자를 목표로. 리랭커 입력에 매 후보마다 붙는다.
FOLDERS = {
    "policy": ("정책·규정", "환불·반품·교환·배송·품질 보증의 공식 기준과 조건"),
    "guide":  ("제품·이용 안내", "제품 사용법과 서비스 이용 절차 안내"),
    "script": ("상담 응대 스크립트", "상담원 응대 화법 예시 — 정책의 근거 문서가 아님"),
    "table":  ("요금·기준표", "금액·품목·일정 등 수치를 정리한 표"),
}

# 파일명 키워드 → 폴더 키. 위에서부터 먼저 걸리는 것을 채택한다.
# ⚠ 확장자(xlsx) 판정이 키워드보다 **먼저**다 — '기준'이 '상품관리기준표.xlsx'·'프로모션기준표.xlsx'를
#   정책으로 끌어가는 오분류가 있었다. 표는 형식으로 먼저 확정한다.
POLICY_WORDS = ("정책", "기준", "보증", "판정", "대응", "규정", "안전관리")
GUIDE_WORDS = ("안내", "가이드", "사용법")
SCRIPT_WORDS = ("상담스크립트", "상담_코드표")


def classify(filename: str, title: str = "") -> str:
    """script → table → guide → policy 순. **guide가 policy보다 먼저**인 게 요점.

    '세척·사용 안내 기준'처럼 두 성격의 낱말이 같이 든 제목이 흔한데, 정책을 먼저 보면
    사용 안내 문서가 정책 폴더로 가고 "환불·반품·교환의 공식 기준" 같은 **무관한 설명**이
    리랭커 입력에 붙는다. 설명이 틀리면 안 붙이느니만 못하므로 안내 쪽에 우선권을 준다.
    (파일명에 '정책'이 명시된 건 그대로 정책 — 아래 예외 참조)
    """
    hay = f"{filename} {title}"
    if any(w in hay for w in SCRIPT_WORDS):
        return "script"
    if filename.lower().endswith(".xlsx"):
        return "table"
    if "정책" in filename:                       # 파일명이 '...정책'이면 안내 낱말이 섞여도 정책
        return "policy"
    if any(w in hay for w in GUIDE_WORDS):
        return "guide"
    if any(w in hay for w in POLICY_WORDS):
        return "policy"
    return "guide"


async def main() -> None:
    ap = argparse.ArgumentParser(description="코퍼스 폴더 분류")
    ap.add_argument("--apply", action="store_true", help="실제 반영 (미지정 시 dry-run)")
    ap.add_argument("--tenant", help="특정 테넌트만")
    ap.add_argument("--clear", action="store_true", help="배정 해제 + 생성한 폴더 삭제")
    args = ap.parse_args()

    tenants = [args.tenant] if args.tenant else V2_TENANTS
    names = {name for name, _ in FOLDERS.values()}

    async with AsyncSessionLocal() as session:
        if args.clear:
            for t in tenants:
                ids = (await session.execute(
                    select(Folder.id).where(Folder.tenant_id == t).where(Folder.name.in_(names))
                )).scalars().all()
                if not ids:
                    continue
                await session.execute(
                    update(Document).where(Document.folder_id.in_(ids)).values(folder_id=None))
                await session.execute(delete(Folder).where(Folder.id.in_(ids)))
                print(f"{t}: 폴더 {len(ids)}개 제거 + 배정 해제")
            if args.apply:
                await session.commit()
                print("\n원복 완료")
            else:
                print("\n(dry-run — --apply 필요)")
            return

        plan: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
        for t in tenants:
            rows = (await session.execute(
                select(Document, Chunk.heading_path, Chunk.text)
                .outerjoin(Chunk, (Chunk.document_id == Document.id) & (Chunk.chunk_index == 0))
                .where(Document.tenant_id == t)
                .where(Document.is_active.is_(True))
                .where(Document.status == "ready")
                .order_by(Document.filename)
            )).all()
            if not rows:
                continue

            folder_ids = {}
            for key, (name, desc) in FOLDERS.items():
                f = (await session.execute(
                    select(Folder).where(Folder.tenant_id == t).where(Folder.name == name)
                )).scalars().first()
                if f is None:
                    f = Folder(tenant_id=t, name=name, description=desc, is_searchable=True)
                    session.add(f)
                    await session.flush()
                else:
                    f.description = desc
                    f.is_searchable = True        # 배정 후 off면 문서가 검색에서 통째로 빠진다
                folder_ids[key] = f.id

            for d, heading, text in rows:
                title = (heading or [""])[0] or (text or "").lstrip().split("\n", 1)[0][:60]
                key = classify(d.filename, title)
                plan[t][FOLDERS[key][0]].append(f"{d.filename}   〔{title[:34]}〕")
                d.folder_id = folder_ids[key]

        for t, groups in plan.items():
            print(f"\n[{t}]")
            for name in (n for n, _ in FOLDERS.values()):
                files = groups.get(name, [])
                if files:
                    print(f"  {name} ({len(files)})")
                    for fn in files:
                        print(f"      {fn}")

        total = sum(len(v) for g in plan.values() for v in g.values())
        if args.apply:
            await session.commit()
            print(f"\n반영 완료 — 문서 {total}건 / 테넌트 {len(plan)}개")
            print("⚠ 리랭커 입력이 바뀌었으므로 검색축 재측정 필요 (재색인은 불필요)")
        else:
            await session.rollback()
            print(f"\n(dry-run) 문서 {total}건 / 테넌트 {len(plan)}개 — 반영하려면 --apply")


if __name__ == "__main__":
    asyncio.run(main())
