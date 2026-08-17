"""문자열 유니코드 정규화 정책 (#34).

한글은 같은 글자를 두 방식으로 표현할 수 있다 — 조합형(NFC, `환` 1코드포인트)과
분해형(NFD, `ㅎ`+`ㅘ`+`ㄴ` 3코드포인트). 눈으로는 같지만 문자열 비교는 `False`다.

실측된 사고(#34): 브라우저로 올린 한글 파일명이 NFD로 `documents.filename`에 저장됐고,
LLM은 프롬프트의 그 라벨을 읽어 NFC로 출력했다. 당시 인용 매칭(구 cited_filenames —
본문 라벨 부분일치)이 못 찾아 `cited_docs`가 빈 배열이 되고, FE 각주와 인용 지표가
**에러 없이 조용히** 0이 됐다. #56 이후 매칭은 rag/citation_tail.resolve_citations —
방어 원리는 동일하다.

── 정책 ────────────────────────────────────────────────────
**경계에서 NFC로 정규화한다. 내부에는 NFC만 존재한다고 가정한다.**

비교 지점마다 정규화하면 그 지점을 전부 찾아야 하고 하나만 빠뜨려도 같은 방식으로 조용히
깨진다. 반면 저장값이 한 형태로 통일되면 이후 비교(supersede·exists·인용 매칭·filename
조인)가 자동으로 맞는다.

**대상은 "비교·식별에 쓰이는 텍스트"뿐이다** — 파일명이 사실상 유일하다. NFD는 파일시스템·
File API를 경유할 때 유입되고, 타이핑으로 만들어지는 문자열(폴더명·검색어·질의)은 IME가
NFC를 내므로 위험군이 아니다(#34 감사). 본문·답변·문서 내용은 **정규화하지 않는다** —
원문이 변조되면 인용 매칭·캐시가 되레 어긋난다.

**NFC이지 NFKC가 아니다.** NFKC는 호환 문자까지 바꿔서(`㈜`→`(주)`, `①`→`1`, `ﬁ`→`fi`)
사용자가 올린 파일명이 달라진다. 조합만 정리하는 NFC가 맞다.

예외: LLM 출력처럼 **우리가 통제할 수 없는 입력**과 비교하는 자리는 비교 시점에도 방어한다
(`rag/citation_tail.py`의 `resolve_citations`). 저장값이 NFC여도 모델이 어느 형태로 낼지는
보장이 없다 (guided decoding이 붙으면 후보 복사라 안전하지만, fail-open 경로는 자유 생성).
"""
import unicodedata


def normalize_filename(name: str) -> str:
    """파일명을 NFC로. 경계(업로드·조회·CLI·첨부 스키마)에서 호출한다."""
    return unicodedata.normalize('NFC', name)


def nfc(text: str) -> str:
    """비교 직전 방어용 NFC 정규화 — 통제 불가 입력(LLM 출력)과 맞출 때만 쓴다.
    저장·전달 값을 바꾸는 용도가 아니다 (그건 normalize_filename의 몫)."""
    return unicodedata.normalize('NFC', text)
