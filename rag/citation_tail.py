"""출처 꼬리 분리·해석 (#56) — 스트리밍 본문에서 ««[1,3]»» 꼬리를 걷어낸다 (형식은 #65).

인라인 인용을 없애면서 "이 답변이 실제로 어느 문서를 썼나"는 답변 끝의 출처 꼬리
(structural_tag로 형식 강제, #65)가 나른다. 이 모듈이 하는 일 둘:

  TailSplitter        스트리밍 도중 꼬리가 delta로 새어나가지 않게 분리 (상태 기계)
  resolve_citations   꼬리 번호 목록 → 범위 검증 → SourceCitation 목록

이 모듈이 "무엇이 인용인가"의 단일 정의점이다. 번호↔문서의 순서는
rag/citation_labels.py가 단일 정의점 (프롬프트 조립과 공유).

**근거없음(ungrounded) = resolve_citations 결과가 빈 목록** — 이 규약의 정의점도 여기다 (#61).
옛 판정은 rag/answer_check.py의 문구 부분일치('제공된 문서에서 확인할 수 없')였고, 폐기 사유는
프롬프트 문구에 묶여 있었다는 것이다: 생성 프롬프트 규칙 3을 완화하니 검출률이 **95~97% →
3~7%로 붕괴**했다(#48 조사 실측). 게다가 문구가 조금 다른 부재 단정("해외 배송은 제공되지
않습니다." + 빈 꼬리)을 아예 놓쳤다 — 인용 개수는 그것도 잡는다. 대신 범위가 넓어져서
"근거 없이 확신하며 답한" 경우까지 포함하는데, 그건 원래 보고 싶던 신호다.
소비처(rag/service.maybe_cache · routers/stats.py · eval/refusal.py)는 이 문단을 참조만 한다.

검증 원칙(#56): 생성 측 강제는 확률을 낮추는 최적화지 신뢰의 근거가 아니다 —
제약이 붙었든(정상) 떨어졌든(fail-open) 파싱 결과는 항상 유효 범위 번호만 통과한다.
#65가 이 원칙의 값을 실증했다: 옛 정규식은 배포 내내 무효였는데도 잘못된 인용이
사용자에게 나가지 않았다 — 이 파일의 범위 검증이 유일한 방어선이었기 때문이다.
malformed 꼬리는 부분 복구를 시도하지 않고 그 번호만 버린다 — 그럴듯한 복구는 실패를
지표에서 숨긴다(탐지 가능한 오류를 탐지 불가능하게 만들지 않는다).
"""
import re

from rag.citation_labels import TAIL_END, TAIL_START, attachment_display
from schemas.kms import SourceCitation

# 꼬리가 이 길이를 넘도록 END가 안 나오면 꼬리가 아니라 본문으로 판정(오탐 복구).
# 번호 목록은 후보 30~40건 전부 인용해도 ~120자 — 200이면 여유 있는 상한이고,
# 본문 속 우연한 마커(오탐)를 붙잡아 두는 창을 짧게 유지한다.
MAX_TAIL_CHARS = 200


class TailSplitter:
    """스트리밍 청크에서 본문과 출처 꼬리를 분리하는 상태 기계.

    feed(chunk) → 지금 delta로 내보낼 본문 조각. 내보낸 것과 정확히 같은 문자열이
    prose에 누적된다 — "화면에 보인 것 = 저장되는 것" 불변식은 이 동일성에서 나온다.
    마커가 청크 경계에서 쪼개져 도착해도 새지 않도록, 본문 끝의 "마커 접두일 수 있는"
    부분(최대 len(TAIL_START)-1자)은 다음 청크가 올 때까지 보류한다.

    오탐 복구: 마커를 봤지만 ① END 없이 MAX_TAIL_CHARS를 넘거나 ② END 뒤에 텍스트가
    더 오면(진짜 꼬리는 스트림 마지막) 버퍼 전체를 본문으로 되돌린다 — 마지막 마커만 인정.
    취소·max_tokens로 꼬리가 잘리면 truncated=True, 잘린 버퍼는 본문·저장 어디에도 안 간다.
    """

    def __init__(self):
        self._hold = ''          # 본문 중 미방출분 (마커 접두 후보)
        self._tail_buf = ''      # 마커 이후 흡수분 (TAIL_START 제외)
        self._in_tail = False
        self._done = False
        self.prose = ''          # 방출한 본문 전체 — feed 반환값의 누적과 항상 동일
        self.tail_raw: str | None = None   # finish 시 확정 — 완결된 꼬리 내부(마커 제외)
        self.truncated = False   # 꼬리가 시작됐지만 END 전에 스트림이 끝남

    def feed(self, chunk: str) -> str:
        assert not self._done, 'finish() 이후 feed 불가'
        emitted = self._feed_tail(chunk) if self._in_tail else self._feed_body(chunk)
        # prose 누적은 여기 최상위 한 곳뿐이다 — 내부 헬퍼(_feed_body/_flush_false_alarm)가
        # 각자 더하면 오탐 복귀가 같은 feed 호출 안에서 재귀할 때 이중 누적된다(리뷰 발견).
        self.prose += emitted
        return emitted

    def _feed_body(self, chunk: str) -> str:
        self._hold += chunk
        out = []
        idx = self._hold.find(TAIL_START)
        if idx != -1:
            out.append(self._hold[:idx])
            self._in_tail = True
            rest = self._hold[idx + len(TAIL_START):]
            self._hold = ''
            out.append(self._feed_tail(rest))
        else:
            # 끝부분이 TAIL_START의 접두이면 그만큼만 보류하고 나머지는 방출
            keep = 0
            max_keep = min(len(self._hold), len(TAIL_START) - 1)
            for k in range(max_keep, 0, -1):
                if TAIL_START.startswith(self._hold[-k:]):
                    keep = k
                    break
            if keep:
                out.append(self._hold[:-keep])
                self._hold = self._hold[-keep:]
            else:
                out.append(self._hold)
                self._hold = ''
        return ''.join(out)

    def _feed_tail(self, chunk: str) -> str:
        self._tail_buf += chunk
        end = self._tail_buf.find(TAIL_END)
        if end != -1:
            rest = self._tail_buf[end + len(TAIL_END):]
            # END 뒤 비공백 텍스트 — 진짜 꼬리는 스트림 마지막이므로 오탐이었다.
            # 공백·개행만 오는 것은 허용(모델·토크나이저가 끝에 공백류를 붙이는 관행) — 폐기.
            if rest.strip():
                return self._flush_false_alarm()
        elif len(self._tail_buf) > MAX_TAIL_CHARS:
            return self._flush_false_alarm()
        return ''

    def _flush_false_alarm(self) -> str:
        """마커째 본문으로 되돌린다 (본문 속 우연한 마커).

        선두 마커는 방금 오탐으로 판정한 것이라 **재스캔 없이** 본문으로 확정 방출하고
        (재스캔하면 같은 마커에 다시 걸려 무한 재귀), 나머지만 재스캔한다 — 그 안에서
        새로 시작하는 마커는 아직 판정 전이므로 정상 후보다.
        """
        flushed, self._tail_buf, self._in_tail = self._tail_buf, '', False
        return TAIL_START + self._feed_body(flushed)

    def finish(self) -> str:
        """스트림 종료 확정 — tail_raw/truncated를 판정하고, 마커 접두인 줄 알았던
        본문 끝자락(보류분)이 있으면 반환한다. **호출자는 반환값을 feed 반환값과 똑같이
        방출해야** prose == 화면 == 저장의 동일성이 유지된다."""
        self._done = True
        if self._in_tail:
            end = self._tail_buf.find(TAIL_END)
            if end != -1 and not self._tail_buf[end + len(TAIL_END):].strip():
                self.tail_raw = self._tail_buf[:end]   # 후행 공백류는 허용·폐기
            else:
                self.truncated = True   # END 전에 잘림(취소·max_tokens) — 버퍼는 폐기
            return ''
        flushed, self._hold = self._hold, ''
        self.prose += flushed
        return flushed


def resolve_citations(tail_raw: str | None, sources, attachment_filenames: list[str]) -> list[SourceCitation]:
    """꼬리의 번호 목록 → 인용 객체. 범위 밖·중복·비숫자는 버린다.

    번호 불변식(citation_labels): 후보 i(1-based) = sources[i-1], 이어서 첨부 순서 —
    프롬프트 컨텍스트 블록의 [번호]와 같은 규칙이므로, 여기 sources는 반드시
    프롬프트를 만든 그 청크의 sources_from_chunks 산출물(prepared.sources)이어야 한다.

    파싱은 숫자 뭉치 추출(\\d+)만 — 구분자가 쉼표든 공백이든 관용한다(fail-open 경로의
    자유 생성 대비). 구 라벨 방식이 필요로 했던 NFC 방어·대괄호 파일명 처리(b83dcd3)는
    숫자에는 해당 없음. 반환 순서는 꼬리 등장 순서.

    첨부 인용은 FAQ 선례(document_id=None인 가짜 인용)를 따라
    SourceCitation(document_id=None, filename='첨부: 파일명', version=1)이 된다 —
    cited_docs·stats 집계에 그대로 노출되는 것이 의도다(첨부 기반 답변도 지표에 잡히게).
    """
    if not tail_raw:
        return []
    candidates: list[SourceCitation] = list(sources)
    candidates += [SourceCitation(document_id=None, filename=attachment_display(n), version=1)
                   for n in attachment_filenames]
    cited: list[SourceCitation] = []
    seen: set[int] = set()
    for token in re.findall(r'\d+', tail_raw):
        n = int(token)
        if 1 <= n <= len(candidates) and n not in seen:
            seen.add(n)
            cited.append(candidates[n - 1])
    return cited
