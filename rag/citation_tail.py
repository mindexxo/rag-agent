"""출처 꼬리 분리·해석 (#56) — 스트리밍 본문에서 §§CITED§§…§§END§§ 꼬리를 걷어낸다.

인라인 인용을 없애면서 "이 답변이 실제로 어느 문서를 썼나"는 답변 끝의 출처 꼬리
(guided decoding으로 형식 강제)가 나른다. 이 모듈이 하는 일 둘:

  TailSplitter        스트리밍 도중 꼬리가 delta로 새어나가지 않게 분리 (상태 기계)
  resolve_citations   꼬리 라벨 → 후보 교집합 검증 → SourceCitation 목록

rag/answer_check.py에 두지 않은 이유: 그쪽은 "완성된 answer 문자열의 순수 판정"이고
이쪽은 스트리밍 도중 상태를 갖는다 — 성격이 다르다. 라벨 문자열 자체는
rag/citation_labels.py가 단일 정의점 (프롬프트 조립과 공유).

검증 원칙(#56): guided decoding은 확률을 낮추는 최적화지 신뢰의 근거가 아니다 —
문법이 붙었든(정상) 떨어졌든(fail-open) 파싱 결과는 항상 후보 교집합만 통과한다.
malformed 꼬리는 부분 복구를 시도하지 않고 citations=[] — 그럴듯한 복구는 실패를
지표에서 숨긴다(탐지 가능한 오류를 탐지 불가능하게 만들지 않는다).
"""
from rag.citation_labels import TAIL_END, TAIL_START, attachment_label, source_label
from schemas.kms import SourceCitation
from text_norm import nfc

# 꼬리가 이 길이를 넘도록 END가 안 나오면 꼬리가 아니라 본문으로 판정(오탐 복구).
# 후보 30~40건(라벨 ~50자)도 여유 있는 상한.
MAX_TAIL_CHARS = 2000


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
    """꼬리에서 후보 라벨을 찾아 인용 객체로 — 후보 주도 매칭. 후보 밖·malformed는 버린다.

    매칭 주체가 후보다: "꼬리에서 라벨을 추출"(정규식)이 아니라 "각 후보 라벨이 꼬리 안에
    통째로 있는가"(substring)를 본다. 라벨은 우리가 조립하는 문자열이라(citation_labels)
    파일명에 대괄호 등 어떤 문자가 있든 추출 문법이 필요 없다 — 정규식 방식은
    `환불[개정].pdf` 같은 파일명의 인용을 조용히 놓쳤다(중첩 대괄호 추출 불가).
    구 cited_filenames(#41)와 같은 방향 — 후보→출력 매칭이라 오탐이 구조적으로 없다.
    (남는 이론적 엣지: 파일명 자체에 '다른 후보의 라벨 전체'가 포함되면 오탐 가능 —
    guided 경로에선 후보 라벨만 생성 가능해 실질 무해.)

    반환 순서는 꼬리 등장 순서. 첨부 인용은 FAQ 선례(document_id=None인 가짜 인용)를 따라
    SourceCitation(document_id=None, filename='첨부: 파일명', version=1)이 된다 —
    cited_docs·stats 집계에 그대로 노출되는 것이 의도다(첨부 기반 답변도 지표에 잡히게).
    NFC: guided가 붙으면 후보 문자열이 그대로 복사돼 불일치가 없지만, fail-open 경로의
    자유 생성은 정규형이 다를 수 있어 구 cited_filenames(#34)와 같은 방어를 유지한다.
    """
    if not tail_raw:
        return []
    tail = nfc(tail_raw)
    candidates: list[tuple[str, SourceCitation]] = [(source_label(s), s) for s in sources]
    candidates += [(attachment_label(n),
                    SourceCitation(document_id=None, filename=f'첨부: {n}', version=1))
                   for n in attachment_filenames]
    found = [(tail.find(nfc(label)), cit) for label, cit in candidates]
    return [cit for pos, cit in sorted(found, key=lambda x: x[0]) if pos != -1]
