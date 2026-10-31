"""청킹 예외 — 베이스 타입이 outbox의 재시도 분류다(ValueError=결정적, ConnectionError=일시)."""


class TableTooLargeError(ValueError):
    """PDF·DOCX·MD 표 하나가 TABLE_MAX_TOKENS를 넘는다 — 결정적 실패(ValueError → outbox.is_transient가 바로 failed).

    메시지가 status_reason으로 화면에 나가므로 엑셀 거절과 같이 토큰 대신 허용 크기 대비 %와 행×열로 말한다.
    엑셀처럼 업로드 순간 거절하지 못하는 것은 표 크기를 docling 변환 뒤에야 알 수 있어서다.
    """

    def __init__(self, page: int | None, header: list[str], percent: int, rows: int, cols: int) -> None:
        self.page, self.percent, self.rows, self.cols = page, percent, rows, cols
        name = ' | '.join(h for h in header if h.strip())[:30]
        where = f"{page}쪽 표" if page else "표"
        super().__init__(
            f"{where}('{name}')가 허용 크기의 {percent}%입니다(현재 {rows}행 × {cols}열). "
            "상담에 필요 없는 열이나 긴 설명을 줄여 다시 올려 주세요."
        )


class PictureLimitError(ValueError):
    """캡션 대상 그림이 상한(settings.docling_max_pictures)을 넘는다 — 결정적 실패(다시 돌려도 그림 수는 같다)."""


class CaptionFailedError(ConnectionError):
    """그림 캡션 요청이 실패했다 — 일시 실패(ConnectionError → outbox가 백오프 재시도).

    docling은 VLM 오류(타임아웃·5xx·연결 거부)를 로그만 남기고 빈 캡션을 돌려준다 — 그대로 두면 그림 내용이 빠진
    문서가 조용히 ready가 된다. 알려진 오분류: 특정 그림에만 나는 결정적 4xx도 MAX_ATTEMPTS까지 재시도된다.
    """


class LlmReadFailedError(ConnectionError):
    """상용 OCR(OCR_PROVIDER=llm)의 LLM 읽기가 빈 응답이다 — 일시 실패(CaptionFailedError와 같은 규칙).

    docling의 api_image_request는 429·5xx·401을 로그만 남기고 상태 코드 없이 빈 글자로 돌려준다. 알려진 오분류:
    401처럼 다시 해도 안 되는 실패도 MAX_ATTEMPTS까지 돌고, 정말 빈 스캔 쪽(백지)도 실패로 본다.
    """
