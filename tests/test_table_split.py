"""표 블록 분할 — 행 묶음·헤더 반복·쪽 넘김 잇기·표 정보와 크기 상한·표 앞뒤 글(rag.chunking.split 중심). docling 없이 돈다."""

import pytest


def _h(text, page=1):
    return ('section_header', text, page, None)


def _t(text, page=1):
    return ('text', text, page, None)


# ── 표 블록 분할·쪽 넘김 잇기·칸 글자 정리 ─────────────────────────────────────


class TestTableBlocks:
    def _long(self, n=45):
        return [['번호', '품목', '단가']] + [
            [str(i), f'품목-{i:03d}', f'{i * 1234 % 97000 + 1000:,}원'] for i in range(1, n + 1)
        ]

    def test_긴_표는_행_묶음으로_나뉘고_묶음마다_헤더가_반복되며_행이_잘리지_않는다(self):
        # 표를 그냥 SentenceSplitter에 맡기면 행 중간에서 쪼개지고 뒷 청크에 헤더가 없다
        from llama_index.core.node_parser import SentenceSplitter

        from rag.chunking import Section, TableBlock, to_markdown_table
        from rag.chunking.split import split_section

        rows = self._long()
        block = TableBlock(rows)
        sec = Section(['문서'], 1, '머리말\n' + block.render(), None, blocks=['머리말', block])
        chunks = split_section(sec, SentenceSplitter(chunk_size=512, chunk_overlap=50))
        tables = [
            t.removeprefix('머리말\n') for t, *_ in chunks if '| 번호 |' in t
        ]  # 큰 표는 묶음마다 표 위 한 줄이 붙는다
        assert len(tables) >= 2  # 1,000토큰 예산이라 둘 이상
        body_lines = [ln for t in tables for ln in t.split('\n')[2:]]  # 헤더·구분선 뺀 데이터 행
        assert body_lines == to_markdown_table(rows[0], rows[1:]).split('\n')[2:]  # 행이 하나도 안 잘리고 순서 그대로
        assert all(t.split('\n')[:2] == to_markdown_table(rows[0], []).split('\n') for t in tables)  # 묶음마다 헤더

    def test_작은_표는_앞뒤_글과_한_청크로_다시_이어진다(self):
        from llama_index.core.node_parser import SentenceSplitter

        from rag.chunking import Section, TableBlock
        from rag.chunking.split import split_section

        block = TableBlock([['a', 'b'], ['1', '2']])
        body = '앞 글.\n' + block.render() + '\n뒤 글.'
        sec = Section(['문서'], 1, body, [1, 1, 1, 2, 2], blocks=['앞 글.', block, '뒤 글.'])
        chunks = split_section(sec, SentenceSplitter(chunk_size=512, chunk_overlap=50))
        assert chunks == [(body, 0, None)]  # 예산 안이면 블록 사이를 다시 잇는다

    def test_쪽을_넘긴_표는_헤더가_같으면_하나로_이어진다(self):
        # docling은 쪽마다 표를 따로 낸다. 바로 이어지고 첫 행이 같으면 뒤 표 헤더를 떼고 잇는다
        from rag.chunking import TableBlock
        from rag.chunking.docling_elements import sections_from_docling_elements

        rows = self._long(5)
        secs = sections_from_docling_elements(
            [_h('제목'), ('table', '', 1, rows[:4]), ('table', '', 2, [rows[0]] + rows[4:])]
        )
        assert [b.rows for b in secs[0].blocks] == [rows]
        assert secs[0].body == TableBlock(rows).render()
        assert secs[0].line_pages[:1] == [1] and len(secs[0].line_pages) == len(rows) + 1

    def test_헤더가_다르거나_사이에_글이_있으면_잇지_않는다(self):
        from rag.chunking import TableBlock
        from rag.chunking.docling_elements import sections_from_docling_elements

        a, b = [['x', 'y'], ['1', '2']], [['x', 'z'], ['3', '4']]
        secs = sections_from_docling_elements([_h('제목'), ('table', '', 1, a), ('table', '', 1, b)])
        assert [x.rows for x in secs[0].blocks] == [a, b]
        secs = sections_from_docling_elements([_h('제목'), ('table', '', 1, a), _t('설명'), ('table', '', 2, a)])
        assert [x.rows if isinstance(x, TableBlock) else x for x in secs[0].blocks] == [a, '설명', a]

    def test_옛_계약_markdown_문자열_표는_글_블록으로_남는다(self):
        from rag.chunking.docling_elements import sections_from_docling_elements

        md = '| 구 분 | 무상수리 |\n| --- | --- |\n| 침수 | 없음 |'
        secs = sections_from_docling_elements([_h('제목'), ('table', '', 1, md)])
        assert secs[0].blocks == [md] and secs[0].body == md

    def test_칸_글자는_줄바꿈을_공백으로_접고_엑셀_자정_꼬리를_뗀다(self):
        from rag.chunking.docling_elements import cell_text_norm

        assert cell_text_norm('1. 요청서 수신\n2. 담당자 배정') == '1. 요청서 수신 2. 담당자 배정'  # C
        assert cell_text_norm('2024-10-02 00:00:00') == '2024-10-02'  # F
        assert cell_text_norm('2024-10-02 09:30:00') == '2024-10-02 09:30:00'  # 실제 시각은 유지
        assert cell_text_norm('구분\xa0(원)') == '구분 (원)'

    def test_묶기는_블록을_잇고_body는_블록에서_파생된다(self):
        from rag.chunking import Section, TableBlock
        from rag.chunking.sections import pack_sections

        block = TableBlock([['a'], ['1']])
        s1 = Section(['문서', '1장', '1.1'], None, '글', None, blocks=['글'])
        s2 = Section(['문서', '1장', '1.2'], None, block.render(), None, blocks=[block])
        (packed,) = pack_sections([s1, s2])
        assert packed.blocks == ['1.1\n글\n1.2', block]
        assert packed.body == '\n'.join(['1.1', '글', '1.2', block.render()])

    def test_이어진_표의_행_묶음은_자기_쪽_번호를_가진다(self):
        # 쪽 넘김 잇기로 한 표가 되면 뒷 쪽 행 묶음이 앞 쪽 번호를 물려받는다 — row_pages로 막는다
        from llama_index.core.node_parser import SentenceSplitter

        from rag.chunking.docling_elements import sections_from_docling_elements
        from rag.chunking.sections import page_at
        from rag.chunking.split import split_section

        rows = self._long(60)
        secs = sections_from_docling_elements(
            [_h('제목'), ('table', '', 1, rows[:31]), ('table', '', 2, [rows[0]] + rows[31:])]
        )
        chunks = split_section(secs[0], SentenceSplitter(chunk_size=512, chunk_overlap=50))
        pages = [page_at(secs[0], off) for _, off, _ in chunks]
        assert pages[0] == 1 and pages[-1] == 2 and pages == sorted(pages)  # 앞 묶음은 1쪽, 뒷 묶음은 2쪽
        assert secs[0].line_pages[:2] == [1, 1] and secs[0].line_pages[-1] == 2

    def test_표_뒤에_오는_글의_오프셋과_목록_번호_복원(self):
        from llama_index.core.node_parser import SentenceSplitter

        from rag.chunking import Section, TableBlock, to_markdown_table
        from rag.chunking.split import split_section

        block = TableBlock([['a', 'b'], ['1', '2']])
        body = block.render() + '\n1. 항목 하나\n2. 항목 둘'
        sec = Section(['문서'], None, body, None, blocks=[block, '1. 항목 하나\n2. 항목 둘'])
        chunks = split_section(sec, SentenceSplitter(chunk_size=512, chunk_overlap=50))
        assert chunks == [
            (body, len(to_markdown_table(['a', 'b'], [])) + 1, None)
        ]  # 표 + 뒤 글이 한 청크, 오프셋은 첫 데이터 행

    def test_그림이_사이에_있으면_잇지_않는다(self):
        from rag.chunking import TableBlock
        from rag.chunking.docling_elements import sections_from_docling_elements

        a = [['x', 'y'], ['1', '2']]
        secs = sections_from_docling_elements(
            [_h('제목'), ('table', '', 1, a), ('picture', '캡션', 1, None), ('table', '', 2, a)]
        )
        assert secs[0].blocks == [TableBlock(a, [1, 1]), '캡션', TableBlock(a, [2, 2])]

    def test_작은_표는_앞_문장과_한_청크에_남는다(self):
        # 블록마다 따로 자르면 '2. 사이즈 스펙 (단위: cm)'만 혼자 남고 표가 맥락을 잃는다
        from llama_index.core.node_parser import SentenceSplitter

        from rag.chunking import Section, TableBlock
        from rag.chunking.split import split_section

        block = TableBlock([['사이즈', '총장'], ['S', '68'], ['M', '70']])
        intro = '2. 품목별 사이즈 스펙 (단위: cm)'
        sec = Section(['문서'], None, intro + '\n' + block.render(), None, blocks=[intro, block])
        chunks = split_section(sec, SentenceSplitter(chunk_size=512, chunk_overlap=50))
        assert [t for t, *_ in chunks] == [intro + '\n' + block.render()] and chunks[0][1] == 0

    def test_큰_표는_묶음마다_표_위_한_줄을_머리에_붙인다(self):
        from llama_index.core.node_parser import SentenceSplitter

        from rag.chunking import Section, TableBlock
        from rag.chunking.split import split_section

        block = TableBlock(self._long(60))
        intro = '본문 문단입니다.\n별첨 1) 품목 단가표'
        sec = Section(['문서'], None, intro + '\n' + block.render(), None, blocks=[intro, block])
        chunks = [t for t, *_ in split_section(sec, SentenceSplitter(chunk_size=512, chunk_overlap=50))]
        tables = [t for t in chunks if '| 번호 |' in t]
        assert len(tables) >= 2 and all(t.startswith('별첨 1) 품목 단가표\n| 번호 |') for t in tables)
        assert chunks[0] == intro  # 앞 글은 따로(큰 표와는 합치지 않음)

    def test_같은_글_블록에서_나온_조각끼리는_합치지_않는다(self):
        from llama_index.core.node_parser import SentenceSplitter

        from rag.chunking import Section
        from rag.chunking.split import split_section

        text = '\n'.join(
            f'{i}번째 문장은 충분히 길어서 분할기가 여러 조각으로 나누게 만든다 반복 반복 반복.' for i in range(60)
        )
        sec = Section(['문서'], None, text, None, blocks=[text])
        splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
        assert [t for t, *_ in split_section(sec, splitter)] == [c for c in splitter.split_text(text)]

    def test_큰_표_위_한_줄뿐인_글은_혼자_남지_않는다(self):
        from llama_index.core.node_parser import SentenceSplitter

        from rag.chunking import Section, TableBlock
        from rag.chunking.split import split_section

        block = TableBlock(self._long(60))
        intro = '[ 별표 1] 품목 단가표'
        sec = Section(['문서'], None, intro + '\n' + block.render(), None, blocks=[intro, block])
        chunks = [t for t, *_ in split_section(sec, SentenceSplitter(chunk_size=512, chunk_overlap=50))]
        assert all(t.startswith(intro + '\n| 번호 |') for t in chunks) and len(chunks) >= 2


# ── 큰 표 묶음의 표 정보·크기 상한 ────────────────────────────────────────────


class TestTableExpansionMeta:
    def _long(self, n=60, head=('번호', '품목', '단가')):
        return [list(head)] + [[str(i), f'품목-{i:03d}', f'{i * 1234 % 97000 + 1000:,}원'] for i in range(1, n + 1)]

    def _sec(self, blocks, page=None):
        from rag.chunking import Section, TableBlock

        body = '\n'.join(b.render() if isinstance(b, TableBlock) else b for b in blocks)
        return Section(['문서'], page, body, None, blocks=blocks)

    def test_큰_표의_묶음은_표_전체를_함께_돌려주고_작은_표는_안_붙인다(self):
        from llama_index.core.node_parser import SentenceSplitter

        from rag.chunking import TableBlock
        from rag.chunking.split import split_section

        big, small = TableBlock(self._long()), TableBlock([['a', 'b'], ['1', '2']])
        pieces = split_section(
            self._sec(['별첨 1) 단가표', big, '설명 문단.', small]), SentenceSplitter(chunk_size=512, chunk_overlap=50)
        )
        refs = [p.table for p in pieces if p.table is not None]
        assert len(refs) >= 2 and len({r.block for r in refs}) == 1  # 묶음 전부 같은 표
        assert refs[0].text == '별첨 1) 단가표\n' + big.render()  # 표 위 한 줄 + 헤더 한 번 + 전체 행
        assert pieces[-1].table is None and small.render() in pieces[-1].text  # 작은 표는 한 조각 안에 이미 다 있다

    def test_chunk_file은_문서_안_표_순번을_table_key로_싣는다(self, monkeypatch, tmp_path):
        import rag.chunking as C
        from rag.chunking import TableBlock

        t1, t2 = TableBlock(self._long()), TableBlock(self._long(head=('번호', '품목', '원가')))
        sections = [self._sec(['1장 본문', t1]), self._sec(['2장 본문', t2, '끝.', TableBlock([['a'], ['1']])])]
        sections[1].heading_path = ['문서', '2장']  # 묶이지 않게 다른 경로
        monkeypatch.setitem(C._SECTION_PARSERS, '.pdf', lambda _p: sections)
        monkeypatch.setattr(C.sections, 'pack_sections', lambda secs: secs)
        chunks = C.chunk_file(tmp_path / 'x.pdf')
        tables = [c for c in chunks if c.meta and c.meta.get('is_table')]
        assert {c.meta['table_key'] for c in tables} == {'t1', 't2'}
        assert {c.meta['table_text'] for c in tables if c.meta['table_key'] == 't2'} == {'2장 본문\n' + t2.render()}
        assert all(
            not (c.meta or {}).get('is_table') for c in chunks if '| a |' in c.text
        )  # 작은 표 청크는 표 묶음이 아니다

    def test_상한을_넘는_표는_TableTooLargeError_결정적_실패(self):
        from llama_index.core.node_parser import SentenceSplitter

        from rag.chunking import TABLE_MAX_TOKENS, TableBlock, TableTooLargeError
        from rag.chunking.split import split_section
        from rag.outbox import is_transient
        from rag.tokens import estimate_tokens_conservative

        rows = self._long(400)
        assert estimate_tokens_conservative(TableBlock(rows).render()) > TABLE_MAX_TOKENS  # 시험 표가 실제로 상한 위
        with pytest.raises(TableTooLargeError) as e:
            split_section(
                self._sec([TableBlock(rows, [7] * len(rows))], page=7),
                SentenceSplitter(chunk_size=512, chunk_overlap=50),
            )
        msg = str(e.value)
        assert msg.startswith("7쪽 표('번호 | 품목 | 단가')가 허용 크기의 ") and '(현재 400행 × 3열)' in msg
        assert '토큰' not in msg  # 거절 문구에 토큰은 쓰지 않는다(엑셀 거절 문구와 같은 규칙)
        assert not is_transient(e.value)  # 재시도 없이 바로 failed

    def test_상한_바로_아래_표는_통과한다(self):
        from llama_index.core.node_parser import SentenceSplitter

        from rag.chunking import TABLE_MAX_TOKENS, TableBlock
        from rag.chunking.split import split_section
        from rag.tokens import estimate_tokens_conservative

        n = 60
        while estimate_tokens_conservative(TableBlock(self._long(n + 1)).render()) <= TABLE_MAX_TOKENS:
            n += 1
        pieces = split_section(
            self._sec([TableBlock(self._long(n))]), SentenceSplitter(chunk_size=512, chunk_overlap=50)
        )
        assert pieces and all(p.table is not None for p in pieces)

    def test_DOCX처럼_쪽이_없으면_표_이름만_말한다(self):
        from rag.chunking import TableTooLargeError

        assert str(TableTooLargeError(None, ['구분', '', '금액'], 150, 500, 3)).startswith(
            "표('구분 | 금액')가 허용 크기의 150%"
        )


# ── 표 전체에 싣는 표 앞뒤 글 ──────────────────────────────────────────────────


class TestTableContext:
    def test_앞_설명은_끝에서_최대_3줄이고_제목에서_멈춘다(self):
        from rag.chunking.split import _table_lead_lines

        text = '제5조(단가)\n첫째 줄.\n둘째 줄.\n셋째 줄.\n넷째 줄.'
        assert _table_lead_lines(text, set()) == ['둘째 줄.', '셋째 줄.', '넷째 줄.']
        assert _table_lead_lines('제5조(단가)\n유일한 설명.', set()) == ['유일한 설명.']  # 조 제목 꼴은 안 가져온다
        assert _table_lead_lines('2. 반품 규정\n설명.', {'2. 반품 규정'}) == ['설명.']  # 기록된 헤딩도 멈춘다

    def test_앞_설명은_합계_200자를_넘기지_않는다(self):
        from rag.chunking.split import _TABLE_CONTEXT_CHARS, _table_lead_lines

        assert _TABLE_CONTEXT_CHARS == 200  # 실문서 설명 문단(196자)을 덮는 최소선
        long = '가' * 150
        assert _table_lead_lines(f'{long}\n{long}', set()) == [long]
        assert _table_lead_lines('가' * 196, set()) == ['가' * 196]
        assert _table_lead_lines('가' * (_TABLE_CONTEXT_CHARS + 1), set()) == []

    def test_아래_주석은_표지로_시작하는_줄이_이어지는_동안만(self):
        from rag.chunking.split import _table_note_lines

        text = '※ 단위는 퍼밀이다.\n* 재검사는 1회.\n주) 부가세 별도.\n(단위: 원)\n불합격 업체는 공지한다.'
        assert _table_note_lines(text) == ['※ 단위는 퍼밀이다.', '* 재검사는 1회.', '주) 부가세 별도.']  # 최대 3줄
        assert _table_note_lines('불합격 업체는 공지한다.\n※ 뒤에 와도 안 가져온다') == []
        assert _table_note_lines('단위: kg당 원\n출처: 구매팀') == ['단위: kg당 원', '출처: 구매팀']
        assert _table_note_lines('**굵은 글씨 문장**') == []  # 마크다운 강조는 표지가 아니다
        assert _table_note_lines(f'※ {"가" * 120}\n※ 짧은 주석') == []  # 합계 100자 상한
        assert _table_note_lines('※ 짧은 주석\n※ ' + '가' * 120) == ['※ 짧은 주석']

    def test_큰_표의_표_전체에만_앞뒤_글이_붙고_묶음_본문은_그대로다(self):
        from llama_index.core.node_parser import SentenceSplitter

        from rag.chunking import Section, TableBlock
        from rag.chunking.split import split_section

        rows = [['번호', '품목', '단가']] + [
            [str(i), f'품목-{i:03d}', f'{i * 1234 % 97000 + 1000:,}원'] for i in range(1, 61)
        ]
        block = TableBlock(rows)
        intro, notes = (
            '아래 단가는 2027년 1월 1일부터 적용한다.\n단가 단위는 kg당 원이다.',
            '※ 부가세는 포함하지 않는다.\n그 밖의 안내 문장.',
        )
        sec = Section(['문서'], None, f'{intro}\n{block.render()}\n{notes}', None, blocks=[intro, block, notes])
        pieces = split_section(sec, SentenceSplitter(chunk_size=512, chunk_overlap=50))
        batches = [p for p in pieces if p.table is not None]
        whole = batches[0].table.text
        assert whole == '\n'.join([*intro.split('\n'), block.render(), '※ 부가세는 포함하지 않는다.'])
        assert all(
            p.text.startswith('단가 단위는 kg당 원이다.\n| 번호 |') for p in batches
        )  # 묶음 머리는 표 위 한 줄만
        assert not any('※' in p.text for p in batches)  # 주석은 묶음 본문에 안 들어간다
        assert any('※ 부가세는 포함하지 않는다.' in p.text for p in pieces if p.table is None)  # 원래 자리에도 그대로
