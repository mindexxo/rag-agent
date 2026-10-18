"""캡션 프롬프트 제거 실험·비교 (#247 결정 13·14) — 운영과 같은 VLM에 크롭을 직접 보내 프롬프트 변형별 캡션을 만들고 채점한다.

색인은 건드리지 않는다(재색인 없이 프롬프트만 비교). 크롭은 extract 단계가 남긴 것 — PDF는 운영 VLM 입력과 같은 배수(4)로
쪽에서 자른 것, DOCX·XLSX는 파일 속 원본이라 docling이 보내는 그림과 같다.

변형(VARIANTS)은 #247 이전 운영 프롬프트(V0)를 한 줄씩 바꿔 가며 효과를 가른다 — 추측으로 고치지 않는다. 채택안은
V13(결정 15, 10/18)이고 지금 rag/chunking._PICTURE_CAPTION_PROMPT와 같다(테스트가 강제). 채점(score_variant)은 합성 그림 정답(picture_specs)으로 자동:
  글자·숫자·짝·순서(synthetic.score_picture) + 빈 칸 환각(P08 '2026년 4분기'에 숫자를 지어냈나) + 흐름도 화살표 연결(edges가
  '→'가 든 한 줄에 함께 있나) + 영문 유지(P05의 영문 메뉴).
실문서는 정답이 없어 캡션만 남기고 Claude 판정(judge.judge_one)·사람 확인으로 본다.
"""
import base64
import io
import json
import re
from pathlib import Path

from eval.doc_fidelity.pictures import norm

BASE_LINES = [
    '이미지의 내용을 한국어로 옮겨라.',
    '- 이미지 안의 글자를 하나도 빠뜨리지 말고 원문 그대로 옮긴다. 고쳐 쓰거나 요약하지 않는다.',
    '- 요소들 사이에 관계가 있으면 그 관계를 함께 적는다.',
    '- 관계가 분명하지 않으면 보이는 위치대로 나열만 한다. 관계가 없다는 말은 쓰지 않는다.',
    '- 이미지에 없는 내용은 쓰지 않는다.',
    '- 머리말·맺음말·질문·제안 없이, 첫 줄부터 이미지 안의 글자로 시작한다.',
]
BLANK = '- 값이 비어 있는 칸·막대·점은 지어내지 말고 "값 없음"으로 적는다.'
HIERARCHY = '- 조직도·계층도라면 위에서 아래로, 연결선 하나를 한 줄로 "<상위 이름> → <하위 이름>" 형식으로 적는다.'
CONDITIONAL = [
    '- 흐름도·절차도라면 화살표 하나를 한 줄로 "출발 → 도착" 형식으로 적고, 화살표 위 글자(예/아니오 등)는 "출발 → 도착 (예)"처럼 '
    '괄호로 붙인다. 박스 아래·옆의 설명은 그 박스 이름 뒤에 "박스: 설명"으로 적는다.',
    '- 차트라면 제목·축 이름·단위·범례를 먼저 적고, 값은 "가로축 항목 (계열): 값" 형식으로 항목마다 한 줄씩 적는다.',
]

CONDITIONAL_V2 = [
    '- 흐름도·절차도라면 화살표 하나를 한 줄로 "<출발 박스 이름> → <도착 박스 이름>" 형식으로 적고, 화살표 위 글자(예/아니오 등)가 '
    '있으면 줄 끝에 괄호로 붙인다. 박스 아래·옆의 설명은 "<그 박스 이름>: <설명>" 형식으로, 설명이 속한 박스의 이름을 앞에 붙여 적는다.',
    '- 차트라면 제목·축 이름·단위·범례를 먼저 적고, 값은 "<가로축 항목> (<계열 이름>): <값>" 형식으로 값 하나를 한 줄에 적는다.',
]

CONDITIONAL_V3 = [
    '- 흐름도·절차도라면 화살표 하나를 한 줄로 "<출발 박스 이름> → <도착 박스 이름>" 형식으로 적고, 화살표 위 글자(예/아니오 등)가 '
    '있으면 줄 끝에 괄호로 붙인다.',
    CONDITIONAL_V2[1],
]


def _join(lines):
    return '\n'.join(lines)


VARIANTS = {
    'V0_현행': _join(BASE_LINES),
    'V1_한국어로_제거': _join(['이미지 안의 글자와 내용을 글로 옮겨라.'] + BASE_LINES[1:]),
    'V2_위치나열_제거': _join(BASE_LINES[:3] + BASE_LINES[4:]),
    'V3_빈칸_추가': _join(BASE_LINES[:5] + [BLANK] + BASE_LINES[5:]),
    'V4_기본수정': _join(['이미지 안의 글자와 내용을 글로 옮겨라.'] + BASE_LINES[1:3] + [BASE_LINES[4], BLANK, BASE_LINES[5]]),
    'V5_기본수정+조건부': _join(['이미지 안의 글자와 내용을 글로 옮겨라.'] + BASE_LINES[1:3] + CONDITIONAL + [BASE_LINES[4], BLANK, BASE_LINES[5]]),
    # V1·V2·V4에서 현행 첫 줄·넷째 줄은 빼면 나빠지거나 효과가 없었다(10/18) — 현행을 그대로 두고 더하기만 한다
    'V6_현행+빈칸+조건부': _join(BASE_LINES[:4] + CONDITIONAL + [BASE_LINES[4], BLANK, BASE_LINES[5]]),
    'V7_V6-위치나열': _join(BASE_LINES[:3] + CONDITIONAL + [BASE_LINES[4], BLANK, BASE_LINES[5]]),
    # V6은 '"박스: 설명"'을 글자 그대로 따라 써 '박스: 사업부장 승인 必'이 됐다(채용 흐름도, 10/18) — 자리표시자로 쓴다.
    # 구체적인 예시 문장은 넣지 않는다 — 예시가 출력 형태를 지배한다(실험 규율, #15·#212)
    'V8_V6+자리표시자': _join(BASE_LINES[:4] + CONDITIONAL_V2 + [BASE_LINES[4], BLANK, BASE_LINES[5]]),
    # V8은 박스 설명을 한 칸씩 밀어 붙였다('모집공고: 면접 대상자 선정' — 첫 박스에 설명이 없는 채용 흐름도, 10/18). '바로 아래 설명만'
    # 으로 좁히자(V9 시험) 귀속을 아예 포기했다 — 7B는 박스-설명 공간 귀속을 안정적으로 못 한다. 틀린 귀속은 귀속 없음보다 위험하므로
    # 흐름도 지시는 화살표 연결만 요구한다
    'V10_현행+빈칸+화살표+차트': _join(BASE_LINES[:4] + CONDITIONAL_V3 + [BASE_LINES[4], BLANK, BASE_LINES[5]]),
    # V10의 차트 자리표시자('<가로축 항목>')는 세로축 제목을 항목 자리에 넣게 했다(D01 '문의건수(건): 120') — 차트는 V6 문구를 쓴다
    'V11_현행+빈칸+화살표+차트V6': _join(BASE_LINES[:4] + [CONDITIONAL_V3[0], CONDITIONAL[1]] + [BASE_LINES[4], BLANK, BASE_LINES[5]]),
    # V11은 실문서 화면 캡처에서 토·일 빈 칸에 근무를 지어내고(15736-008), 표형 화면을 마크다운 표로 바꾸며 열을 뒤바꿨다(15736-023) —
    # 어느 지시 탓인지 가르려고 차트 지시를 뺀다
    'V12_현행+빈칸+화살표': _join(BASE_LINES[:4] + [CONDITIONAL_V3[0]] + [BASE_LINES[4], BLANK, BASE_LINES[5]]),
    # 홀드아웃 조직도에서 V11이 계층 묶음을 잃었다(현행은 '영업본부' 아래를 묶었다, 10/18) — 사용자 요청으로 계층형 지시를 더한다.
    # 홀드아웃을 보고 고친 것이라 검증은 새 홀드아웃(HOLDOUT2)으로 한다
    'V13_V11+계층': _join(BASE_LINES[:4] + [CONDITIONAL_V3[0], HIERARCHY, CONDITIONAL[1]] + [BASE_LINES[4], BLANK, BASE_LINES[5]]),
}

# 흐름도 화살표 정답 — (출발, 도착): 캡션의 '→'가 든 한 줄(또는 연속된 '→' 사슬)에 둘이 이 순서로 있어야 한다
EDGES = {
    'P03_flow': [('고장 신고', '원격 점검'), ('원격 점검', '원격 해결'), ('원격 해결', '처리 종료'), ('원격 해결', '보증 기간'),
                 ('보증 기간', '무상 수리'), ('보증 기간', '유상 견적'), ('무상 수리', '수리 검수')],
    'D03_flow': [('재고 이동 요청', '점장 확인'), ('점장 확인', '재고 충분'), ('재고 충분', '출고'), ('재고 충분', '인근 매장'),
                 ('인근 매장', '매장 간 이동'), ('인근 매장', '본사 발주'), ('출고', '입고 확인'), ('입고 확인', '점장 확인')],
}
ENGLISH = {'P05_screen': ['Store Ops Console', 'Dashboard', 'Payroll', 'Shift Code', 'Break Time', 'Submit']}


def edge_hits(caption: str, edges) -> int:
    hit = 0
    lines = [norm(x) for x in (caption or '').split('\n') if '→' in x]
    for a, b in edges:
        na, nb = norm(a), norm(b)
        if any(na in ln and nb in ln and ln.find(na) < ln.rfind(nb) for ln in lines):
            hit += 1
    return hit


def blank_hallucinated(caption: str) -> bool:
    """P08: 집계 전이라 비어 있는 '2026년 4분기'에 숫자를 붙였나. 그 줄(또는 '2026년' 블록의 4분기 항목)에 숫자가 있으면 환각."""
    for line in (caption or '').split('\n'):
        if '4분기' in line and ('2026' in line or line.strip().startswith('4분기')):
            tail = line.split('4분기', 1)[1]
            if '값 없음' in tail or '없' in tail:
                continue
            if re.search(r'\d{2,}', tail):
                if '2026' in line or '2025' not in line:
                    return True
    return False


def ask(url: str, model: str, image_png: bytes, prompt: str, timeout: float = 180) -> str:
    import httpx
    body = {'model': model, 'temperature': 0, 'messages': [{'role': 'user', 'content': [
        {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + base64.b64encode(image_png).decode()}},
        {'type': 'text', 'text': prompt}]}]}
    r = httpx.post(url, json=body, timeout=timeout)
    r.raise_for_status()
    return r.json()['choices'][0]['message']['content'].strip()


def run(targets: list[tuple[str, Path]], variants: dict[str, str], out: Path) -> list[dict]:
    """targets = [(picture_id, crop 경로)] → 변형마다 캡션. 이미 만든 (id, 변형)은 다시 부르지 않는다(중간 재개)."""
    from config import settings
    done = {}
    if out.exists():
        for line in out.read_text().splitlines():
            r = json.loads(line)
            done[(r['picture_id'], r['variant'])] = r
    rows = list(done.values())
    with out.open('a') as fh:
        for pid, crop in targets:
            buf = io.BytesIO()
            from PIL import Image
            Image.open(crop).convert('RGB').save(buf, format='PNG')
            for name, prompt in variants.items():
                if (pid, name) in done:
                    continue
                cap = ask(settings.vlm_caption_url, settings.vlm_caption_model, buf.getvalue(), prompt)
                r = {'picture_id': pid, 'variant': name, 'caption': cap}
                fh.write(json.dumps(r, ensure_ascii=False) + '\n')
                fh.flush()
                rows.append(r)
                print(f'  {pid} {name} {len(cap)}자', flush=True)
    return rows


def score_variant(spec_id: str, spec: dict, caption: str) -> dict:
    from eval.doc_fidelity.synthetic import score_picture
    s = score_picture(spec, caption)
    if spec_id in EDGES:
        s['edges'], s['edges_hit'] = len(EDGES[spec_id]), edge_hits(caption, EDGES[spec_id])
    if spec_id == 'P08_grouped':
        s['blank_hallucinated'] = blank_hallucinated(caption)
    if spec_id in ENGLISH:
        s['english'], s['english_hit'] = len(ENGLISH[spec_id]), sum(norm(t) in norm(caption) for t in ENGLISH[spec_id])
    return s
