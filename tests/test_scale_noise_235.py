"""#235 규모 실험 하네스의 순수 로직 — 골드 겹침 필터·작업 목록·규모별 노이즈 포함 관계(DB·OpenSearch 없음)."""
import json

from eval.scale_noise import build, filter as flt
from eval.scale_noise.topics import jobs, prompt

GOLD = [
    {'id': 'inticube_sf001', 'type': 'single_fact', 'query': 'q',
     'expected_chunks': [{'filename': 'a.pdf', 'snippet': '경조사 지원금은 신청일로부터 삼십일 이내에 지급한다'}],
     'expected_points': ['경조사 지원금은 신청 후 30일 이내 지급', '지급 한도는 50만원'], 'expected_answer': ''},
    {'id': 'summers_sf001', 'type': 'single_fact', 'query': 'q',
     'expected_chunks': [{'filename': 'b.md', 'snippet': '단순변심 반품은 칠일 이내 신청한다'}],   # 공백 빼고 15자 이상이어야 규칙 1 대상
     'expected_points': ['반품은 7일 이내', '배송비 3,000원 부담'], 'expected_answer': ''},
]


def test_필터_스니펫_포함은_어느_테넌트_골드든_버린다():
    doc = '# 규정\n\n단순 변심 반품은 칠일 이내 신청한다.'      # 공백 차이는 무시
    assert flt.check(doc, GOLD)['reason'] == 'snippet'


def test_필터_사실_규칙은_측정_테넌트_골드만_보고_숫자_경계를_지킨다():
    # 모의 테넌트(summers)의 흔한 값('7일'·'3,000원')은 핵심어가 같이 있어도 버리지 않는다
    assert flt.check('반품 7일 이내, 배송비 3,000원을 부담한다.', GOLD) is None
    # 측정 테넌트 값 둘 + 핵심어('경조사') → 버림
    assert flt.check('경조사 지원금은 30일 이내 지급하며 한도는 50만원이다.', GOLD)['reason'] == 'facts'
    # 숫자 앞 경계 — '130일'·'150만원'은 '30일'·'50만원'이 아니다
    assert flt.check('경조사 지원금은 130일 이내 지급하며 한도는 150만원이다.', GOLD) is None
    # 값 하나만 겹치면 통과
    assert flt.check('경조사 지원금은 30일 이내 지급한다.', GOLD) is None


def test_작업_목록은_결정적이고_id가_겹치지_않는다():
    a, b = jobs(40), jobs(40)
    assert a == b and len({j['id'] for j in a}) == 40
    assert jobs(100)[:40] == a                      # --n을 늘려도 앞부분은 같다(이어 돌리기)
    assert {j['kind'] for j in a} == {'reg', 'other'}
    assert all('코드펜스 없이' in prompt(j) for j in a)


def test_규모별_노이즈는_같은_순서로_채워_x5가_x10에_포함된다(tmp_path, monkeypatch):
    chunks = tmp_path / 'chunks.jsonl'
    with chunks.open('w', encoding='utf-8') as fp:
        for d in range(12):
            for i in range(3):
                fp.write(json.dumps({'doc': f'r{d:04d}', 'filename': f'회사 규정{d}.md', 'chunk_index': i, 'text': f'제{i}조 본문 {d}',
                                     'heading_path': ['규정'], 'page': None, 'meta': {}, 'dense': [0.0] * 4}) + '\n')
    flt_file = tmp_path / 'filter.json'
    flt_file.write_text(json.dumps({'kept': [f'r{d:04d}' for d in range(10)]}), encoding='utf-8')   # 2편은 필터 탈락
    monkeypatch.setattr(build, 'CHUNKS', chunks)
    monkeypatch.setattr(build, 'FILTER', flt_file)
    small, n_small = build._noise_docs(5, 'inticube')
    big, n_big = build._noise_docs(14, 'inticube')
    assert len(small) == 6 and n_small == 2                     # 문서 단위로 채운다(3조각씩)
    assert len(big) == 15 and n_big == 5
    assert [d['_id'] for d in small] == [d['_id'] for d in big][:6]   # 앞부분이 같다 → x5 ⊂ x10
    assert {d['document_id'] for d in big} < set(range(build.NOISE_DOC_ID_BASE, build.NOISE_DOC_ID_BASE + 12))
    assert all(d['tenant_id'] == 'inticube' and d['searchable'] and d['index_text_nori'].startswith('회사 규정') for d in big)
    assert all(d['document_id'] < build.NOISE_DOC_ID_BASE + 10 for d in big)   # 필터 탈락 문서는 쓰지 않는다
