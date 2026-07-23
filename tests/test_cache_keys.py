"""캐시 키·직렬화 단위 테스트 — cache.normalize_query / build_cache_digest / sources 직렬화.

digest는 semantic 캐시(answer_cache.cache_key)의 키 — 정규화가 바뀌면 기존 캐시 전체가 미스가 된다.
"""
from rag.cache import build_cache_digest, normalize_query, sources_from_json, sources_to_json
from schemas.kms import SourceCitation


class TestNormalizeQuery:
    def test_공백_변형_동일화(self):
        assert normalize_query('  환불   언제  돼요  ') == '환불 언제 돼요'

    def test_끝_문장부호_제거(self):
        # rstrip 문자셋("?.!。？！") 전체 커버 — 한 문자라도 빠지면 그 부호의 질의가 다른 digest가 됨
        for punct in ('?', '.', '!', '。', '？', '！', '?!'):
            assert normalize_query(f'환불 언제 돼요{punct}') == '환불 언제 돼요', punct

    def test_중간_물음표는_유지(self):
        assert normalize_query('7일? 아니면 14일') == '7일? 아니면 14일'

    def test_선행_문장부호는_보존(self):
        # rstrip(끝만 제거)이 스펙 — strip으로 바뀌면 선행 부호가 사라져 digest가 달라진다
        assert normalize_query('?환불 되나요') == '?환불 되나요'

    def test_영문_소문자화(self):
        # 절대값 단언 — 자기참조 비교는 lower→upper 뮤테이션을 못 잡는다
        assert normalize_query('VIP 등급') == 'vip 등급'

    def test_빈_문자열(self):
        # 현재 동작 문서화: 빈/공백 질의는 모두 '' 키를 공유한다 (schemas 검증 부재와 연결된 알려진 함정)
        assert normalize_query('') == ''
        assert normalize_query('   ') == ''


class TestCacheDigest:
    def test_동치_질의는_같은_digest(self):
        assert build_cache_digest('환불 언제 돼요?') == build_cache_digest('환불  언제 돼요')

    def test_다른_질의는_다른_digest(self):
        assert build_cache_digest('환불 기간') != build_cache_digest('교환 기간')

    def test_sha256_알고리즘_고정(self):
        # digest 알고리즘·인코딩이 바뀌면 기존 캐시(answer_cache.cache_key) 전체가 미스 — 값 자체를 고정
        import hashlib
        assert build_cache_digest('환불') == hashlib.sha256('환불'.encode('utf-8')).hexdigest()


class TestSourcesRoundTrip:
    def test_직렬화_복원_동일(self):
        sources = [
            SourceCitation(document_id=3, filename='환불정책.pdf', version=2),
            SourceCitation(document_id=None, filename='FAQ', version=1),  # FAQ 인용 — None 보존
        ]
        assert sources_from_json(sources_to_json(sources)) == sources

    def test_여분_키는_무시하고_복원(self):
        # 과거 캐시/메시지 row에 필드가 더 있어도 복원돼야 함 — extra='forbid'로 바뀌는 회귀 방지
        raw = [{'document_id': 1, 'filename': 'a.pdf', 'version': 1, 'score': 0.9}]
        assert sources_from_json(raw) == [SourceCitation(document_id=1, filename='a.pdf', version=1)]
