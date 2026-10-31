"""테스트 인프라는 로컬 컨테이너뿐이다 — 개발계 공유 서버에 붙지 않는다(INV-08).

주소는 루트 conftest.py의 TEST_INFRA가 OS 환경변수로 고정하고, 컨테이너는 docker-compose.test.yml이 띄운다.
둘이 어긋나면(포트를 한쪽만 바꾸면) 테스트가 엉뚱한 곳에 붙거나 연결에 실패하므로 여기서 맞물림을 본다.
"""

import importlib.util
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parent.parent
# `import conftest`는 tests/conftest.py를 집으므로 루트 파일을 경로로 읽는다
_spec = importlib.util.spec_from_file_location('_root_conftest', ROOT / 'conftest.py')
root = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(root)
COMPOSE = root.COMPOSE_FILE.read_text(encoding='utf-8')


def test_설정이_보는_인프라_주소는_TEST_INFRA_그대로다() -> None:
    from config import settings

    assert settings.database_url == root.TEST_INFRA['DATABASE_URL']
    assert settings.redis_url == root.TEST_INFRA['REDIS_URL']
    assert settings.opensearch_url == root.TEST_INFRA['OPENSEARCH_URL']


def test_TEST_INFRA는_로컬이고_compose가_127_0_0_1에_연_포트와_같다() -> None:
    for key, url in root.TEST_INFRA.items():
        parts = urlsplit(url)
        assert parts.hostname in root.LOCAL_HOSTS, key
        assert f'"127.0.0.1:{parts.port}:' in COMPOSE, key
