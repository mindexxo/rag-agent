"""pytest 설정 파일 (rootdir 마커 + 공유 fixture 자리).

이 파일이 있는 디렉토리가 pytest의 rootdir로 인식되고,
그 디렉토리가 sys.path에 자동 추가됨 → 테스트가 `from database import ...` 가능.
"""

import hashlib
import os
import subprocess
import warnings
from pathlib import Path
from urllib.parse import urlsplit

import pytest

# 테스트 환경 격리(#7): 개발자 로컬 .env의 OTEL_ENDPOINT가 스위트에 새어들면
# no-op 계약 테스트가 깨지고 테스트 스팬이 Phoenix로 전송된다.
# config(Settings)가 import되기 전(=pytest가 이 파일을 가장 먼저 읽는 시점)에 비워 고정.
os.environ['OTEL_ENDPOINT'] = ''

# 테스트 인프라는 로컬 컨테이너(docker-compose.test.yml)뿐이다 — .env가 가리키는 개발계 공유 서버를 때리지 않게
# OS 환경변수로 덮는다(설정 우선순위: OS 환경변수 > .env). 포트를 바꾸면 compose 파일도 같이 바꾼다.
TEST_INFRA = {
    'DATABASE_URL': 'postgresql+asyncpg://postgres@127.0.0.1:55432/kms',
    'REDIS_URL': 'redis://127.0.0.1:56379/0',
    'OPENSEARCH_URL': 'http://127.0.0.1:59200',
}
os.environ.update(TEST_INFRA)

COMPOSE_FILE = Path(__file__).resolve().parent / 'docker-compose.test.yml'
LOCAL_HOSTS = ('127.0.0.1', 'localhost')


def _shared_schema_copy() -> Path:
    """schema.sql을 내용 해시로 이름 붙인 워크트리 밖 경로에 복사한다 — DB 컨테이너의 마운트 원본.

    워크트리 경로를 그대로 마운트하면 워크트리마다 compose 설정이 달라 컨테이너가 재생성되고, 다른 워크트리에서
    돌던 테스트의 연결이 끊긴다. 내용이 같으면 경로가 같고, 스키마가 달라졌을 때만 새 경로(= 재생성)가 된다.
    """
    src = Path(__file__).resolve().parent / 'schema.sql'
    data = src.read_bytes()
    dst = Path.home() / '.cache' / 'kms-test' / f'schema-{hashlib.sha256(data).hexdigest()[:16]}.sql'
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(data)
    return dst


@pytest.fixture(autouse=True, scope='session')
def _test_infra() -> None:
    """주소가 로컬이 아니면 스위트를 멈추고, compose로 컨테이너를 띄워 헬스체크 통과까지 기다린다.

    포트 열림으로 판정하지 않는다 — Docker Desktop은 서비스가 뜨기 전에도 포트 연결을 받는다.
    도커가 없거나 기동에 실패하면 경고만 남긴다 — DB 없이 도는 순수 로직 테스트는 그대로 돌고,
    인프라를 쓰는 테스트는 연결 오류로 실패한다."""
    from config import settings

    urls = (settings.database_url, settings.redis_url, settings.opensearch_url)
    remote = sorted({urlsplit(u).hostname or '' for u in urls} - set(LOCAL_HOSTS))
    if remote:
        pytest.exit(
            f'테스트 인프라 주소가 로컬이 아니다: {remote} — 루트 conftest.py의 TEST_INFRA를 확인하라', returncode=2
        )
    cmd = ['docker', 'compose', '-f', str(COMPOSE_FILE), 'up', '-d', '--wait']
    try:
        env = {**os.environ, 'KMS_TEST_SCHEMA': str(_shared_schema_copy())}
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=600, env=env)
    except (OSError, subprocess.SubprocessError) as e:
        detail = getattr(e, 'stderr', '') or e
        warnings.warn(f'테스트 인프라 기동 실패 — 인프라를 쓰는 테스트는 실패한다: {detail}', stacklevel=1)
