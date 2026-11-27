"""ICCS 앱 서버 배포 구성(deploy/)이 운영 전제를 지키는지 — 파일만 읽는 정적 검사다(도커 불필요)."""

import re
from pathlib import Path

import yaml

from config import Settings

ROOT = Path(__file__).resolve().parent.parent
DEPLOY = ROOT / 'deploy'


def _services() -> dict:
    return yaml.safe_load((DEPLOY / 'docker-compose.yml').read_text(encoding='utf-8'))['services']


def _mounted_container_paths(service: dict) -> list[str]:
    return [v.split(':')[1] for v in service.get('volumes', [])]


def test_INV90_API와_워커는_같은_업로드_원본_경로를_같은_NAS_마운트_아래로_본다():
    api, worker = _services()['kms-api'], _services()['kms-worker']
    blob = api['environment']['BLOB_STORAGE_DIR']
    assert worker['environment']['BLOB_STORAGE_DIR'] == blob
    assert api['volumes'] == worker['volumes']
    assert any(blob.startswith(p.rstrip('/') + '/') for p in _mounted_container_paths(api)), '원본 경로가 마운트 밖이다'
    host_paths = [v.split(':')[0] for v in api['volumes']]
    assert host_paths == _mounted_container_paths(api), (
        '호스트·컨테이너 경로가 같아야 ICCS 서비스와 같은 절대경로를 쓴다'
    )


def test_INV91_운영_이미지에는_개발계_설정이_들어가지_않는다():
    ignored = {line.strip() for line in (ROOT / '.dockerignore').read_text(encoding='utf-8').splitlines()}
    assert {'.env', '.env.dev'} <= ignored
    assert _services()['kms-api']['env_file'] == 'kms.env'


def test_워커는_프로필로만_뜨고_API는_기본으로_뜬다():
    services = _services()
    assert services['kms-worker']['profiles'] == ['worker']
    assert 'profiles' not in services['kms-api']
    assert services['kms-worker']['mem_limit']


def test_API는_서버_밖으로_포트를_열지_않는다():
    for name, svc in _services().items():
        assert 'ports' not in svc, name


def test_재시작_실행사용자_네트워크는_ICCS_관례를_따른다():
    compose = yaml.safe_load((DEPLOY / 'docker-compose.yml').read_text(encoding='utf-8'))
    for name, svc in compose['services'].items():
        assert svc['restart'] == 'unless-stopped', name
        assert svc['user'] == '1001:1002', name
        assert svc['networks'] == ['isac-cc-application-network'], name
    assert compose['networks']['isac-cc-application-network']['external'] is True


def test_이미지는_ccadmin으로_돌고_docling_모델을_내려받은_경로에서_읽는다():
    dockerfile = (ROOT / 'Dockerfile').read_text(encoding='utf-8')
    assert re.search(r'useradd .*-u 1001 -g admin ccadmin', dockerfile)
    assert re.search(r'groupadd -g 1002 admin', dockerfile)
    assert re.findall(r'^USER (\S+)', dockerfile, re.M)[-1] == 'ccadmin'
    artifacts = re.search(r'DOCLING_ARTIFACTS_PATH=(\S+)', dockerfile).group(1)
    assert re.search(rf'docling-tools models download .* -o {re.escape(artifacts)}', dockerfile)
    assert 'HF_HUB_OFFLINE=1' in dockerfile
    assert re.search(rf'chmod -R a\+rX {re.escape(artifacts)}', dockerfile), (
        '모델은 root가 받으므로 ccadmin이 읽을 권한이 필요하다'
    )
    assert 'docling_artifacts_path' in Settings.model_fields, (
        '환경변수가 설정 필드에 닿지 않으면 오프라인에서 모델을 못 찾는다'
    )


def test_헬스체크는_API가_실제로_여는_포트를_본다():
    api = _services()['kms-api']
    port_var = re.search(r'--port.*?(\$\{KMS_API_PORT[^}]*\})', ' '.join(api['command'])).group(1)
    assert port_var in ' '.join(api['healthcheck']['test'])


def _template() -> str:
    return (DEPLOY / 'kms.env.example').read_text(encoding='utf-8')


def test_운영_설정_틀의_키는_전부_config에_있는_이름이다():
    keys = re.findall(r'^(?:# )?([A-Z][A-Z0-9_]*)=', _template(), re.M)  # 주석 처리된 선택 키 포함
    fields = {f.upper() for f in Settings.model_fields}
    assert keys and not [k for k in keys if k not in fields], '오타거나 없는 설정 키'
    assert 'BLOB_STORAGE_DIR' not in keys, 'compose가 고정하는 값 — 서버 파일로 바꿀 수 있으면 API·워커가 어긋난다'
