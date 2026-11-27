#!/bin/bash
# KMS를 ICCS 앱 서버 한 대에 배포한다. --with-worker를 준 서버만 워커까지 띄우고, 안 준 서버의 워커는 내린다.
# 사용: KMS_DOCKER_REGISTRY=… KMS_SSH_USER=… KMS_SSH_PORT=… deploy/deploy-docker.sh <dev|prod> <서버> <버전> [--with-worker]
# 선택: KMS_API_PORT(기본 8000)
set -euo pipefail

if [ $# -lt 3 ] || [ $# -gt 4 ]; then
  echo "Usage: $0 <dev|prod> <server> <version> [--with-worker]"
  exit 1
fi
ENVIRONMENT=$1
SERVER=$2
VERSION=$3
WITH_WORKER=0
if [ $# -eq 4 ]; then
  [ "$4" = "--with-worker" ] || { echo "네 번째 인자는 --with-worker만 허용"; exit 1; }
  WITH_WORKER=1
fi
case $ENVIRONMENT in
  dev|prod) ;;
  *) echo "환경은 dev 또는 prod"; exit 1 ;;
esac
: "${KMS_DOCKER_REGISTRY:?KMS_DOCKER_REGISTRY가 필요합니다}"
: "${KMS_SSH_USER:?KMS_SSH_USER가 필요합니다}"
: "${KMS_SSH_PORT:?KMS_SSH_PORT가 필요합니다}"
KMS_API_PORT=${KMS_API_PORT:-8000}

IMAGE="${KMS_DOCKER_REGISTRY}/isac-cc/cc-app-be-kms-service:${VERSION}-${ENVIRONMENT}"
SERVER_DIRECTORY="/app/isac-cc-application/backend/kms"
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

ssh -p "${KMS_SSH_PORT}" "${KMS_SSH_USER}@${SERVER}" "mkdir -p ${SERVER_DIRECTORY}"
scp -P "${KMS_SSH_PORT}" "${SCRIPT_DIR}/docker-compose.yml" "${KMS_SSH_USER}@${SERVER}:${SERVER_DIRECTORY}/docker-compose.yml"

ssh -p "${KMS_SSH_PORT}" "${KMS_SSH_USER}@${SERVER}" bash -s <<EOF
set -euo pipefail
cd ${SERVER_DIRECTORY}

# 사전 조건 — 하나라도 없으면 띄우기 전에 멈춘다(deploy/README.md 사전 준비)
[ -f kms.env ] || { echo "kms.env가 없습니다: ${SERVER_DIRECTORY}/kms.env"; exit 1; }
docker network inspect isac-cc-application-network >/dev/null 2>&1 || { echo "도커 네트워크 isac-cc-application-network가 없습니다"; exit 1; }
[ -d /data/nas/cc-app/kms/blob ] || { echo "NAS 폴더 /data/nas/cc-app/kms/blob이 없습니다"; exit 1; }

# compose 치환값 — 수동 재기동(docker compose up -d)도 같은 이미지·포트를 쓰게 파일로 남긴다
printf 'KMS_IMAGE=%s\nKMS_API_PORT=%s\n' "${IMAGE}" "${KMS_API_PORT}" > .env

if docker compose version >/dev/null 2>&1; then DC="docker compose"; else DC="docker-compose"; fi

if [ ${WITH_WORKER} -eq 1 ]; then
  \$DC --profile worker pull
  \$DC --profile worker up -d
else
  \$DC pull kms-api
  \$DC up -d kms-api
  # 워커를 다른 서버로 옮긴 경우 이 서버의 워커를 내린다
  \$DC --profile worker rm -sf kms-worker
fi

# API 헬스체크 대기(최대 120초)
HEALTHY=0
for i in \$(seq 1 60); do
  STATUS=\$(docker inspect -f '{{.State.Health.Status}}' kms-api 2>/dev/null || echo none)
  [ "\$STATUS" = "healthy" ] && { HEALTHY=1; break; }
  sleep 2
done
[ \$HEALTHY -eq 1 ] || { echo "kms-api가 120초 안에 healthy가 되지 않았습니다 — docker logs kms-api 확인"; exit 1; }
echo "kms-api healthy"

# 워커는 HTTP가 없어 헬스체크 대신 "20초 동안 살아 있고 재시작하지 않았는가"를 본다 — 설정 오류는 기동 직후 죽는다
if [ ${WITH_WORKER} -eq 1 ]; then
  BEFORE=\$(docker inspect -f '{{.RestartCount}}' kms-worker)
  sleep 20
  STATE=\$(docker inspect -f '{{.State.Running}} {{.RestartCount}}' kms-worker)
  [ "\$STATE" = "true \$BEFORE" ] || { echo "kms-worker가 안정적으로 뜨지 않았습니다(running·재시작: \$STATE) — docker logs kms-worker 확인"; exit 1; }
  echo "kms-worker running"
fi
EOF

echo "배포 완료: ${SERVER} (${IMAGE}, worker=${WITH_WORKER})"
