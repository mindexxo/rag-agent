#!/bin/bash
# KMS 이미지를 빌드해 사설 레지스트리에 올린다. 저장소 루트의 Dockerfile을 쓴다.
# 사용: KMS_DOCKER_REGISTRY=<레지스트리 호스트:포트> deploy/build-docker.sh <dev|prod> <버전>
set -euo pipefail

if [ $# -ne 2 ]; then
  echo "Usage: KMS_DOCKER_REGISTRY=<host:port> $0 <dev|prod> <version>"
  exit 1
fi
ENVIRONMENT=$1
VERSION=$2
case $ENVIRONMENT in
  dev|prod) ;;
  *) echo "환경은 dev 또는 prod"; exit 1 ;;
esac
: "${KMS_DOCKER_REGISTRY:?KMS_DOCKER_REGISTRY가 필요합니다}"

REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
IMAGE="${KMS_DOCKER_REGISTRY}/isac-cc/cc-app-be-kms-service:${VERSION}-${ENVIRONMENT}"

echo "빌드: ${IMAGE}"
# 앱 서버는 x86_64 — 다른 아키텍처(예: Apple Silicon)에서 빌드해도 서버용 이미지가 나오게 고정한다.
docker build --platform linux/amd64 -t "${IMAGE}" "${REPO_ROOT}"
docker image push "${IMAGE}"
echo "완료: ${IMAGE}"
