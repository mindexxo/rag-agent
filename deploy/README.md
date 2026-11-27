# KMS 배포 — ICCS 앱 서버

KMS(API + arq 워커)를 ICCS 상담 앱 서버에 도커 compose로 올린다. 구성은 ICCS 서비스 관례를 따른다
(사설 레지스트리 이미지 · `isac-cc-application-network` · NAS `/data/nas/cc-app` · `ccadmin:admin`).

- API(`kms-api`): 앱 서버 **2대 모두**. 포트를 서버 밖으로 열지 않는다 — 같은 서버의 customer-service가
  도커 네트워크 안에서 `http://kms-api:<KMS_API_PORT>`로 부른다(인증 없는 API라 노출하지 않는다).
- 워커(`kms-worker`): **1대에만**(`--with-worker`). 업로드 색인(docling 변환, 메모리 상한 3g)과 정리 작업을 돈다.
- 업로드 원본: NAS `/data/nas/cc-app/kms/blob`. DB에 절대경로가 저장되므로 모든 컨테이너가 같은 경로로 본다.

## 사전 준비 (서버마다 1회)

1. 도커 네트워크 `isac-cc-application-network` — ICCS 배포 때 이미 만들어져 있다.
2. NAS 폴더: `mkdir -p /data/nas/cc-app/kms/blob && chown -R 1001:1002 /data/nas/cc-app/kms` (NAS 공유라 한 서버에서만)
3. 앱 설정: `/app/isac-cc-application/backend/kms/kms.env` — `deploy/kms.env.example`을 복사해 채운다.
   빈 값(`KEY=`)은 필수, 주석(`# KEY=`)은 선택 키다 — 빈 값으로 남기면 빈 문자열이 들어가 기동이 실패한다.
   **저장소에 커밋하지 않는다.** 이미지에는 개발계 설정이 없어서, 빠뜨린 키는 개발계가 아니라 localhost로 가서 실패로 드러난다.
4. 서버에서 레지스트리 로그인(`docker login <레지스트리>`) — ICCS 이미지를 받는 계정과 같다.
5. DB 스키마: KMS `schema.sql` 적용(`vector`·`pg_trgm` 확장 필요). 적용한 스키마를 `kms.env`의 `DB_SEARCH_PATH`에 적는다.

## 빌드

```bash
KMS_DOCKER_REGISTRY=<레지스트리 host:port> deploy/build-docker.sh prod 1.0.0
```

이미지 이름: `<레지스트리>/isac-cc/cc-app-be-kms-service:<버전>-<환경>`. 아키텍처는 linux/amd64로 고정한다.
docling 모델(레이아웃·표)이 이미지 안 `/opt/docling-models`에 들어가므로 서버는 모델을 받으러 나가지 않는다.

## 배포

```bash
export KMS_DOCKER_REGISTRY=<레지스트리 host:port> KMS_SSH_USER=<계정> KMS_SSH_PORT=<SSH 포트>
deploy/deploy-docker.sh prod <서버1> 1.0.0 --with-worker   # API + 워커
deploy/deploy-docker.sh prod <서버2> 1.0.0                 # API만
```

- 포트를 바꾸려면 `KMS_API_PORT=<포트>`를 함께 준다(기본 8000). customer-service의 `kms.base-url`도 같은 포트로 맞춘다.
- 스크립트는 사전 준비(kms.env·네트워크·NAS 폴더)를 확인하고, 서버에 compose와 `.env`(이미지·포트)를 둔 뒤 띄우고,
  `kms-api`가 healthy가 될 때까지 최대 120초 기다린다.
- 워커 이전: 새 서버에 `--with-worker`로 배포하고, 이전 서버를 옵션 없이 다시 배포하면 그 서버의 워커가 내려간다.
  워커가 잠깐 두 대가 되어도 정리 작업은 Redis 기준으로 하나만 돈다(메모리만 두 배).

## 운영

서버의 `/app/isac-cc-application/backend/kms`에서:

```bash
docker compose ps
docker logs -f kms-api          # 로그는 표준 출력 — 로테이션 100m × 3
docker compose up -d kms-api    # 재기동(.env의 이미지·포트를 그대로 쓴다)
docker inspect -f '{{.State.OOMKilled}}' kms-worker   # 워커가 메모리 상한으로 죽었는지
```

컨테이너는 `restart: unless-stopped` — 죽거나 서버가 재부팅되면 다시 뜬다(사람이 멈춘 것은 그대로).
