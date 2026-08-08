"""pytest 설정 파일 (rootdir 마커 + 공유 fixture 자리).

이 파일이 있는 디렉토리가 pytest의 rootdir로 인식되고,
그 디렉토리가 sys.path에 자동 추가됨 → 테스트가 `from database import ...` 가능.
"""
import os

# 테스트 환경 격리(#7): 개발자 로컬 .env의 OTEL_ENDPOINT가 스위트에 새어들면
# no-op 계약 테스트가 깨지고 테스트 스팬이 Phoenix로 전송된다.
# config(Settings)가 import되기 전(=pytest가 이 파일을 가장 먼저 읽는 시점)에 비워 고정.
os.environ['OTEL_ENDPOINT'] = ''
