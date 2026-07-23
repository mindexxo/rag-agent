"""pytest 설정 파일 (rootdir 마커 + 공유 fixture 자리).

이 파일이 있는 디렉토리가 pytest의 rootdir로 인식되고,
그 디렉토리가 sys.path에 자동 추가됨 → 테스트가 `from database import ...` 가능.

지금은 비어있고, 향후 여러 테스트가 공유할 fixture를 여기 둘 수 있음.
"""
