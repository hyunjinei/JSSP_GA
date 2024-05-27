# JSSP with MIO score

Process의 병렬처리가 불가능해 JSSP 문제에만 활용 가능합니다.

`machine_input_order` 인코딩 방식의 메타휴리스틱 최적화 권장

# 파일 실행 순서
1. data.py에 데이터 연결
2. config.py에서 여러가지 시뮬레이션 관련 변수 설정
3. test_main.py 나 main.py 실행

# Executable Files

1. main.py
2. test_main.py
3. GA_V2.py

# Module Descriptions
  

- GA.py : GA에 필요한 Individual class 포함
- GA_V2.py : pyGAD 라이브러리로 실질적인 GA 프로세스 구현
- main.py
- config.py : 시뮬레이션 관련 세팅
- data.py : Job data ( + 필요할 경우 검증용 solution까지) 입력
