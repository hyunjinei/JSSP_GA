# 📊 Hybrid Genetic Algorithm with MIO for Solving JSSP

|                   개발자                   |                 개발자                  |                
| :--------------------------------------: | :-----------------------------------: | 
| [백지원](https://github.com/Jiwon-Baek) | [오현진](https://github.com/hyunjinei) |
|                   🧑‍💻 AI-Development |                 🧑‍💻 AI-Development                  |                

<br>

## 👀 프로젝트 개요

-  📆 프로젝트 기간
   -  2024.05.11 ~ 2024.06.12 (5주)


<br><br>

## 💁‍♂️ 프로젝트 소개


<br>

**Hybrid Genetic Algorithm with MIO**는 JSSP 문제를 해결하기 위한 최적화 알고리즘입니다.
<br><br>
이 프로젝트는 작업 순서와 기계 상태를 추적하며, 작업과 기계 간의 상호 관계를 고려한 MIO를 사용하여 초기 해를 생성합니다. 또한, 다양한 교차, 돌연변이 및 선택 방법을 활용하여 최적의 솔루션을 찾아냅니다. 또한 Local Serach와 결합을 통해 보다 나은 최적해를 찾아 냅니다.

<br>

## 💡 주요 기능

### 1️⃣ 전체 절차

<img src="docs/소개이미지.png"/>

<br>

### 2️⃣ GA 개발

#### 2.1 Population Initialization

- Population Initialization 구현 종류
    - **basic**
        - 랜덤으로 개체 생성
    - **MIO**
        - 작업 순서와 기계 상태를 추적, 작업과 기계 간의 상호 관계를 고려하여 초기 해를 생성
        - 랜덤으로 개체 생성한 후 MIO 사용하여 초기화
    - **GifflerThompson**
        - 랜덤으로 개체 생성한 후 각 개체마다 Giffler-Thompson 우선순위 규칙('SPT', 'LPT', 'MWR', 'LWR', 'MOR', 'LOR', 'EDD') 중 하나를 무작위로 선택하여 초기화

<br>

#### 2.2 Crossover
- Crossover 구현 종류
    - **OrderCrossover (OX)**
        - 유전자의 순서를 유지하는 데 중점
    - **PMXCrossover (PMX)**
        - 교차 구간의 매핑을 기반으로 충돌 해결하며 두 부모 간의 부분 문자열 교환
    - **LOXCrossover (LOX)**
        - 두 부모 간의 유전자 순서를 직선적으로 교차
    - **OrderBasedCrossover (OBC)**
        - 한 부모로부터 선택된 위치의 유전자 순서를 다른 부모에 적용하여 순서 충돌 없이 자손을 생성
    - **PositionBasedCrossover**
        - 랜덤으로 선택된 위치의 유전자로 두 부모에서 유전자를 교환하여 자손을 생성
    - **CycleCrossover (CX)**
        - 부모 간의 순환 구조를 이용해 순환되는 위치에 따라 유전자를 교환
    - **SubstringExchangeCrossover (SXX)**
        - 일반 literal string Encodings에 대한 two cut-points crossover의 종류
    - **PartialScheduleExchangeCrossover (PSX)**
        - 부분 일정을 블록으로 간주하여 교환

<br>

#### 2.3 Mutation
- Mutation 구현 종류
    - **DisplacementMutation**
        - 무작위로 부분 문자열을 선택하고 그것을 무작위 위치에 삽입
    - **InsertionMutation**
        - 무작위로 유전자 하나를 선택하고 그것을 무작위 위치에 삽입
    - **ReciprocalExchangeMutation**
        - 무작위로 두 위치를 선택한 후 이 위치에 있는 유전자들을 교환
    - **ShiftMutation**
        - 무작위로 유전자 하나를 선택하고 그것을 무작위 위치로 이동
    - **InversionMutation**
        - chromosome 내에서 무작위로 두 위치를 선택하고 이 두 위치 사이의 부분 문자열을 반전
    - **SwapMutation**
        - 무작위로 두 위치를 선택하고 이 위치의 유전자들을 교환
    - **GeneralMutation**
        - 해당 유전자 위치를 선택된 다른 유전자와 교환

<br>

#### 2.4 Selection
- Selection 구현 종류
    - **TournamentSelection**
        - 무작위로 선택된 개체 집합에서 토너먼트를 통해 가장 우수한 개체를 선택
    - **SeedSelection**
        - 한 부모는 우수한 개체, 다른 부모는 랜덤으로 선택한 개체 선택
    - **RouletteSelection**
        - 적합도 비례 방식으로 개체를 선택, 적합도가 높은 개체가 선택될 확률이 높아짐

<br>

#### 2.5 Selective Mutation
- 적합도(또는 성능)에 따라 개체를 두 그룹으로 나눔 (상위/하위)
- 각 그룹에 서로 다른 돌연변이 확률을 적용하여 지역 최적해를 방지
    - **pm_high**
        - 높은 돌연변이 확률 (낮은 순위의 개체에 적용)
    - **pm_low**
        - 낮은 돌연변이 확률 (높은 순위의 개체에 적용)
    - **rank_divide**
        - 개체군을 성능에 따라 두 그룹으로 나누는 기준

<br>

#### 2.6 Fitness
- Selection에 사용
- Target makespan을 이용하여 fitness 함수 사용
    - $Fitness = \frac{1}{\frac{Makespan}{Targetmakespan}}$
- Population에서 scaling method 구현
    - **min-max**
    - **rank**
    - **sigma**
    - **boltzmann**

<br>

#### 2.7 Target Makespan
- Target makespan에 도달하면 해당 GA 종료

<br>

### 3️⃣ Elitism
- 엘리트 개체 보장
    - 최악의 개체를 대체
    - Random 대체

<br>

### 4️⃣ Hybrid
#### 4.1 Local Search
- 특정 주기마다 Local Search 이벤트 발생
- 리스트로 들고와서 순차적으로 진행, None 가능
    - **HillClimbing**
        - 현재 해에서 이웃으로 이동하면서 더 나은 해를 찾음
    - **TabuSearch**
        - 탐색 과정에서 이전에 방문한 해를 금지하여 지역 최적해 방지
    - **SimulatedAnnealing**
        - 초기 온도에서 시작하여 서서히 온도를 낮추면서 해를 탐색
    - **GifflerThompson**
        - Giffler-Thompson 우선순위 규칙을 사용하여 makespan과 fitness를 비교

#### 4.2 PSO (Particle Swarm Optimization)
- 모든 generation 완료 후 PSO 진행
    - 입자라고 불리는 후보 Solution 집단을 갖고, 검색 공간에서 입자를 이동시켜 최적화

<br>

### 5️⃣ Island Migration
- 각 Population은 독립적으로 evolution 진행
- 특정 주기마다 migration 이벤트 발생, 각 섬마다 해를 교환
    - **Independent**
        - 각 Population은 독립적으로 evolution 계속 진행
    - **Sequential Migration**
        - A섬의 상위 10%가 B섬의 하위 90%와 교환
    - **Random Migration**
        - A섬의 상위 10%가 랜덤한 섬의 하위 90%와 교환

---

## 🗂 성능 테스트

### 벤치마킹 문제에 대한 성능 테스트
- la 데이터셋 & Ta 데이터셋 & abz5 데이터셋
    - JSSP에 대한 대표적인 벤치마킹 데이터셋
    - 비교 알고리즘
        - **DDQN**
            - 두 개의 Q-네트워크 사용
        - **ACRL35**
            - Actor-critic deep reinforcement
        - **ML-CNN**
            - Multilevel CNN and Iterative Local Search
        - **ILS**
            - Only Iterative Local Search

### 성능 비교 결과

| Problem     | Optimal | DDQN | ML-CNN | ILS  | Non-Local | Proposed |
|-------------|---------|------|--------|------|-----------|----------|
| La01 (10x5) | 666     | 666  | 666    | 666  | 666       | 666      |
| La02 (10x5) | 655     | 655  | 655    | 667  | 688       | 655      |
| La03 (10x5) | 597     | 597  | 603    | 617  | 620       | 597      |
| La04 (10x5) | 590     | 609  | 590    | 590  | -         | 590      |
| La05 (10x5) | 593     | 593  | 593    | 593  | -         | 593      |
| La06 (15x5) | 926     | 926  | 926    | 926  | -         | 926      |
| La07 (15x5) | 890     | 890  | 890    | 890  | -         | 890      |
| Score       | -       | 6    | 6      | 5    | -         | 7        |
| TA21 (20x20)| 1642    | -    | -      | -    | 1952      | 진행중    |
| TA22 (20x20)| 1561    | -    | -      | -    | 1958      | 진행예정 |
| TA31 (30x15)| 1764    | -    | -      | -    | 2112      | 진행예정 |
| Abz5 (10x10)| 1234    | -    | -      | -    | 1338      | 1276     |

<br>

## 📂 파일 설명

### GAS 폴더
1. **run.py**: 실행 파일
2. **GA.py**: GAEngine 클래스에 관한 파일
3. **Individual.py**: Individual 클래스에 관한 파일
4. **Population.py**: Population 클래스에 관한 파일

### environment 폴더, Config 폴더, postprocessing 폴더
1. **environment 폴더 내 파일**: simpy 환경 설정
2. **Config 폴더 파일 내 RunConfig.py**: Run_Config 클래스 파일
3. **postprocessing 폴더 내 파일**: generate_machine_log 함수 파일

### visualization 폴더
1. **GUI.py**: GUI 파일
2. **Gantt.py**: Gantt 파일

### result 폴더
- **result_Gantt 폴더**: 종료 후 Gantt 차트 png 파일 생성
- **result_txt 폴더**: run.py 실행 후 machine 및 전체 csv 생성

### Data 폴더
- 해당 폴더 내의 파일로 사용 가능

<br>

## 🏃 실행 방법
1. GAS 폴더로 이동
2. Run.py 내부 하이퍼 파라미터 조정
    - **TARGET_MAKESPAN**: 목표 Makespan
    - **MIGRATION_FREQUENCY**: 이주 간격
    - **random_seed**: 랜덤 시드
    - main 함수 내 **file = 'filename.txt'** 수정
    - **Run_Config(n_job=50, n_machine=20, n_op=1000, population_size=1000, generations=100)** 수정: 데이터의 job, machine, operation 갯수 수정 및 원하는 population size 및 generation 조정
    - **custom_settings** 내 원하는 GA 갯수로 만들기 및 내부 Crossover, Mutation, Selection, local_search, PSO, selective_mutation 및 확률 조정
    - **local_search_frequency**: local_search 간격
    - **selective_mutation_frequency**: 선택 mutation 간격
3. 파라미터 조정 후 `python run.py`로 실행
4. 실행 후 Random, MIO, heuristic으로 population 초기화 방법 선택
5. Migration 방법 선택 (독립, 순차, 랜덤)
