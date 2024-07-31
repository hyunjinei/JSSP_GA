import os
import sys
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Data.Dataset.Dataset import Dataset
from visualization.Gantt import Gantt
from Config.Run_Config_makespan import Run_Config_Makespan

def calculate_makespan(dataset, job_order):
    num_machines = dataset.n_machine
    completion_times = [[0 for _ in range(num_machines)] for _ in range(len(job_order))]
    
    for i, job in enumerate(job_order):
        for j in range(num_machines):
            machine, processing_time = dataset.op_data[job][j]
            if i == 0 and j == 0:
                completion_times[i][j] = processing_time
            elif i == 0:
                completion_times[i][j] = completion_times[i][j-1] + processing_time
            elif j == 0:
                completion_times[i][j] = completion_times[i-1][j] + processing_time
            else:
                completion_times[i][j] = max(completion_times[i-1][j], completion_times[i][j-1]) + processing_time
    
    return completion_times[-1][-1]

def create_machine_log(dataset, job_order):
    num_machines = dataset.n_machine
    machine_log = []
    completion_times = [[0 for _ in range(num_machines)] for _ in range(len(job_order))]
    start_times = [[0 for _ in range(num_machines)] for _ in range(len(job_order))]
    
    for i, job in enumerate(job_order):
        for j in range(num_machines):
            machine, processing_time = dataset.op_data[job][j]
            if i == 0 and j == 0:
                start_time = 0
            elif i == 0:
                start_time = completion_times[i][j-1]
            elif j == 0:
                start_time = completion_times[i-1][j]
            else:
                start_time = max(completion_times[i-1][j], completion_times[i][j-1])
            
            start_times[i][j] = start_time
            completion_times[i][j] = start_time + processing_time
            
            machine_log.append({
                'Job': f'Part{job}',
                'Machine': f'Machine{machine}',
                'Start': start_time,
                'Finish': completion_times[i][j]
            })
    
    return pd.DataFrame(machine_log), start_times, completion_times

# 데이터셋 로드
file = 'test_506_fixed.txt'
dataset = Dataset(file)

# Run_Config_Makespan 객체 생성
config = Run_Config_Makespan(n_job=dataset.n_job, n_machine=dataset.n_machine, n_op=dataset.n_op)
config.set_dataset_filename(file)

# 주어진 순서대로 makespan 계산 및 Gantt 차트 생성
# job_orders = [
#     list(range(50))  # 50개의 작업을 순서대로 나열
# ]
job_orders = [
     [46, 26, 45, 29, 28, 48, 6, 44, 42, 10, 5, 30, 13, 32, 11, 27, 20, 25, 40, 17, 23, 14, 22, 1, 7, 9, 4, 41, 2, 36, 0, 12, 38, 39, 37, 24, 35, 34, 16, 21, 8, 31, 49, 47, 15, 19, 33, 18, 3, 43]
  # 50개의 작업을 순서대로 나열
]
# 각 순서에 대해 makespan 계산, 출력 및 Gantt 차트 생성
for order in job_orders:
    makespan = calculate_makespan(dataset, order)
    print(f"\n{'-'.join(map(str, order))}: Makespan = {makespan}")
    
    # Gantt 차트 생성
    machine_log, start_times, completion_times = create_machine_log(dataset, order)
    config.gantt_title = f"Gantt Chart for {'-'.join(map(str, order))}"
    config.update_gantt_filename(list(map(str, order)))  # 인덱스를 문자열로 변환하여 전달
    Gantt(machine_log, config, makespan)
    
    # 각 작업의 시작 시간과 종료 시간 출력
    for i, job in enumerate(order):
        print(f"Job {job}:")
        for j in range(dataset.n_machine):
            print(f"  M{j+1}: 시작 시간 = {start_times[i][j]}, 종료 시간 = {completion_times[i][j]}")

# 최적의 순서 찾기
optimal_order = min(job_orders, key=lambda x: calculate_makespan(dataset, x))
optimal_makespan = calculate_makespan(dataset, optimal_order)
print(f"\n최적 순서: {'-'.join(map(str, optimal_order))}, Makespan: {optimal_makespan}")

# 최적 순서에 대한 Gantt 차트 생성
optimal_index_order = optimal_order
machine_log, start_times, completion_times = create_machine_log(dataset, optimal_index_order)
config.gantt_title = f"Optimal Gantt Chart ({'-'.join(map(str, optimal_order))})"
config.update_gantt_filename(['optimal'] + list(map(str, optimal_order)))  # 인덱스를 문자열로 변환하여 전달
Gantt(machine_log, config, optimal_makespan)

# 최적 순서의 각 작업 시작 시간과 종료 시간 출력
print("\n최적 순서의 작업 시간:")
for i, job in enumerate(optimal_order):
    print(f"Job {job}:")
    for j in range(dataset.n_machine):
        print(f"  M{j+1}: 시작 시간 = {start_times[i][j]}, 종료 시간 = {completion_times[i][j]}")
