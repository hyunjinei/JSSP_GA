import os
import numpy as np
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class RLDataset_FJSSP:
    def __init__(self, filename):
        self.name, _ = os.path.splitext(filename)
        file_path = os.path.join(os.getcwd(), filename)
        
        with open(file_path, 'r') as file:
            first_line = file.readline().strip().split()
        
        self.n_job = int(first_line[0])
        self.n_machine = int(first_line[1])
        self.flexibility = float(first_line[2])  # 유연성 값을 추가로 저장

        self.op_data = []

        with open(file_path, 'r') as file:
            lines = file.readlines()[1:]  # 첫 줄을 제외한 나머지 읽기

        self.total_operations = 0  # 총 Operation 수 초기화
        self.max_op_counts = []  # 각 작업의 최대 작업 단계를 저장

        job_index = 0
        for line in lines:
            parts = list(map(int, line.strip().split()))
            num_operations = parts[0]
            self.max_op_counts.append(num_operations - 1)  # 최대 작업 단계 수를 추가
            self.total_operations += num_operations  # 각 작업의 Operation 수를 누적
            self.op_data.append([])
            index = 1
            for _ in range(num_operations):
                machine_options = []
                num_machine_options = parts[index]
                index += 1
                for _ in range(num_machine_options):
                    machine = parts[index] - 1  # 기계 번호 (0부터 시작하도록 조정)
                    time = parts[index + 1]  # 처리 시간
                    machine_options.append((machine, time))
                    index += 2
                self.op_data[job_index].append(machine_options)
            job_index += 1

        print(f"Loaded data from {filename}")
        print(f"Number of jobs: {self.n_job}, Number of machines: {self.n_machine}, Flexibility: {self.flexibility}")
        print(f"Total operations: {self.total_operations}")
        print(f"Max operation counts: {self.max_op_counts}")

def load_all_datasets(filenames):
    datasets = []
    for filename in filenames:
        dataset_loader = RLDataset_FJSSP(filename)
        datasets.append(dataset_loader)
    return datasets
