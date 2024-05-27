# Crossover/OperationBasedGA.py
import sys
import os
import random
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from GAS.Crossover.base import Crossover
from GAS.Individual import Individual

class OperationBasedGA(Crossover):
    def __init__(self, jobs_data, num_machines):
        self.jobs_data = jobs_data
        self.num_machines = num_machines

    def create_individual(self):
        chromosome = []
        for job_index, job in enumerate(self.jobs_data):
            for operation_index in range(len(job)):
                chromosome.append(job_index)  # 각 작업의 인덱스를 추가
        random.shuffle(chromosome)  # 무작위로 섞음
        return chromosome

    def decode_individual(self, individual):
        job_operations = {job_index: 0 for job_index in range(len(self.jobs_data))}
        machine_end_times = [0] * self.num_machines  # 머신 별 종료 시간
        job_end_times = [0] * len(self.jobs_data)  # 작업 별 종료 시간

        schedule = []
        for gene in individual:
            job_index = gene
            operation_index = job_operations[job_index]
            
            if operation_index >= len(self.jobs_data[job_index]):
                continue

            machine_index, duration = self.jobs_data[job_index][operation_index]

            start_time = max(machine_end_times[machine_index], job_end_times[job_index])  # 시작 시간
            end_time = start_time + duration  # 종료 시간

            schedule.append((job_index, operation_index, machine_index, start_time, end_time))  # 스케줄 추가
            machine_end_times[machine_index] = end_time  # 머신 종료 시간 업데이트
            job_end_times[job_index] = end_time  # 작업 종료 시간 업데이트
            job_operations[job_index] += 1  # 작업 단위 증가
        
        return schedule

    def cross(self, parent1, parent2):
        child1_seq = self.create_individual()
        child2_seq = self.create_individual()
        random.shuffle(child1_seq)
        random.shuffle(child2_seq)

        return Individual(config=parent1.config, seq=child1_seq, op_data=parent1.op_data), Individual(config=parent2.config, seq=child2_seq, op_data=parent2.op_data)
