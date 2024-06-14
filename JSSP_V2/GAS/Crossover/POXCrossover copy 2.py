# Crossover/POXCrossover.py
import sys
import os
import random
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from GAS.Crossover.base import Crossover
from GAS.Individual import Individual

class POXCrossover(Crossover):
    def __init__(self, pc):
        self.pc = pc

    def cross(self, parent1, parent2):
        if random.random() > self.pc:
            return parent1, parent2

        # POX 교차: 작업 순서 벡터에 적용
        seq_length = len(parent1.seq)
        child1_seq = [-1] * seq_length
        child2_seq = [-1] * seq_length

        # 무작위로 두 개의 하위 작업을 선택
        job_indices = list(set(num // parent1.config.n_machine for num in range(seq_length)))
        sub_jobs = random.sample(job_indices, 2)
        sub_jobs.sort()
        sj1, sj2 = sub_jobs[0], sub_jobs[1]

        # 하위 작업의 유전자들을 복사
        for i in range(seq_length):
            if i // parent1.config.n_machine == sj1 or i // parent1.config.n_machine == sj2:
                child1_seq[i] = parent1.seq[i]
                child2_seq[i] = parent2.seq[i]

        # 부모2에서 하위 작업 유전자를 제거하고 나머지 유전자로 자리를 채움
        def fill_remaining(child_seq, parent_seq):
            index = 0
            for gene in parent_seq:
                if gene not in child_seq:
                    while child_seq[index] != -1:
                        index += 1
                    child_seq[index] = gene

        fill_remaining(child1_seq, parent2.seq)
        fill_remaining(child2_seq, parent1.seq)

        # Uniform 교차: 기계 할당 벡터에 적용
        assignment_length = len(parent1.machine_assignment)
        child1_assignment = [-1] * assignment_length
        child2_assignment = [-1] * assignment_length

        # 이진 마스크 생성
        mask = [random.randint(0, 1) for _ in range(assignment_length)]

        # 마스크 적용
        for i in range(assignment_length):
            if mask[i] == 1:
                child1_assignment[i] = parent1.machine_assignment[i]
                child2_assignment[i] = parent2.machine_assignment[i]
            else:
                child1_assignment[i] = parent2.machine_assignment[i]
                child2_assignment[i] = parent1.machine_assignment[i]

        return Individual(config=parent1.config, seq=child1_seq, machine_assignment=child1_assignment, op_data=parent1.op_data), Individual(config=parent1.config, seq=child2_seq, machine_assignment=child2_assignment, op_data=parent1.op_data)
