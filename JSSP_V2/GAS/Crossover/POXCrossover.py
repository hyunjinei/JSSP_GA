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

        seq_length = len(parent1.seq)
        child1_seq = [-1] * seq_length
        child2_seq = [-1] * seq_length

        # 무작위로 두 개의 하위 작업을 선택
        sub_jobs = random.sample(range(seq_length), 2)
        sub_jobs.sort()
        sj1, sj2 = sub_jobs[0], sub_jobs[1]

        # 하위 작업의 유전자들을 복사
        child1_seq[sj1:sj2+1] = parent1.seq[sj1:sj2+1]
        child2_seq[sj1:sj2+1] = parent2.seq[sj1:sj2+1]

        # 부모2에서 하위 작업 유전자를 제거하고 나머지 유전자로 자리를 채움
        p2_remaining_genes = [gene for gene in parent2.seq if gene not in parent1.seq[sj1:sj2+1]]
        p1_remaining_genes = [gene for gene in parent1.seq if gene not in parent2.seq[sj1:sj2+1]]

        child1_index, child2_index = 0, 0
        for i in range(seq_length):
            if child1_seq[i] == -1:
                child1_seq[i] = p2_remaining_genes[child1_index]
                child1_index += 1
            if child2_seq[i] == -1:
                child2_seq[i] = p1_remaining_genes[child2_index]
                child2_index += 1

        return Individual(config=parent1.config, seq=child1_seq, op_data=parent1.op_data), Individual(config=parent1.config, seq=child2_seq, op_data=parent1.op_data)
