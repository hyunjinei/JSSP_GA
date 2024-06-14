# Crossover/POXCrossover.py
import sys
import os
import random
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from GAS.Crossover.base import Crossover
from GAS.Individual import Individual

class UniformCrossover(Crossover):
    def __init__(self, pc):
        self.pc = pc

    def cross(self, parent1, parent2):
        if random.random() > self.pc:
            return parent1, parent2

        seq_length = len(parent1.assignment)
        child1_assignment = [-1] * seq_length
        child2_assignment = [-1] * seq_length

        mask = [random.randint(0, 1) for _ in range(seq_length)]

        for i in range(seq_length):
            if mask[i] == 1:
                child1_assignment[i] = parent1.assignment[i]
                child2_assignment[i] = parent2.assignment[i]
            else:
                child1_assignment[i] = parent2.assignment[i]
                child2_assignment[i] = parent1.assignment[i]

        return (Individual(config=parent1.config, seq=parent1.seq, assignment=child1_assignment, op_data=parent1.op_data),
                Individual(config=parent1.config, seq=parent2.seq, assignment=child2_assignment, op_data=parent1.op_data))
