# Crossover/CX.py
import sys
import os
import random
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from GAS.Crossover.base import Crossover
from GAS.Individual import Individual

# Cycle crossover 
class CXCrossover(Crossover):
    def __init__(self, pc):
        self.pc = pc

    def cross(self, parent1, parent2):
        if random.random() > self.pc:
            return parent1, parent2

        size = len(parent1.seq)
        child1, child2 = [None]*size, [None]*size

        def create_cycle(parent1_seq, parent2_seq):
            cycle = []
            index = 0
            while index not in cycle:
                cycle.append(index)
                index = parent1_seq.index(parent2_seq[index])
            return cycle

        cycle_indices = create_cycle(parent1.seq, parent2.seq)

        for i in cycle_indices:
            child1[i], child2[i] = parent1.seq[i], parent2.seq[i]

        for i in range(size):
            if child1[i] is None:
                child1[i] = parent2.seq[i]
            if child2[i] is None:
                child2[i] = parent1.seq[i]

        return Individual(config=parent1.config, seq=child1, op_data=parent1.op_data), Individual(config=parent1.config, seq=child2, op_data=parent1.op_data)
