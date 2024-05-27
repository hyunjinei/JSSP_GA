# Crossover/PMX.py
import sys
import os
import random
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from GAS.Crossover.base import Crossover
from GAS.Individual import Individual

# partial-mapped crossover
class PMXCrossover(Crossover):
    def __init__(self, pc):
        self.pc = pc

    def cross(self, parent1, parent2):
        if random.random() > self.pc:
            # If the random number is greater than pc, return parents as children without crossover
            return parent1, parent2

        point1, point2 = sorted(random.sample(range(len(parent1.seq)), 2))
        child1, child2 = parent1.seq[:], parent2.seq[:]

        for i in range(point1, point2):
            val1, val2 = parent1.seq[i], parent2.seq[i]
            idx1, idx2 = parent1.seq.index(val2), parent2.seq.index(val1)
            child1[i], child1[idx1] = child1[idx1], child1[i]
            child2[i], child2[idx2] = child2[idx2], child2[i]

        return Individual(config=parent1.config, seq=child1, op_data=parent1.op_data), Individual(config=parent1.config, seq=child2, op_data=parent1.op_data)
