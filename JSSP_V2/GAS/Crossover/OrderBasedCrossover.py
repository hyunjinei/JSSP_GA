# Crossover/OrderBasedCrossover.py
import sys
import os
import random
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from GAS.Crossover.base import Crossover
from GAS.Individual import Individual

# OrderBasedCrossover
class OBC(Crossover):
    def __init__(self, pc):
        self.pc = pc

    def cross(self, parent1, parent2):
        if random.random() > self.pc:
            return parent1, parent2

        size = len(parent1.seq)
        child1, child2 = [None]*size, [None]*size

        # Step 1: Select positions from Parent 1
        positions = sorted(random.sample(range(size), random.randint(1, size - 1)))

        # Step 2: Produce Proto-child
        for pos in positions:
            child1[pos] = parent1.seq[pos]
            child2[pos] = parent2.seq[pos]

        # Step 3: Remove selected positions' symbols from the other parent
        parent2_filtered = [item for item in parent2.seq if item not in child1]
        parent1_filtered = [item for item in parent1.seq if item not in child2]

        # Step 4: Fill unfixed positions in the order of the other parent
        idx1, idx2 = 0, 0
        for i in range(size):
            if child1[i] is None:
                child1[i] = parent2_filtered[idx1]
                idx1 += 1
            if child2[i] is None:
                child2[i] = parent1_filtered[idx2]
                idx2 += 1

        return Individual(config=parent1.config, seq=child1, op_data=parent1.op_data), Individual(config=parent1.config, seq=child2, op_data=parent2.op_data)

