# Crossover/PositionBasedCrossover.py
import sys
import os
import random
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from GAS_FJSP.Crossover.base import Crossover
from GAS_FJSP.Individual import Individual

# PositionBasedCrossover
class PositionBasedCrossover(Crossover):
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
            child1[pos] = parent2.seq[pos]
            child2[pos] = parent1.seq[pos]

        # Step 3: Remove selected positions' symbols from the other parent
        parent1_remaining = [item for item in parent1.seq if item not in child1]
        parent2_remaining = [item for item in parent2.seq if item not in child2]

        # Step 4: Fill unfixed positions
        idx1, idx2 = 0, 0
        for i in range(size):
            if child1[i] is None:
                child1[i] = parent1_remaining[idx1]
                idx1 += 1
            if child2[i] is None:
                child2[i] = parent2_remaining[idx2]
                idx2 += 1

        return Individual(config=parent1.config, seq=child1, op_data=parent1.op_data), Individual(config=parent1.config, seq=child2, op_data=parent2.op_data)
