# Crossover/OrderCrossover.py
import sys
import os
import random
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from GAS.Crossover.base import Crossover
from GAS.Individual import Individual

# OrderCrossover
class OrderCrossover(Crossover):
    def __init__(self, pc):
        self.pc = pc

    def cross(self, parent1, parent2):
        if random.random() > self.pc:
            return parent1, parent2

        point1, point2 = sorted(random.sample(range(len(parent1.seq)), 2))
        child1, child2 = parent1.seq[:], parent2.seq[:]

        # Create proto-children by inserting the selected substring into the corresponding positions
        child1[point1:point2], child2[point1:point2] = parent1.seq[point1:point2], parent2.seq[point1:point2]

        # Remove the selected substring symbols from the other parent
        temp1 = [item for item in parent2.seq if item not in parent1.seq[point1:point2]]
        temp2 = [item for item in parent1.seq if item not in parent2.seq[point1:point2]]

        # Fill unfixed positions
        idx1, idx2 = 0, 0
        for i in range(len(child1)):
            if not (point1 <= i < point2):
                child1[i] = temp1[idx1]
                idx1 += 1
                child2[i] = temp2[idx2]
                idx2 += 1

        return Individual(config=parent1.config, seq=child1, op_data=parent1.op_data), Individual(config=parent1.config, seq=child2, op_data=parent1.op_data)
