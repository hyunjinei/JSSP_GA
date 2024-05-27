# Crossover/LOX.py
import sys
import os
import random
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from GAS.Crossover.base import Crossover
from GAS.Individual import Individual

# Linear order crossover 
class LOXCrossover(Crossover):
    def __init__(self, pc):
        self.pc = pc

    def cross(self, parent1, parent2):
        if random.random() > self.pc:
            return parent1, parent2

        size = len(parent1.seq)
        
        # Step 1: Randomly select sublists from both parents
        point1, point2 = sorted(random.sample(range(size), 2))
        sublist1 = parent1.seq[point1:point2]
        sublist2 = parent2.seq[point1:point2]

        # Step 2: Remove sublists and create holes
        child1_holes = [gene if gene not in sublist2 else None for gene in parent1.seq]
        child2_holes = [gene if gene not in sublist1 else None for gene in parent2.seq]

        # Slide holes to the end
        child1 = [gene for gene in child1_holes if gene is not None]
        child2 = [gene for gene in child2_holes if gene is not None]

        child1.extend([None] * (size - len(child1)))
        child2.extend([None] * (size - len(child2)))

        # Step 3: Insert sublists into the holes
        child1[point1:point2] = sublist2
        child2[point1:point2] = sublist1

        # Handle remaining holes by filling with the remaining elements in the order they appear in the other parent
        remaining1 = [gene for gene in parent2.seq if gene not in child1]
        remaining2 = [gene for gene in parent1.seq if gene not in child2]

        for i in range(size):
            if child1[i] is None:
                child1[i] = remaining1.pop(0)
            if child2[i] is None:
                child2[i] = remaining2.pop(0)

        return Individual(config=parent1.config, seq=child1, op_data=parent1.op_data), Individual(config=parent1.config, seq=child2, op_data=parent1.op_data)

# Test the LOX
