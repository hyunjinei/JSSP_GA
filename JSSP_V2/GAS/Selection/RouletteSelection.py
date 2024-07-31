import sys
import os
import random
import copy
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from GAS.Individual import Individual

class RouletteSelection:
    def __init__(self):
        pass

    def select(self, population):
        selected = []
        for _ in range(len(population)):
            total_fitness = sum(ind.fitness for ind in population)
            pick = random.uniform(0, total_fitness)
            current = 0
            for individual in population:
                current += individual.fitness
                if current > pick:
                    selected.append(individual)
                    break
            if not selected:  # 만약 선택되지 않았다면 마지막 개체 추가
                selected.append(population[-1])
        return selected