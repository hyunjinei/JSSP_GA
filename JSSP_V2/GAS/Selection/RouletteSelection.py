# Selection/RouletteSelection.py
import sys
import os
import random
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from GAS.Individual import Individual

class RouletteSelection:
    def __init__(self):
        pass

    def select(self, population):
        # max_fitness = sum(ind.fitness for ind in population)
        max_fitness = sum(ind.fitness for ind in population)
        pick = random.uniform(0, max_fitness)
        current = 0
        for individual in population:
            # current += individual.fitness
            current += individual.fitness
            if current > pick:
                # print(f"Selected individual with fitness: {individual.fitness}")
                return individual
