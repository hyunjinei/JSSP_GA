# Selection/TruncationSelection.py
import random
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from GAS.Individual import Individual

class TruncationSelection:
    def __init__(self, elite_TS=0.2):
        self.elite_TS = elite_TS  # elite_TS 비율 설정

    def select(self, population):
        # 상위 elite_TS% 개체를 선택
        elite_count = max(1, int(len(population) * self.elite_TS))
        elite_individuals = sorted(population, key=lambda ind: ind.fitness)[:elite_count]

        # 나머지 개체는 무작위로 선택
        remaining_individuals = [ind for ind in population if ind not in elite_individuals]
        selected_individuals = elite_individuals[:]

        while len(selected_individuals) < len(population):
            tournament = random.sample(remaining_individuals, min(len(remaining_individuals), 2))
            winner = max(tournament, key=lambda ind: ind.fitness)
            selected_individuals.append(winner)

        return selected_individuals
