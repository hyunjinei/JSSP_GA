# Selection/SeedSelection.py
import random
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from GAS.Individual import Individual

class SeedSelection:
    def __init__(self, k=0.75):
        self.k = k  # 확률값 k 설정

    def select(self, population):
        # male: 가장 적합한 염색체
        male = max(population, key=lambda ind: ind.fitness)
        # female: 랜덤하게 선택
        female = random.choice(population)
        # 확률 k에 따라 male 또는 female 선택
        selected = male if random.random() < self.k else female
        # print(f"Selected individual with fitness: {selected.fitness}")  # Selection log
        return selected
