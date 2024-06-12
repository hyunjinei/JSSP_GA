# Selection/TournamentSelection.py
import random
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from GAS.Individual import Individual

class TournamentSelection:
    def __init__(self, tournament_size=2):
        self.tournament_size = tournament_size

    def select(self, population):
        # 토너먼트에 참가할 염색체 무작위 선택
        tournament = random.sample(population, self.tournament_size)
        # 토너먼트에서 가장 적합한 염색체 선택
        winner = max(tournament, key=lambda ind: ind.fitness)
        # winner = max(tournament, key=lambda ind: ind.scaled_fitness)
        # print(f"TournamentSelection: Selected individual with fitness: {winner.fitness}")  # Selection log
        return winner
