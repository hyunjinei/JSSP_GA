# Mutation/DiverseSwapMutation.py
import sys
import os
import random
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from GAS.Mutation.base import Mutation
from GAS.Individual import Individual

class DiverseSwapMutation:
    def __init__(self, pm):
        self.pm = pm

    def mutate(self, individual):
        if random.random() < self.pm:
            seq = individual.seq[:]
            i, j = random.sample(range(len(seq)), 2)
            seq[i], seq[j] = seq[j], seq[i]

            # 새로운 개체 반환
            return Individual(config=individual.config, seq=seq, op_data=individual.op_data)
        return individual
