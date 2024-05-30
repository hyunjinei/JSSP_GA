# Mutation/GeneralMutation.py
import sys
import os
import random
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from GAS.Mutation.base import Mutation
from GAS.Individual import Individual


# Mutation/GeneralMutation.py
class GeneralMutation:
    def __init__(self, pm):
        self.pm = pm

    def mutate(self, individual):
        seq = individual.seq[:]
        for i in range(len(seq)):
            if random.random() < self.pm:
                j = random.randint(0, len(seq) - 1)
                seq[i], seq[j] = seq[j], seq[i]
        return Individual(config=individual.config, seq=seq, op_data=individual.op_data)
