import sys
import os
import random
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from GAS.Mutation.base import Mutation
from GAS.Individual import Individual

class ReciprocalExchangeMutation:
    def __init__(self, pm):
        self.pm = pm

    def mutate(self, individual):
        if random.random() < self.pm:
            seq = individual.seq[:]
            pos1, pos2 = random.sample(range(len(seq)), 2)
            seq[pos1], seq[pos2] = seq[pos2], seq[pos1]
            return Individual(config=individual.config, seq=seq, op_data=individual.op_data)
        return individual

