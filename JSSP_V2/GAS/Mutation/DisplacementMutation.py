import sys
import os
import random
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from GAS.Mutation.base import Mutation
from GAS.Individual import Individual

class DisplacementMutation:
    def __init__(self, pm):
        self.pm = pm

    def mutate(self, individual):
        if random.random() < self.pm:
            seq = individual.seq[:]
            start, end = sorted(random.sample(range(len(seq)), 2))
            sub_seq = seq[start:end]
            seq = seq[:start] + seq[end:]
            pos = random.randint(0, len(seq))
            seq = seq[:pos] + sub_seq + seq[pos:]
            return Individual(config=individual.config, seq=seq, op_data=individual.op_data)
        return individual

