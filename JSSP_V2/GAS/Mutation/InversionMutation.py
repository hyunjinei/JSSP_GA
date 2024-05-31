import sys
import os
import random
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from GAS.Mutation.base import Mutation
from GAS.Individual import Individual

class InversionMutation:
    def __init__(self, pm):
        self.pm = pm

    def mutate(self, individual):
        if random.random() < self.pm:
            seq = individual.seq[:]
            start, end = sorted(random.sample(range(len(seq)), 2))
            original_seq = seq[:]
            seq[start:end] = seq[start:end][::-1]
            # print(f"Mutation applied: original seq = {original_seq}, mutated seq = {seq}")
            return Individual(config=individual.config, seq=seq, op_data=individual.op_data)
        # else:
        #     print(f"No mutation applied: seq = {individual.seq}")            
        return individual

