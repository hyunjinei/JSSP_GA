import sys
import os
import random
import copy
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from GAS.Mutation.base import Mutation
from GAS.Individual import Individual

class InversionMutation:
    def __init__(self, pm, min_inverse_length=3, max_inverse_length=10, num_inversions=2):
        self.pm = pm
        self.min_inverse_length = min_inverse_length
        self.max_inverse_length = max_inverse_length
        self.num_inversions = num_inversions

    def mutate(self, individual):
        if random.random() < self.pm:
            size = len(individual.seq)
            for _ in range(self.num_inversions):
                inverse_length = random.randint(self.min_inverse_length, min(self.max_inverse_length, size-1))
                start = random.randint(0, size - inverse_length)
                end = start + inverse_length
                individual.seq[start:end] = reversed(individual.seq[start:end])
        return individual