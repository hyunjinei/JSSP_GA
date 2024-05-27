# Mutation/GeneralMutation.py
import sys
import os
import random
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from GAS.Mutation.base import Mutation
from GAS.Individual import Individual


class GeneralMutation:
    def __init__(self, pm):
        self.pm = pm

    def mutate(self, individual):
        for i in range(len(individual.seq)):
            if random.random() < self.pm:
                j = random.randint(0, len(individual.seq) - 1)
                individual.seq[i], individual.seq[j] = individual.seq[j], individual.seq[i]
        individual.makespan, individual.mio_score = individual.evaluate(individual.get_machine_order())
        individual.calculate_fitness()
        return individual
