import sys
import os
import random
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from GAS.Mutation.base import Mutation
from GAS.Individual import Individual

class SelectiveMutation(Mutation):
    def __init__(self, pm_high, pm_low, rank_divide):
        self.pm_high = pm_high  # 높은 돌연변이 확률
        self.pm_low = pm_low    # 낮은 돌연변이 확률
        self.rank_divide = rank_divide

    def mutate(self, population, config):
        # 적합도에 따라 개체군을 랭킹합니다.
        ranked_population = sorted(population, key=lambda ind: ind.fitness, reverse=True)
        divide_index = int(len(ranked_population) * self.rank_divide)
        
        # 랭크에 따라 좋은 그룹과 나쁜 그룹으로 나눕니다.
        good_group = ranked_population[:divide_index]
        bad_group = ranked_population[divide_index:]

        for ind in good_group:
            if random.random() < self.pm_low:
                self.apply_mutation(ind, config, lower_bits=True)
        
        for ind in bad_group:
            if random.random() < self.pm_high:
                self.apply_mutation(ind, config, lower_bits=False)

    def apply_mutation(self, individual, config, lower_bits):
        # print(f'selective_mutation 적용')
        seq = individual.seq[:]
        if lower_bits:
            # 염색체의 하위 부분에 돌연변이를 적용합니다.
            start, end = sorted(random.sample(range(len(seq)//2), 2))
        else:
            # 염색체의 상위 부분에 돌연변이를 적용합니다.
            start, end = sorted(random.sample(range(len(seq)//2, len(seq)), 2))

        # 역위 돌연변이를 수행합니다.
        seq[start:end] = seq[start:end][::-1]
        individual.seq = seq
        individual.calculate_fitness(config.target_makespan)
        # print(f'selective_mutation 적용 완료')

        return individual
