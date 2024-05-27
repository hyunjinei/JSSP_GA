from __future__ import absolute_import

from geneticpython.core.operators.crossover import Crossover
from geneticpython.core.individual import Individual
from geneticpython.models import PermutationIndividual
from geneticpython.utils.validation import check_random_state

from copy import deepcopy
from random import Random
from typing import Callable

import random
import numpy as np


# 정렬 함수 정의
def custom_sort(num_list):
    return sorted(num_list, key=lambda x: (x % 15, x // 15))


class PMXCrossover(Crossover):
    def __init__(self, pc: float):
        super(PMXCrossover, self).__init__(pc=pc)
        self.p_mio = 0.9
        self.num_called = 0

    def cross(self, father: PermutationIndividual, mother: PermutationIndividual, random_state=None):
        ''' Cross chromsomes of parent using single point crossover method.
        '''
        random_state = check_random_state(random_state)
        do_cross = True if random_state.random() <= self.pc else False
        self.p_mio = 0.99 * self.p_mio
        if not do_cross:
            return father.clone(), mother.clone()

        if random.random() < self.p_mio:

            self.num_called += 1
            chrom1 = deepcopy(father.chromosome)
            # 0부터 99까지의 수 생성
            numbers = list(range(1500))
            # 패턴대로 정렬
            sorted_numbers = custom_sort(numbers)

            child1 = father.clone()
            for i in range(1500):
                chrom1[i] = sorted_numbers[i]
            child1.init(chromosome=chrom1)

            return child1, mother.clone()

        chrom1 = deepcopy(father.chromosome)
        chrom2 = deepcopy(mother.chromosome)

        N_OPERATION = len(father.chromosome)
        c_1 = [0 for i in range(N_OPERATION)]
        c_2 = [0 for j in range(N_OPERATION)]
        # select range
        r = [random.randint(0, N_OPERATION - 1) for _ in range(2)]
        while min(r) == max(r):
            r = [random.randint(0, N_OPERATION - 1) for _ in range(2)]

        r1 = min(r)  # left end
        r2 = max(r)  # right end

        # Chromsomes for two children.
        p_1 = deepcopy(father.chromosome.genes.tolist())
        p_2 = deepcopy(mother.chromosome.genes.tolist())

        slice_1 = p_1[r1:r2]  # c_2에 들어갈 부분
        slice_2 = p_2[r1:r2]  # c_1에 들어갈 부분
        p1a = p_1[:r1] + p_1[r2:]
        p2a = p_2[:r1] + p_2[r2:]
        repeated_idx_1 = []  # p1의 leftover에 slice 2의 요소가 있음
        repeated_idx_2 = []  # p2의 leftover에 slice 1의 요소가 있음

        for e in slice_2:  # c_1에 들어가야 하는 slice 2의 요소가 이미 P1의 leftover에 있으면
            if e in p1a:  # p1a는 c2에게 전달될 예정
                # C1 입장에서, P1에서 반복되어 없어져야 하는 것들
                repeated_idx_1.append(p_1.index(e))  # 해당 중복원소들의 위치를 기록

        for e in slice_1:  # c_2에 들어가야 하는 slice 1의 요소가 이미 P2의 leftover에 있으면
            if e in p2a:
                repeated_idx_2.append(p_2.index(e))  # C2 입장에서, P2에서 반복되어 없어져야 하는 것들

        # 등장하는 index 순서대로 정렬
        repeated_idx_1 = sorted(repeated_idx_1)
        repeated_idx_2 = sorted(repeated_idx_2)

        repeated_p1 = [p_1[n] for n in repeated_idx_1]
        repeated_p2 = [p_2[n] for n in repeated_idx_2]

        left_1 = deepcopy(repeated_p1)
        left_2 = deepcopy(repeated_p2)
        for i in range(len(p_1)):
            if i not in range(r1, r2):
                if p_1[i] not in repeated_p1:
                    c_1[i] = p_1[i]
                else:
                    # print(p_1[i])
                    if len(left_2) == 0: return father, mother
                    c_1[i] = left_2.pop(0)
            else:
                if len(slice_2) == 0: return father, mother
                c_1[i] = slice_2.pop(0)

        for i in range(len(p_1)):
            if i not in range(r1, r2):
                if p_2[i] not in repeated_p2:
                    c_2[i] = p_2[i]
                else:
                    if len(left_1) == 0: return father, mother

                    c_2[i] = left_1.pop(0)
            else:
                if len(slice_1) == 0: return father, mother

                c_2[i] = slice_1.pop(0)
        # print('C1 : ', c_1)
        # print('C2 : ', c_2)
        for i in range(N_OPERATION):
            chrom1[i] = c_1[i]
            chrom2[i] = c_2[i]

        child1, child2 = father.clone(), father.clone()
        child1.init(chromosome=chrom1)
        child2.init(chromosome=chrom2)

        return child1, child2


def duplicate(father, chrom1, chrom2):
    child1, child2 = father.clone(), father.clone()
    child1.init(chromosome=chrom1)
    child2.init(chromosome=chrom2)
    return child1, child2
