# Population.py

import copy
import numpy as np
import random
from GAS.Individual import Individual
from Data.Dataset.Dataset import Dataset

print_console = False

class Operation:
    def __init__(self, i, j, machine, n_machine):
        self.job = i
        self.precedence = j
        self.machine = machine
        self.idx = n_machine * i + j
        self.job_ready = j == 0
        self.machine_ready = j == 0
        self.op_prior = None
        self.op_following = None

class MIOMachine:
    def __init__(self, id, n_machine):
        self.id = id
        self.op_ready = []
        self.op_by_order = [[] for _ in range(n_machine)]
        self.current_position = 0
        self.finished = False

    def initialize_op_ready(self):
        while self.current_position < len(self.op_by_order) and not self.op_by_order[self.current_position]:
            self.current_position += 1
        if self.current_position < len(self.op_by_order):
            self.op_ready = self.op_by_order[self.current_position]
            for op in self.op_ready:
                op.machine_ready = True

    def update_op_ready(self):
        while not self.op_ready and not self.finished:
            self.current_position += 1
            if self.current_position >= len(self.op_by_order):
                self.finished = True
            else:
                self.op_ready = self.op_by_order[self.current_position]
                for op in self.op_ready:
                    op.machine_ready = True

class JSSP:
    def __init__(self, dataset):
        self.dataset = dataset
        self.op_data = dataset.op_data
        self.op_list = [[] for _ in range(self.dataset.n_job)]
        self.machine_list = [MIOMachine(i, dataset.n_machine) for i in range(self.dataset.n_machine)]

        # Initialization
        for i in range(self.dataset.n_job):
            for j in range(self.dataset.n_machine):
                self.op_list[i].append(Operation(i, j, self.machine_list[self.op_data[i][j][0]], dataset.n_machine))

        for i in range(self.dataset.n_job):
            for j in range(self.dataset.n_machine):
                self.machine_list[self.op_data[i][j][0]].op_by_order[j].append(self.op_list[i][j])

        for i in range(self.dataset.n_machine):
            self.machine_list[i].initialize_op_ready()

        # 연결 관계 수립
        for i in range(self.dataset.n_job):
            for j in range(1, self.dataset.n_machine):
                self.op_list[i][j].op_prior = self.op_list[i][j - 1]
                self.op_list[i][j - 1].op_following = self.op_list[i][j]

    def get_seq(self):
        self.ready = []
        self.seq = []
        for i in range(self.dataset.n_job):
            for j in range(self.dataset.n_machine):
                if self.op_list[i][j].job_ready and self.op_list[i][j].machine_ready:
                    self.ready.append(self.op_list[i][j])

        while len(self.seq) < self.dataset.n_op:
            if print_console: print('1. 현재 대기중인 작업 : ', [op.idx for op in self.ready])
            random.shuffle(self.ready)
            op = self.ready.pop()
            if print_console: print('2. 결정된 작업 : ', (op.job, op.precedence))
            self.seq.append(op)

            if op.precedence != self.dataset.n_machine - 1:
                if print_console: print('3. 현재까지 형성된 sequence : ', [op.idx for op in self.seq])
                if print_console: print('3-1. sequence 길이 :', len(self.seq))

                op.op_following.job_ready = True
                if print_console: print('4. 같은 job의 다음 operation의 작업 가능 현황 : ',
                                        (op.op_following.job_ready, op.op_following.machine_ready))

                if print_console: print('5-1. machine의 ready list 수정 전 : ', [op.idx for op in op.machine.op_ready])
                
                # Check if op is in op_ready before removing
                if op in op.machine.op_ready:
                    op.machine.op_ready.remove(op)
                    if print_console: print('5-2. machine의 ready list 수정 후 : ', [op.idx for op in op.machine.op_ready])
                else:
                    print(f"Error: Operation {op.idx} not found in machine {op.machine.id} op_ready list")
                    print(f"Current op_ready list: {[op.idx for op in op.machine.op_ready]}")
                    break  # Stop the process to prevent further errors

                op.machine.update_op_ready()

                # check
                if op.op_following.job_ready and op.op_following.machine_ready:
                    if op.op_following not in self.ready:
                        self.ready.append(op.op_following)
                        if print_console: print('6. Job 진행으로 인해 새롭게 ready list에 추가되는 작업 : ',
                                                (op.op_following.job, op.op_following.precedence))

                for x in op.machine.op_ready:
                    if x.job_ready and x.machine_ready:
                        if x not in self.ready:
                            self.ready.append(x)
                            if print_console: print('7. Machine 진행으로 인해 새롭게 ready list에 추가되는 작업 : ',
                                                    (x.idx))

        s = [op.idx for op in self.seq]
        self.__init__(self.dataset)
        return s

class GifflerThompson:
    def __init__(self, priority_rules=None):
        if priority_rules is None:
            self.priority_rules = ['SPT', 'LPT', 'MWR', 'LWR', 'MOR', 'LOR', 'EDD']
        else:
            self.priority_rules = priority_rules

    def optimize(self, individual, config):
        selected_rule = random.choice(self.priority_rules)
        schedule = self.giffler_thompson(individual.seq, individual.op_data, config, selected_rule)
        optimized_individual = copy.deepcopy(individual)
        optimized_individual.seq = schedule
        optimized_individual.calculate_fitness(config.target_makespan)
        return optimized_individual

    def giffler_thompson(self, seq, op_data, config, rule):
        sorted_seq = self.apply_priority_rule(seq, op_data, config, rule)
        return sorted_seq

    def apply_priority_rule(self, seq, op_data, config, rule):
        def safe_get_op_data(x, idx):
            try:
                return op_data[x // config.n_machine][x % config.n_machine][idx]
            except IndexError:
                return float('inf') if idx == 1 else 0

        if rule == 'SPT':
            sorted_seq = sorted(seq, key=lambda x: safe_get_op_data(x, 1))
        elif rule == 'LPT':
            sorted_seq = sorted(seq, key=lambda x: -safe_get_op_data(x, 1))
        elif rule == 'MWR':
            sorted_seq = sorted(seq, key=lambda x: -sum(safe_get_op_data(x, 1) for i in range(config.n_machine)))
        elif rule == 'LWR':
            sorted_seq = sorted(seq, key=lambda x: sum(safe_get_op_data(x, 1) for i in range(config.n_machine)))
        elif rule == 'MOR':
            sorted_seq = sorted(seq, key=lambda x: -len([op for op in op_data[x // config.n_machine] if op[1] > 0]))
        elif rule == 'LOR':
            sorted_seq = sorted(seq, key=lambda x: len([op for op in op_data[x // config.n_machine] if op[1] > 0]))
        elif rule == 'EDD':
            sorted_seq = sorted(seq, key=lambda x: safe_get_op_data(x, 2))
        else:
            sorted_seq = seq  # Default to no sorting
        return sorted_seq

class Population:
    def __init__(self, config, op_data):
        self.config = config
        self.op_data = op_data
        self.individuals = [Individual(config, seq=random.sample(range(config.n_op), config.n_op), op_data=op_data) for _ in range(config.population_size)]

    @classmethod
    def from_mio(cls, config, op_data, dataset_filename):
        dataset = Dataset(dataset_filename)
        jssp = JSSP(dataset)
        individuals = [Individual(config, seq=jssp.get_seq(), op_data=dataset.op_data) for _ in range(config.population_size)]
        population = cls(config, dataset.op_data)  # Create the Population instance with required arguments
        population.individuals = individuals
        return population

    @classmethod
    def from_giffler_thompson(cls, config, op_data, dataset_filename):
        dataset = Dataset(dataset_filename)
        giffler_thompson = GifflerThompson()
        individuals = [Individual(config, seq=giffler_thompson.optimize(Individual(config, seq=random.sample(range(config.n_op), config.n_op), op_data=dataset.op_data), config).seq, op_data=dataset.op_data) for _ in range(config.population_size)]
        population = cls(config, dataset.op_data)
        population.individuals = individuals
        return population

    def evaluate(self, target_makespan):
        for individual in self.individuals:
            individual.makespan, individual.mio_score = individual.evaluate(individual.machine_order)
            individual.calculate_fitness(target_makespan)
        # 스케일링 방법 선택 (Rank Scaling, Sigma Scaling, Boltzmann Scaling)
        scaling_method = 'min-max'  # 'min-max', 'sigma', 'boltzmann' 등을 사용할 수 있습니다.

        if scaling_method == 'min-max':
            self.min_max_scaling()
        elif scaling_method == 'rank':
            self.rank_scaling()
        elif scaling_method == 'sigma':
            self.sigma_scaling()
        elif scaling_method == 'boltzmann':
            self.boltzmann_scaling()
    
    def min_max_scaling(self):
        fitness_values = [ind.fitness for ind in self.individuals]
        min_fitness = min(fitness_values)
        max_fitness = max(fitness_values)

        if max_fitness - min_fitness > 0:
            for individual in self.individuals:
                individual.scaled_fitness = (individual.fitness - min_fitness) / (max_fitness - min_fitness)
        else:
            for individual in self.individuals:
                individual.scaled_fitness = 1.0  # In case all fitness values are the same

    def rank_scaling(self):
        sorted_individuals = sorted(self.individuals, key=lambda ind: ind.fitness, reverse=True)
        for rank, individual in enumerate(sorted_individuals):
            individual.scaled_fitness = rank + 1  # 순위를 적합도로 사용

    def sigma_scaling(self):
        fitness_values = [ind.fitness for ind in self.individuals]
        mean_fitness = np.mean(fitness_values)
        std_fitness = np.std(fitness_values)
        
        for individual in self.individuals:
            if std_fitness > 0:
                individual.scaled_fitness = 1 + (individual.fitness - mean_fitness) / (2 * std_fitness)
            else:
                individual.scaled_fitness = 1  # 표준편차가 0인 경우

    def boltzmann_scaling(self, T=1.0):
        fitness_values = [ind.fitness for ind in self.individuals]
        exp_values = np.exp(fitness_values / T)
        sum_exp_values = np.sum(exp_values)
        
        for individual in self.individuals:
            individual.scaled_fitness = exp_values[self.individuals.index(individual)] / sum_exp_values

    def select(self, selection):
        self.individuals = [selection.select(self.individuals) for _ in range(self.config.population_size)]

    def crossover(self, crossover):
        next_generation = []
        for i in range(0, len(self.individuals), 2):
            parent1, parent2 = self.individuals[i], self.individuals[i + 1]
            child1, child2 = crossover.cross(parent1, parent2)
            next_generation.extend([child1, child2])
        self.individuals = next_generation

    def mutate(self, mutation):
        for individual in self.individuals:
            mutation.mutate(individual)

    def preserve_elites(self, elites):
        self.individuals[:len(elites)] = elites
