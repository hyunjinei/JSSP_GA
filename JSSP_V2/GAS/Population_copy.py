# Population.py


import numpy as np
import random
from GAS.Individual import Individual

class Population:
    def __init__(self, config, op_data):
        self.config = config
        self.op_data = op_data
        self.individuals = [Individual(config, seq=random.sample(range(config.n_op), config.n_op), op_data=op_data) for _ in range(config.population_size)]

    def evaluate(self, target_makespan):
        for individual in self.individuals:
            individual.makespan, individual.mio_score = individual.evaluate(individual.machine_order)
            individual.calculate_fitness(target_makespan)
        # Apply min-max scaling
        fitness_values = [ind.fitness for ind in self.individuals]
        min_fitness = min(fitness_values)
        max_fitness = max(fitness_values)

        if max_fitness - min_fitness > 0:
            for individual in self.individuals:
                individual.scaled_fitness = (individual.fitness - min_fitness) / (max_fitness - min_fitness)
                print(f"Scaled fitness: {individual.scaled_fitness}, Makespan: {individual.makespan}")
        else:
            for individual in self.individuals:
                individual.scaled_fitness = 1.0  # In case all fitness values are the same
                print(f"Scaled fitness: {individual.scaled_fitness}, Makespan: {individual.makespan}")

    '''
    fit 수정본
    def evaluate(self):
        for individual in self.individuals:
            individual.makespan, individual.mio_score = individual.evaluate(individual.machine_order)
            individual.calculate_fitness(self.config.target_makespan)  # Pass target_makespan

        # Apply min-max scaling
        fitness_values = [ind.fitness for ind in self.individuals]
        min_fitness = min(fitness_values)
        max_fitness = max(fitness_values)

        if max_fitness - min_fitness > 0:
            for individual in self.individuals:
                individual.scaled_fitness = (individual.fitness - min_fitness) / (max_fitness - min_fitness)
                print(f"Scaled fitness: {individual.scaled_fitness}, Makespan: {individual.makespan}")
        else:
            for individual in self.individuals:
                individual.scaled_fitness = 1.0  # In case all fitness values are the same
                print(f"Scaled fitness: {individual.scaled_fitness}, Makespan: {individual.makespan}")
    '''
            
    '''
    오리지날 evaluate 함수
    
    def evaluate(self):
        for individual in self.individuals:
            individual.makespan, individual.mio_score = individual.evaluate(individual.machine_order)
            individual.calculate_fitness()
        # Apply min-max scaling
        fitness_values = [ind.fitness for ind in self.individuals]
        min_fitness = min(fitness_values)
        max_fitness = max(fitness_values)

        if max_fitness - min_fitness > 0:
            for individual in self.individuals:
                individual.scaled_fitness = (individual.fitness - min_fitness) / (max_fitness - min_fitness)
                print(f"Scaled fitness: {individual.scaled_fitness}, Makespan: {individual.makespan}")
        else:
            for individual in self.individuals:
                individual.scaled_fitness = 1.0  # In case all fitness values are the same
                print(f"Scaled fitness: {individual.scaled_fitness}, Makespan: {individual.makespan}")
    '''
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