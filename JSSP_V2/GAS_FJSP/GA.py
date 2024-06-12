import sys
import os
import random
import time
import copy
import csv
from concurrent.futures import ProcessPoolExecutor

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from GAS_FJSP.Population import Population
from GAS_FJSP.Local_Search.TabuSearch import TabuSearch
from Data.Dataset.Dataset_multi import Dataset
from GAS_FJSP.Meta.PSO import PSO
from GAS_FJSP.Mutation.SelectiveMutation import SelectiveMutation
from GAS_FJSP.Local_Search.HillClimbing import HillClimbing
from GAS_FJSP.Local_Search.SimulatedAnnealing import SimulatedAnnealing
from GAS_FJSP.Local_Search.GifflerThompson_LS import GifflerThompson_LS
from concurrent.futures import ProcessPoolExecutor
import datetime
# from Meta.ORtools import ORToolsOptimizer
import simpy


class GAEngine:
    def __init__(self, config, op_data, crossover, mutation, selection, local_search=None, pso=None, selective_mutation=None, elite_ratio=0.1, ga_engines=None, island_mode=1, migration_frequency=10, initialization_mode='1', dataset_filename=None, initial_population=None, local_search_frequency=2, selective_mutation_frequency=10, random_seed=None):
        self.config = config
        self.op_data = op_data
        self.crossover = crossover
        self.mutation = mutation
        self.selection = selection
        self.local_search = local_search
        self.local_search_methods = local_search if local_search else []
        self.pso = pso
        self.selective_mutation = selective_mutation
        self.elite_ratio = elite_ratio
        self.best_time = None
        self.ga_engines = ga_engines
        self.island_mode = island_mode
        self.migration_frequency = migration_frequency
        self.dataset_filename = dataset_filename
        self.local_search_frequency = local_search_frequency
        self.selective_mutation_frequency = selective_mutation_frequency
        self.random_seed = random_seed
        self.local_search_top_percentage = 0.05
        self.current_generation = 0
        self.execution_time = 0
        self.best_time = 0

        if initialization_mode == '2':
            self.population = Population.from_mio(config, op_data, dataset_filename, random_seed=random_seed)
        elif initialization_mode == '3':
            self.population = Population.from_giffler_thompson(config, op_data, dataset_filename, random_seed=random_seed)
        else:
            self.population = Population(config, op_data, random_seed=random_seed)

    def evolve(self, index, sync_generation, sync_lock, events=None):
        try:
            env = simpy.Environment()  # 여기에서 환경을 생성합니다.
            best_individual = None
            best_crossover = None
            best_mutation = None
            all_generations = []
            execution_time = 0
            best_time = 0
         
            all_generations = []
            start_time = time.time()
            best_individual = None
            best_fitness = float('inf')

            new_populations = [[] for _ in range(len(self.ga_engines))]

            while sync_generation[index] < self.config.generations:
                print(f"GA{index+1}_Evaluating generation {sync_generation[index]}")

                self.population.evaluate(self.config.target_makespan, multi_objective=True)
                print(f"GA{index+1} Population: {self.population is not None}")
                print(f"GA{index+1} Best Individual: {best_individual is not None}")
                num_elites = int(self.elite_ratio * len(self.population.individuals))
                elites = sorted(self.population.individuals, key=lambda ind: ind.fitness, reverse=True)[:num_elites]

                self.population.select(self.selection)
                self.population.crossover(self.crossover)
                self.population.mutate(self.mutation)

                for elite in elites:
                    random_index = random.randint(0, len(self.population.individuals) - 1)
                    self.population.individuals[random_index] = elite

                if sync_generation[index] > 0 and self.selective_mutation and sync_generation[index] % self.selective_mutation_frequency == 0:
                    print(f'GA{index+1}_Selective Mutation 전반부 적용')
                    self.selective_mutation.mutate(self.population.individuals, self.config)

                if sync_generation[index] > 0 and sync_generation[index] % self.local_search_frequency == 0:
                    print(f"GA{index+1}_Applying local search")
                    top_individuals = sorted(self.population.individuals, key=lambda ind: ind.fitness, reverse=True)[:int(len(self.population.individuals) * self.local_search_top_percentage)]
                    for method in self.local_search_methods:
                        for i in range(len(top_individuals)):
                            individual = top_individuals[i]
                            optimized_ind = method.optimize(individual, self.config)
                            if optimized_ind in self.population.individuals:
                                self.population.individuals[self.population.individuals.index(individual)] = optimized_ind
                            else:
                                self.population.individuals[random.randint(0, len(self.population.individuals) - 1)] = optimized_ind
                                print(f"GA{index+1}_Added optimized individual to population.")

                if self.island_mode != 1 and sync_generation[index] % self.migration_frequency == 0 and sync_generation[index] != 0 and self.ga_engines:
                    if events:
                        for event in events:
                            event.set()
                        for event in events:
                            event.wait()
                        for event in events:
                            event.clear()

                    print(f'island_mode{self.island_mode}')
                    print(f"GA{index+1}_Preparing for migration at generation {sync_generation[index]}")
                    if self.island_mode == 2:
                        print(f"GA{index+1}_Migration 중 (순차) at generation {sync_generation[index]}")
                        migration_order = [(i + 1) % len(self.ga_engines) for i in range(len(self.ga_engines))]
                        print(f"Migration order: {migration_order}")
                    elif self.island_mode == 3:
                        print(f"GA{index+1}_Migration 중 (랜덤) at generation {sync_generation[index]}")
                        migration_order = list(range(len(self.ga_engines)))
                        random.shuffle(migration_order)
                        print(f'migration_order:{migration_order}')
                    else:
                        migration_order = range(len(self.ga_engines))
                    print(f'island_mode{self.island_mode}')

                    print(f"Migration order: {migration_order}")

                    new_populations[index] = sorted(self.population.individuals, key=lambda ind: ind.fitness, reverse=True)[:max(1, len(self.population.individuals) // 5)]

                    for i in range(len(self.ga_engines)):
                        target_index = migration_order[i]
                        if target_index != i:
                            if new_populations[i]:
                                elites_indices = sorted(range(len(self.ga_engines[target_index].population.individuals)), key=lambda idx: self.ga_engines[target_index].population.individuals[idx].fitness, reverse=True)[:len(new_populations[i])]
                                for j in range(len(new_populations[i])):
                                    migrating_individual = copy.deepcopy(new_populations[i][j])
                                    replaced_individual = self.ga_engines[target_index].population.individuals[elites_indices[j]]
                                    self.ga_engines[target_index].population.individuals[elites_indices[j]] = migrating_individual
                                    print(f"Migrating from GA{i+1} to GA{target_index+1}: Migrating individual fitness: {migrating_individual.fitness}, Replaced individual fitness: {replaced_individual.fitness}")

                    self.population.individuals = sorted(self.population.individuals, key=lambda ind: ind.fitness, reverse=True)
                    best_individual = min(self.population.individuals, key=lambda ind: ind.makespan)
                    best_fitness = best_individual.makespan
                    print(f"GA{index+1}_Best individual after migration: Fitness: {best_fitness}, Sequence: {best_individual.seq}")

                print(f"GA{index+1}_Generation {sync_generation[index]} evaluated")
                current_best = min(self.population.individuals, key=lambda ind: ind.makespan)
                if current_best.makespan < best_fitness:
                    best_individual = current_best
                    best_fitness = current_best.makespan
                    print(f"GA{index+1}_Best fitness at generation {sync_generation[index]}: {best_fitness}")

                    if self.best_time is None or current_best.makespan < self.config.target_makespan:
                        self.best_time = time.time() - start_time

                generation_data = [(ind.seq, ind.makespan) for ind in self.population.individuals]
                all_generations.append((sync_generation[index], generation_data))

                if best_individual is not None and best_individual.makespan <= self.config.target_makespan:
                    print(f"GA{index+1}_Stopping early as best makespan {best_individual.makespan} is below target {self.config.target_makespan}.")
                    break

                with sync_lock:
                    sync_generation[index] += 1

            if self.pso:
                print(f"GA{index+1}_Applying PSO after all generations")
                for i in range(len(self.population.individuals)):
                    individual = self.population.individuals[i]
                    optimized_individual = self.apply_pso(individual)
                    self.population.individuals[i] = optimized_individual

            end_time = time.time()
            execution_time = end_time - start_time

            if best_individual is not None and hasattr(best_individual, 'monitor'):
                best_individual.monitor.save_event_tracer(self.config.filename['log'])
            else:
                print("No valid best individual or monitor to save the event tracer.")
            return best_individual, self.crossover, self.mutation, all_generations, execution_time, self.best_time

        except Exception as e:
            print(f"Exception during evolution in GA{index+1}: {e}")
            return None, None, None, [], 0, None

    def apply_local_search(self, individual):
        best_individual = copy.deepcopy(individual)
        for method in self.local_search_methods:
            improved_individual = method.optimize(best_individual, self.config)
            if improved_individual.makespan < best_individual.makespan:
                best_individual = improved_individual
        return best_individual

    def apply_pso(self, individual):
        best_individual = copy.deepcopy(individual)
        optimized_individual = self.pso.optimize(best_individual, self.config)
        if optimized_individual.makespan < best_individual.makespan:
            best_individual = optimized_individual
        return best_individual

    def save_csv(self, all_generations, execution_time, file_path):
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        file_path_with_timestamp = file_path.replace('.csv', f'_{timestamp}.csv')

        with open(file_path_with_timestamp, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['Generation', 'Chromosome', 'Makespan'])
            for generation, individuals in all_generations:
                for seq, makespan in individuals:
                    csvwriter.writerow([generation, seq, makespan])
