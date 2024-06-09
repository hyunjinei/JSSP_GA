import sys
import os
import random
import time
import copy
import csv
from concurrent.futures import ProcessPoolExecutor

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from GAS.Population import Population
from Local_Search.TabuSearch import TabuSearch
from Data.Dataset.Dataset import Dataset
from Meta.PSO import PSO  # PSO를 추가합니다
from GAS.Mutation.SelectiveMutation import SelectiveMutation
from Local_Search.HillClimbing import HillClimbing  # import 추가
from Local_Search.SimulatedAnnealing import SimulatedAnnealing
from Local_Search.GifflerThompson_LS import GifflerThompson_LS
from multiprocessing import Pool

# migrate_top_10_percent 함수 정의
def migrate_top_10_percent(ga_engines, migration_order, island_mode):
    num_islands = len(ga_engines)
    for source_island_idx in range(num_islands):
        if island_mode == '2':  # Sequential Migration
            target_island_idx = (source_island_idx + 1) % num_islands
        elif island_mode == '3':  # Random Migration
            target_island_idx = migration_order[source_island_idx]

        source_island = ga_engines[source_island_idx]
        target_island = ga_engines[target_island_idx]

        # 상위 10% 개체를 찾기
        top_10_percent = sorted(source_island.population.individuals, key=lambda ind: ind.fitness, reverse=True)[:max(1, len(source_island.population.individuals) // 10)]
        print(f"GA{source_island_idx+1} 상위 10% 개체:")
        for ind in top_10_percent:
            print(f"  Sequence: {ind.seq}, Fitness: {ind.fitness}, Makespan: {ind.makespan}")

        # 대상 섬에서 무작위 개체와 교체
        for best_individual in top_10_percent:
            replacement_idx = random.randint(0, len(target_island.population.individuals) - 1)
            replaced_individual = target_island.population.individuals[replacement_idx]
            print(f"GA{target_island_idx+1}에서 교체된 개체:")
            print(f"  Sequence: {replaced_individual.seq}, Fitness: {replaced_individual.fitness}, Makespan: {replaced_individual.makespan}")

            target_island.population.individuals[replacement_idx] = copy.deepcopy(best_individual)
            print(f"Migration: {best_individual.seq} from GA {source_island_idx+1} to GA {target_island_idx+1}")

class GAEngine:
    def __init__(self, config, op_data, crossover, mutation, selection, local_search=None, pso=None, selective_mutation=None, elite_ratio=0.1, ga_engines=None, island_mode=1, migration_frequency=10, initialization_mode='1', dataset_filename=None, initial_population=None, local_search_frequency=2, selective_mutation_frequency=10, random_seed=None):
        self.config = config
        self.op_data = op_data
        self.crossover = crossover
        self.mutation = mutation
        self.selection = selection
        self.local_search = local_search  # local_search 속성 설정
        self.local_search_methods = local_search if local_search else []  # local_search를 리스트로 변경
        self.pso = pso  # PSO 추가
        self.selective_mutation = selective_mutation  # SelectiveMutation 추가
        self.elite_ratio = elite_ratio
        self.best_time = None
        self.ga_engines = ga_engines
        self.island_mode = island_mode
        self.migration_frequency = migration_frequency
        self.dataset_filename = dataset_filename  # 새로운 인자 추가
        self.local_search_frequency = local_search_frequency  # 로컬 서치 주기 설정
        self.selective_mutation_frequency = selective_mutation_frequency # selective_mutation 주기 설정
        self.random_seed = random_seed  # 랜덤 시드 설정
        # 상위 10% 개체에만 Local Search 적용
        self.local_search_top_percentage = 0.2  


        if initialization_mode == '2':
            self.population = Population.from_mio(config, op_data, dataset_filename, random_seed=random_seed)
        elif initialization_mode == '3':
            self.population = Population.from_giffler_thompson(config, op_data, dataset_filename, random_seed=random_seed)
        else:
            self.population = Population(config, op_data, random_seed=random_seed)

    def evolve(self, index):
        try:
            all_generations = []
            start_time = time.time()
            best_individual = None
            best_fitness = float('inf')

            for generation in range(self.config.generations):
                print(f"GA{index+1}_Evaluating generation {generation}")
                                
                self.population.evaluate(self.config.target_makespan)  # Ensure target_makespan is passed

                # Elitism: 최상의 해를 보존합니다.
                num_elites = int(self.elite_ratio * len(self.population.individuals))
                elites = sorted(self.population.individuals, key=lambda ind: ind.fitness, reverse=True)[:num_elites]

                self.population.select(self.selection)
                self.population.crossover(self.crossover)
                self.population.mutate(self.mutation)

                # Elitism: 최상의 해를 새로운 Population에 랜덤하게 추가합니다.
                for elite in elites:
                    random_index = random.randint(0, len(self.population.individuals) - 1)
                    self.population.individuals[random_index] = elite

                # Selective Mutation 적용
                if generation > 0 and self.selective_mutation and generation % self.selective_mutation_frequency == 0:
                    print(f"GA{index+1}_Selective Mutation 전반부 적용")
                    self.selective_mutation.mutate(self.population.individuals, self.config)
                    
                print(f"GA{index+1}_Generation {generation} evaluated")
                current_best = min(self.population.individuals, key=lambda ind: ind.makespan)
                if current_best.makespan < best_fitness:
                    best_individual = current_best
                    best_fitness = current_best.makespan
                    print(f"GA{index+1}_Best fitness at generation {generation}: {best_fitness}")

                    if self.best_time is None or current_best.makespan < self.config.target_makespan:
                        self.best_time = time.time() - start_time  # 최소 makespan이 처음으로 달성된 시간 기록

                generation_data = [(ind.seq, ind.makespan) for ind in self.population.individuals]
                all_generations.append((generation, generation_data))

                # 목표 Makespan에 도달하면 멈춤
                if best_individual is not None and best_individual.makespan <= self.config.target_makespan:
                    print(f"GA{index+1}_Stopping early as best makespan {best_individual.makespan} is below target {self.config.target_makespan}.")
                    break

                # Migration 수행
                if self.migration_frequency and (generation + 1) % self.migration_frequency == 0 and self.ga_engines:
                    print(f"GA{index+1}_Preparing for migration at generation {generation + 1}")
                    if self.island_mode == '2':
                        print(f"GA{index+1}_Migration 중 (순차) at generation {generation + 1}")
                        migration_order = list(range(len(self.ga_engines)))
                        migrate_top_10_percent(self.ga_engines, migration_order, self.island_mode)
                    elif self.island_mode == '3':
                        print(f"GA{index+1}_Migration 중 (랜덤) at generation {generation + 1}")
                        migration_order = random.sample(range(len(self.ga_engines)), len(self.ga_engines))
                        migrate_top_10_percent(self.ga_engines, migration_order, self.island_mode)

                # Apply local search at specified frequency and only to top individuals
                if generation > 0 and generation % self.local_search_frequency == 0:
                    print(f"GA{index+1}_Applying local search")
                    top_individuals = sorted(self.population.individuals, key=lambda ind: ind.fitness, reverse=True)[:int(len(self.population.individuals) * self.local_search_top_percentage)]
                    for method in self.local_search_methods:
                        for i in range(len(top_individuals)):
                            individual = top_individuals[i]
                            optimized_ind = method.optimize(individual, self.config)
                            if optimized_ind in self.population.individuals:
                                self.population.individuals[self.population.individuals.index(individual)] = optimized_ind
                            else:
                                # 최적화된 개체가 인구에 없으면 추가합니다.
                                self.population.individuals[random.randint(0, len(self.population.individuals) - 1)] = optimized_ind
                                print(f"GA{index+1}_Added optimized individual to population.")
                            
            # 모든 generation 완료 후 PSO 진행
            if self.pso:
                print(f"GA{index+1}_Applying PSO after all generations")
                for i in range(len(self.population.individuals)):
                    individual = self.population.individuals[i]
                    optimized_ind = self.apply_pso(individual)
                    self.population.individuals[i] = optimized_ind

            end_time = time.time()
            execution_time = end_time - start_time

            if best_individual is not None and hasattr(best_individual, 'monitor'):
                best_individual.monitor.save_event_tracer(self.config.filename['log'])
            else:
                print("No valid best individual or monitor to save the event tracer.")
            return best_individual, self.crossover, self.mutation, all_generations, execution_time, self.best_time

        except Exception as e:
            print(f"GA{index+1}_Exception during evolution: {e}")
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
        with open(file_path, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['Generation', 'Chromosome', 'Makespan'])
            
            for generation, generation_data in all_generations:
                for chromosome, makespan in generation_data:
                    csvwriter.writerow([generation, ' -> '.join(map(str, chromosome)), makespan])
            
            csvwriter.writerow(['Execution Time', '', execution_time])
