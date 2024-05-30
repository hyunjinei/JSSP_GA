import sys
import os
import random
import time
import copy
import csv
from concurrent.futures import ThreadPoolExecutor

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from GAS.Population import Population
from Local_Search.TabuSearch import TabuSearch

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

        print(f"Selected source island {source_island_idx + 1} and target island {target_island_idx + 1}")

        # 상위 10% 개체를 찾기
        top_10_percent = sorted(source_island.population.individuals, key=lambda ind: ind.fitness, reverse=True)[:max(1, len(source_island.population.individuals) // 10)]

        print(f"Top 10% individuals selected from island {source_island_idx + 1}")

        # 대상 섬에서 무작위 개체와 교체
        for best_individual in top_10_percent:
            replacement_idx = random.randint(0, len(target_island.population.individuals) - 1)
            print(f"Replacing individual at index {replacement_idx} on island {target_island_idx + 1} with an individual from island {source_island_idx + 1}")
            target_island.population.individuals[replacement_idx] = copy.deepcopy(best_individual)

        print(f"Migrating top 10% individuals from Island {source_island_idx + 1} to Island {target_island_idx + 1}")


class GAEngine:
    def __init__(self, config, op_data, crossover, mutation, selection, local_search, elite_ratio=0.1, ga_engines=None, island_mode=None, migration_frequency=None):
        self.config = config
        self.op_data = op_data
        self.crossover = crossover
        self.mutation = mutation
        self.selection = selection
        self.local_search = local_search
        self.elite_ratio = elite_ratio
        self.population = Population(config, op_data)
        self.best_time = None
        self.ga_engines = ga_engines
        self.island_mode = island_mode
        self.migration_frequency = migration_frequency

    def evolve(self):
        try:
            all_generations = []
            start_time = time.time()
            best_individual = None
            best_fitness = float('inf')

            for generation in range(self.config.generations):
                print(f"Evaluating generation {generation}")
                self.population.evaluate(self.config.target_makespan)  # Ensure target_makespan is passed

                # Elitism: 최상의 해를 보존합니다.
                num_elites = int(self.elite_ratio * len(self.population.individuals))
                elites = sorted(self.population.individuals, key=lambda ind: ind.fitness, reverse=True)[:num_elites]

                self.population.select(self.selection)
                self.population.crossover(self.crossover)
                self.population.mutate(self.mutation)

                # Apply local search to each individual in the population
                if self.local_search:
                    print("Local Search 시작")
                    for i in range(len(self.population.individuals)):
                        optimized_ind = self.local_search.optimize(self.population.individuals[i], self.config)
                        self.population.individuals[i] = optimized_ind

                # Elitism: 최상의 해를 새로운 Population에 추가합니다.
                self.population.individuals[:num_elites] = elites

                print(f"Generation {generation} evaluated")
                current_best = min(self.population.individuals, key=lambda ind: ind.makespan)
                if current_best.makespan < best_fitness:
                    best_individual = current_best
                    best_fitness = current_best.makespan
                    print(f"Best fitness at generation {generation}: {best_fitness}")

                    if self.best_time is None or current_best.makespan < self.config.target_makespan:
                        self.best_time = time.time() - start_time  # 최소 makespan이 처음으로 달성된 시간 기록

                generation_data = [(ind.seq, ind.makespan) for ind in self.population.individuals]
                all_generations.append((generation, generation_data))

                # Migration 수행
                if self.migration_frequency and (generation + 1) % self.migration_frequency == 0 and self.ga_engines:
                    print(f"Preparing for migration at generation {generation + 1}")
                    if self.island_mode == '2':
                        print(f"Migration 중 (순차) at generation {generation + 1}")
                        migration_order = list(range(len(self.ga_engines)))
                        migrate_top_10_percent(self.ga_engines, migration_order, self.island_mode)
                    elif self.island_mode == '3':
                        print(f"Migration 중 (랜덤) at generation {generation + 1}")
                        migration_order = random.sample(range(len(self.ga_engines)), len(self.ga_engines))
                        migrate_top_10_percent(self.ga_engines, migration_order, self.island_mode)


                # 목표 Makespan에 도달하면 멈춤
                if best_individual.makespan <= self.config.target_makespan:
                    print(f"Stopping early as best makespan {best_individual.makespan} is below target {self.config.target_makespan}.")
                    break

            end_time = time.time()
            execution_time = end_time - start_time

            if best_individual is not None and hasattr(best_individual, 'monitor'):
                best_individual.monitor.save_event_tracer(self.config.filename['log'])
            else:
                print("No valid best individual or monitor to save the event tracer.")
            return best_individual, self.crossover, self.mutation, all_generations, execution_time, self.best_time

        except Exception as e:
            print(f"Exception during evolution: {e}")
            return None, None, None, [], 0, None


    def save_csv(self, all_generations, execution_time, file_path):
        with open(file_path, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['Generation', 'Chromosome', 'Makespan'])
            
            for generation, generation_data in all_generations:
                for chromosome, makespan in generation_data:
                    csvwriter.writerow([generation, ' -> '.join(map(str, chromosome)), makespan])
            
            csvwriter.writerow(['Execution Time', '', execution_time])
