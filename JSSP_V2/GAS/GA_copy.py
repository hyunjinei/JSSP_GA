import sys
import os
import random
import time
import copy
import csv
from concurrent.futures import ThreadPoolExecutor

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from GAS.Population import Population

class GAEngine:
    def __init__(self, config, op_data, crossover, mutation, selection, local_search=None, elite_ratio=0.1):
        self.config = config
        self.op_data = op_data
        self.crossover = crossover
        self.mutation = mutation
        self.selection = selection
        self.local_search = local_search
        self.population = Population(config, op_data)
        self.elite_ratio = elite_ratio
        self.best_time = None

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