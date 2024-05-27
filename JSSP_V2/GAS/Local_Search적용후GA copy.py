# GA.py
import sys
import os
import random
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from GAS.Population import Population

class GAEngine:
    def __init__(self, config, op_data, crossover, mutation, selection, local_search=None):
        self.config = config
        self.op_data = op_data
        self.crossover = crossover
        self.mutation = mutation
        self.selection = selection
        self.local_search = local_search
        self.population = Population(config, op_data)

    def evolve(self):
        best_individual = None
        best_fitness = float('inf')
        for generation in range(self.config.generations):
            self.population.evaluate()
            self.population.select(self.selection)
            self.population.crossover(self.crossover)
            self.population.mutate(self.mutation)

            # Apply local search to each individual in the population
            if self.local_search:
                for i in range(len(self.population.individuals)):
                    optimized_ind = self.local_search.optimize(self.population.individuals[i], self.config)
                    if optimized_ind.fitness is not None:
                        self.population.individuals[i] = optimized_ind  # Replace with optimized individual if valid

            print(f"Generation {generation}")
            current_best = min(self.population.individuals, key=lambda ind: ind.makespan)
            if current_best.makespan < best_fitness:
                best_individual = current_best
                best_fitness = current_best.makespan
                print(f"Best fitness at generation {generation}: {best_fitness}")
        
        # Save the final population machine log
        best_individual.monitor.save_event_tracer(self.config.filename['log'])
        return best_individual, self.crossover, self.mutation  # Return crossover and mutation

