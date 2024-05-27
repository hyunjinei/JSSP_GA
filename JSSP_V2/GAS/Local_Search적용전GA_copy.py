# GA.py

import sys
import os
import random
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from GAS.Population import Population

class GAEngine:
    def __init__(self, config, op_data, crossover, mutation, selection):
        self.config = config
        self.op_data = op_data
        self.crossover = crossover
        self.mutation = mutation
        self.selection = selection
        self.population = Population(config, op_data)

    def evolve(self):
        best_individual = None
        best_fitness = float('inf')
        for generation in range(self.config.generations):
            self.population.evaluate()
            self.population.select(self.selection)
            self.population.crossover(self.crossover)
            self.population.mutate(self.mutation)
            print(f"Generation {generation}")
            current_best = min(self.population.individuals, key=lambda ind: ind.makespan)
            if current_best.makespan < best_fitness:
                best_individual = current_best
                best_fitness = current_best.makespan
                print(f"Best fitness at generation {generation}: {best_fitness}")
        
        # Save the final population machine log
        best_individual.monitor.save_event_tracer(self.config.filename['log'])
        return best_individual, self.crossover, self.mutation  # Return crossover and mutation
