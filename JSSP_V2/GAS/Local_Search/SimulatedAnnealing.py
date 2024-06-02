import copy
import math
import random

class SimulatedAnnealing:
    def __init__(self, initial_temp=1000, cooling_rate=0.95, min_temp=1, iterations=100):
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.min_temp = min_temp
        self.iterations = iterations

    def optimize(self, individual, config):
        print(f"Simulated Annealing 시작")       
        # print(f"Simulated Annealing 시작 - Initial Individual: {individual.seq}, Makespan: {individual.makespan}, Fitness: {individual.fitness}")
        best_solution = copy.deepcopy(individual)
        current_solution = copy.deepcopy(individual)
        best_makespan = individual.makespan
        current_makespan = individual.makespan
        temp = self.initial_temp
        iteration = 0

        while temp > self.min_temp and iteration < self.iterations:
            neighbor = self.get_random_neighbor(current_solution)
            neighbor_makespan = neighbor.makespan

            if neighbor_makespan < best_makespan:
                best_solution = copy.deepcopy(neighbor)
                best_makespan = neighbor_makespan

            if neighbor_makespan < current_makespan or \
                    math.exp((current_makespan - neighbor_makespan) / temp) > random.random():
                current_solution = neighbor
                current_makespan = neighbor_makespan

            temp *= self.cooling_rate
            iteration += 1
            # print(f"Iteration {iteration} - Temperature: {temp}, Current Makespan: {current_makespan}, Best Makespan: {best_makespan}")

        print(f"Simulated Annealing 완료")
        # print(f"Simulated Annealing 완료 - Optimized Individual: {best_solution.seq}, Makespan: {best_solution.makespan}, Fitness: {best_solution.fitness}")
        return best_solution

    def get_random_neighbor(self, individual):
        neighbor = copy.deepcopy(individual)
        i, j = random.sample(range(len(neighbor.seq)), 2)
        neighbor.seq[i], neighbor.seq[j] = neighbor.seq[j], neighbor.seq[i]
        neighbor.makespan, neighbor.mio_score = neighbor.evaluate(neighbor.machine_order)
        neighbor.calculate_fitness(neighbor.config.target_makespan)
        # print(f"Neighbor: {neighbor.seq}, Makespan: {neighbor.makespan}, Fitness: {neighbor.fitness}")
        return neighbor
