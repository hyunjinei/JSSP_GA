# Local_Search/TabuSearch.py

import copy
from collections import deque
import random

class TabuSearch:
    def __init__(self, tabu_tenure=10, iterations=100):
        self.tabu_tenure = tabu_tenure
        self.iterations = iterations

    def optimize(self, individual, config):
        print(f"Tabu Search 시작")
        # print(f"Tabu Search 시작 - Initial Individual: {individual.seq}, Makespan: {individual.makespan}, Fitness: {individual.fitness}")
        best_solution = copy.deepcopy(individual)
        best_makespan = individual.makespan
        tabu_list = []
        tabu_list.append(copy.deepcopy(individual.seq))

        for iteration in range(self.iterations):
            neighbors = self.get_neighbors(individual)
            # print(f"Iteration {iteration + 1} - Number of Neighbors: {len(neighbors)}")
            neighbors = [n for n in neighbors if n.seq not in tabu_list]

            if not neighbors:
                # print("No valid neighbors found, terminating early.")
                break

            current_solution = min(neighbors, key=lambda ind: ind.makespan)
            current_makespan = current_solution.makespan

            if current_makespan < best_makespan:
                best_solution = copy.deepcopy(current_solution)
                best_makespan = current_makespan

            tabu_list.append(copy.deepcopy(current_solution.seq))
            if len(tabu_list) > self.tabu_tenure:
                tabu_list.pop(0)
            # print(f"Iteration {iteration + 1} - Current Best Makespan: {best_makespan}, Fitness: {best_solution.fitness}")

        # 최적화 후 염색체, makespan, fitness 출력
        print(f"Tabu Search 완료")
        # print(f"Tabu Search 완료 - Optimized Individual: {best_solution.seq}, Makespan: {best_solution.makespan}, Fitness: {best_solution.fitness}")
        return best_solution

    def get_neighbors(self, individual):
        neighbors = []
        seq = individual.seq
        for i in range(len(seq) - 1):
            for j in range(i + 1, len(seq)):
                neighbor_seq = seq[:]
                neighbor_seq[i], neighbor_seq[j] = neighbor_seq[j], neighbor_seq[i]
                neighbor = copy.deepcopy(individual)
                neighbor.seq = neighbor_seq
                neighbor.makespan, neighbor.mio_score = neighbor.evaluate(neighbor.machine_order)
                neighbor.calculate_fitness(neighbor.config.target_makespan)
                # print(f"Neighbor: {neighbor.seq}, Makespan: {neighbor.makespan}, Fitness: {neighbor.fitness}")
                neighbors.append(neighbor)
        return neighbors
