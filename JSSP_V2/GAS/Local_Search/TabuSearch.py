# Local_Search/TabuSearch.py

import copy
from collections import deque
from GAS.Individual import Individual

class TabuSearch:
    def __init__(self, tabu_size=10, max_iter=100, no_improve_limit=20):
        self.tabu_size = tabu_size
        self.max_iter = max_iter
        self.no_improve_limit = no_improve_limit

    def optimize(self, individual, config):
        print("Tabu Search 시작")
        current_solution = copy.deepcopy(individual)
        current_fitness = current_solution.fitness
        best_solution = copy.deepcopy(current_solution)
        best_fitness = current_fitness

        tabu_list = deque(maxlen=self.tabu_size)
        tabu_list.append(current_solution.seq)

        no_improve_count = 0

        for iteration in range(self.max_iter):
            neighbors = self.get_neighbors(current_solution, config.target_makespan)
            neighbors = [n for n in neighbors if n.seq not in tabu_list or n.fitness < best_fitness]  # Aspiration Criteria

            if not neighbors:
                break

            next_solution = max(neighbors, key=lambda ind: ind.fitness)
            current_solution = next_solution
            current_fitness = next_solution.fitness
            tabu_list.append(current_solution.seq)

            if next_solution.fitness > best_fitness:
                best_solution = copy.deepcopy(next_solution)
                best_fitness = next_solution.fitness
                no_improve_count = 0
                print(f"Iteration {iteration}: Improved to {best_fitness}")
            else:
                no_improve_count += 1
                print(f"Iteration {iteration}: No improvement (current best: {best_fitness})")

            if no_improve_count >= self.no_improve_limit:
                print(f"No improvement limit reached at iteration {iteration}. Ending Tabu Search.")
                break

        print("Tabu Search 종료")
        return best_solution

    def get_neighbors(self, solution, target_makespan):
        neighbors = []
        for i in range(len(solution.seq)):
            for j in range(i + 1, len(solution.seq)):
                neighbor = copy.deepcopy(solution)
                neighbor.seq[i], neighbor.seq[j] = neighbor.seq[j], neighbor.seq[i]
                neighbor.calculate_fitness(target_makespan)  # Ensure fitness is recalculated for neighbors
                neighbors.append(neighbor)
        return neighbors
