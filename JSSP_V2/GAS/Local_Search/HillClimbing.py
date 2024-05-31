# Local_Search/HillClimbing.py

import copy

class HillClimbing:
    def __init__(self):
        pass

    def optimize(self, individual, config):
        print("Hill Climbing 시작")
        current_solution = copy.deepcopy(individual)
        current_fitness = current_solution.calculate_fitness(config.target_makespan)  # 수정
        iteration = 0

        while True:
            neighbors = self.get_neighbors(current_solution, config)
            next_solution = min(neighbors, key=lambda ind: ind.fitness)

            if next_solution.fitness >= current_fitness:
                print(f"Iteration {iteration}: No improvement (current best: {current_fitness})")
                break

            print(f"Iteration {iteration}: Improved to {current_fitness}")
            current_solution = next_solution
            current_fitness = next_solution.fitness
            iteration += 1

        print("Hill Climbing 종료")
        return current_solution

    def get_neighbors(self, solution, config):
        neighbors = []
        for i in range(len(solution.seq)):
            for j in range(i + 1, len(solution.seq)):
                neighbor = copy.deepcopy(solution)
                neighbor.seq[i], neighbor.seq[j] = neighbor.seq[j], neighbor.seq[i]
                neighbor.fitness = neighbor.calculate_fitness(config.target_makespan)  # 이웃의 적합도를 계산합니다. 수정
                neighbors.append(neighbor)
        return neighbors
