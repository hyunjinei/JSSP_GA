# Local_Search/TabuSearch.py

import copy
from collections import deque
import random
# 기존     def __init__(self, tabu_tenure=5, iterations=20, max_neighbors=10):
class TabuSearch:
    def __init__(self, tabu_tenure=5, iterations=10, max_neighbors=10):
        self.tabu_tenure = tabu_tenure
        self.iterations = iterations
        self.max_neighbors = max_neighbors  # 최대 이웃 개수를 추가
        self.stop_search = False  # 종료 조건 플래그 추가

    def optimize(self, individual, config):
        # print(f"Tabu Search 시작")
        # print(f"Tabu Search 시작 - Initial Individual: {individual.seq}, Makespan: {individual.makespan}, Fitness: {individual.fitness}")
        best_solution = copy.deepcopy(individual)
        best_makespan = individual.makespan
        tabu_list = []
        tabu_list.append(copy.deepcopy(individual.seq))

        for iteration in range(self.iterations):
            neighbors = self.get_neighbors(individual, config)
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

            # 목표 Makespan에 도달하면 Local Search 종료
            if best_solution.fitness >= 1.0:
                print(f"Stopping early as fitness {best_solution.fitness} is 1.0 or higher.")
                self.stop_search = True
                return best_solution

        # 최적화 후 염색체, makespan, fitness 출력
        # print(f"Tabu Search 완료")
        # print(f"Tabu Search 완료 - Optimized Individual: {best_solution.seq}, Makespan: {best_solution.makespan}, Fitness: {best_solution.fitness}")
        return best_solution

    def get_neighbors(self, individual, config):
        neighbors = []
        seq = individual.seq
        for i in range(len(seq) - 1):
            for j in range(i + 1, len(seq)):
                if len(neighbors) >= self.max_neighbors:  # 최대 이웃 개수 조건 추가
                    return neighbors
                neighbor_seq = seq[:]
                neighbor_seq[i], neighbor_seq[j] = neighbor_seq[j], neighbor_seq[i]
                neighbor = self.create_new_individual(individual, neighbor_seq, config)
                # print(f"Neighbor: {neighbor.seq}, Makespan: {neighbor.makespan}, Fitness: {neighbor.fitness}")
                neighbors.append(neighbor)
        return neighbors

    def create_new_individual(self, individual, new_seq, config):
        new_individual = copy.deepcopy(individual)
        new_individual.seq = new_seq
        new_individual.job_seq = new_individual.get_repeatable()
        new_individual.feasible_seq = new_individual.get_feasible()
        new_individual.machine_order = new_individual.get_machine_order()
        new_individual.makespan, new_individual.mio_score = new_individual.evaluate(new_individual.machine_order)
        new_individual.calculate_fitness(config.target_makespan)
        return new_individual

    def ensure_valid_sequence(self, seq, config):
        num_jobs = config.n_job
        num_machines = config.n_machine
        job_counts = {job: 0 for job in range(num_jobs)}
        valid_seq = []

        for operation in seq:
            job = operation // num_machines
            if job_counts[job] < num_machines:
                valid_seq.append(job * num_machines + job_counts[job])
                job_counts[job] += 1

        for job in range(num_jobs):
            while job_counts[job] < num_machines:
                valid_seq.append(job * num_machines + job_counts[job])
                job_counts[job] += 1

        return valid_seq
