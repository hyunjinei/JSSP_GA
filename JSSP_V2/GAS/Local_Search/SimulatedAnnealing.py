# SimulatedAnnealing.py
import copy
import math
import random

class SimulatedAnnealing:
    def __init__(self, initial_temp=1000, cooling_rate=0.95, min_temp=1, max_iter=10):
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.min_temp = min_temp
        self.max_iter = max_iter  # 최대 반복 횟수 추가

    def optimize(self, individual, config):
        current_solution = copy.deepcopy(individual)
        current_fitness = current_solution.calculate_fitness(config.target_makespan)  # 초기 적합도 계산

        best_solution = copy.deepcopy(current_solution)
        best_fitness = current_fitness

        temp = self.initial_temp
        iterations = 0  # 반복 횟수 초기화

        while temp > self.min_temp and iterations < self.max_iter:  # 최대 반복 횟수 조건 추가
            neighbor = self.get_random_neighbor(current_solution, config)
            neighbor_fitness = neighbor.calculate_fitness(config.target_makespan)  # 이웃의 적합도 계산

            if neighbor_fitness < best_fitness:  # 더 나은 적합도를 찾으면 갱신합니다.
                best_solution = copy.deepcopy(neighbor)
                best_fitness = neighbor_fitness

            if neighbor_fitness < current_fitness or \
                    math.exp((current_fitness - neighbor_fitness) / temp) > random.random():
                current_solution = neighbor
                current_fitness = neighbor_fitness

            temp *= self.cooling_rate
            iterations += 1  # 반복 횟수 증가

        return best_solution

    def get_random_neighbor(self, solution, config):
        neighbor = copy.deepcopy(solution)
        i, j = random.sample(range(len(solution.seq)), 2)
        neighbor.seq[i], neighbor.seq[j] = neighbor.seq[j], neighbor.seq[i]
        neighbor.calculate_fitness(config.target_makespan)  # 이웃의 적합도 계산
        return neighbor
