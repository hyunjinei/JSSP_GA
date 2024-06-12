import copy
import math
import random

class SimulatedAnnealing:
    def __init__(self, initial_temp=1000, cooling_rate=0.95, min_temp=1, iterations=10):
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.min_temp = min_temp
        self.iterations = iterations
        self.stop_search = False  # 종료 조건 플래그 추가

    def optimize(self, individual, config):
        # print(f"Simulated Annealing 시작 - Initial Individual: {individual.seq}, Makespan: {individual.makespan}, Fitness: {individual.fitness}")
        best_solution = copy.deepcopy(individual)
        current_solution = copy.deepcopy(individual)
        best_makespan = individual.makespan
        current_makespan = individual.makespan
        temp = self.initial_temp
        iteration = 0

        while temp > self.min_temp and iteration < self.iterations:
            neighbor = self.get_random_neighbor(current_solution, config)
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

            # 목표 Makespan에 도달하면 Local Search 종료
            if best_solution.fitness >= 1.0:
                print(f"Stopping early as fitness {best_solution.fitness} is 1.0 or higher.")
                self.stop_search = True
                break

        # print(f"Simulated Annealing 완료 - Optimized Individual: {best_solution.seq}, Makespan: {best_solution.makespan}, Fitness: {best_solution.fitness}")
        return best_solution

    def get_random_neighbor(self, individual, config):
        new_seq = copy.deepcopy(individual.seq)
        i, j = random.sample(range(len(new_seq)), 2)
        new_seq[i], new_seq[j] = new_seq[j], new_seq[i]
        new_seq = self.ensure_valid_sequence(new_seq, config)
        
        # Create a new individual and recompute makespan and fitness
        neighbor = self.create_new_individual(individual, new_seq, config)
        neighbor.calculate_fitness(neighbor.config.target_makespan)
        # print(f"Neighbor: {neighbor.seq}, Makespan: {neighbor.makespan}, Fitness: {neighbor.fitness}")
        return neighbor

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

    def create_new_individual(self, individual, new_seq, config):
        new_individual = copy.deepcopy(individual)
        new_individual.seq = new_seq
        new_individual.job_seq = new_individual.get_repeatable()
        new_individual.feasible_seq = new_individual.get_feasible()
        new_individual.machine_order = new_individual.get_machine_order()
        new_individual.makespan, new_individual.mio_score = new_individual.evaluate(new_individual.machine_order)
        return new_individual
