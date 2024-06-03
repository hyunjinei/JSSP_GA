import copy

class HillClimbing:
    def __init__(self, iterations=100):
        self.iterations = iterations

    def optimize(self, individual, config):
        # print("HillClimbing 시작")

        best_solution = copy.deepcopy(individual)
        best_makespan = individual.makespan
        iteration = 0

        while iteration < self.iterations:
            neighbors = self.get_neighbors(best_solution, config)
            current_solution = min(neighbors, key=lambda ind: ind.makespan)
            current_makespan = current_solution.makespan

            if current_makespan >= best_makespan:
                break

            best_solution = current_solution
            best_makespan = current_makespan
            iteration += 1
            # print(f"Iteration {iteration} - Current Solution: {current_solution.seq}, Makespan: {current_makespan}, Fitness: {current_solution.fitness}")

        print(f"HillClimbing 완료 - Optimized Individual: {best_solution.seq}, Makespan: {best_solution.makespan}, Fitness: {best_solution.fitness}")

        return best_solution

    def get_neighbors(self, individual, config):
        neighbors = []
        seq = individual.seq
        for i in range(len(seq) - 1):
            for j in range(i + 1, len(seq)):
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
