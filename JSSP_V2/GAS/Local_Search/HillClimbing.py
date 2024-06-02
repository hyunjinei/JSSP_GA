import copy

class HillClimbing:
    def __init__(self, iterations=100):
        self.iterations = iterations

    def optimize(self, individual, config):
        print(f"Hill Climbing 시작")
        # print(f"Hill Climbing 시작 - Initial Individual: {individual.seq}, Makespan: {individual.makespan}, Fitness: {individual.fitness}")
        best_solution = copy.deepcopy(individual)
        best_makespan = individual.makespan
        iteration = 0

        while iteration < self.iterations:
            neighbors = self.get_neighbors(best_solution)
            # print(f"Iteration {iteration + 1} - Number of Neighbors: {len(neighbors)}")
            current_solution = min(neighbors, key=lambda ind: ind.makespan)
            current_makespan = current_solution.makespan

            if current_makespan >= best_makespan:
                # print(f"Iteration {iteration}: No improvement (current best makespan: {best_makespan})")
                break

            # print(f"Iteration {iteration}: Improved to makespan {current_makespan}")
            best_solution = current_solution
            best_makespan = current_makespan
            iteration += 1
        print(f"Hill Climbing 완료")
        # print(f"Hill Climbing 완료 - Optimized Individual: {best_solution.seq}, Makespan: {best_solution.makespan}, Fitness: {best_solution.fitness}")
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
