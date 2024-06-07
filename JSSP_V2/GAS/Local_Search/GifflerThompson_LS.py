import copy
import random

class GifflerThompson_LS:
    def __init__(self, priority_rule=None):
        self.priority_rules = ['SPT', 'LPT', 'MWR', 'LWR', 'MOR', 'LOR', 'EDD']
        self.default_priority_rule = priority_rule
        self.stop_search = False  # 종료 조건 플래그 추가

    def optimize(self, individual, config):
        best_individual = copy.deepcopy(individual)
        best_individual.calculate_fitness(config.target_makespan)
        best_rule = "basic"

        # 기본 우선순위 규칙 적용 결과
        default_schedule = self.giffler_thompson(individual.seq, individual.op_data, config, self.default_priority_rule)
        default_individual = self.create_new_individual(individual, default_schedule, config)
        default_individual.calculate_fitness(config.target_makespan)
        # print(f"Default rule (basic) fitness: {default_individual.fitness}")

        best_fitness = default_individual.fitness
        best_individuals = [(default_individual, "basic")]

        # 모든 우선순위 규칙 적용 결과 비교
        for rule in self.priority_rules:
            schedule = self.giffler_thompson(individual.seq, individual.op_data, config, rule)
            optimized_individual = self.create_new_individual(individual, schedule, config)
            optimized_individual.calculate_fitness(config.target_makespan)
            # print(f"Rule {rule} fitness: {optimized_individual.fitness}")

            if optimized_individual.fitness > best_fitness:
                best_fitness = optimized_individual.fitness
                best_individuals = [(optimized_individual, rule)]
                best_rule = rule
            elif optimized_individual.fitness == best_fitness:
                best_individuals.append((optimized_individual, rule))

        selected_individual, selected_rule = random.choice(best_individuals)
        # print(f"Selected rule: {selected_rule}, Selected individual: {selected_individual}")
        
        # 목표 Makespan에 도달하면 Local Search 종료
        if selected_individual.fitness >= 1.0:
            print(f"Stopping early as fitness {selected_individual.fitness} is 1.0 or higher.")
            self.stop_search = True

        return selected_individual

    def giffler_thompson(self, seq, op_data, config, priority_rule):
        return self.apply_priority_rule(seq, op_data, config, priority_rule)

    def apply_priority_rule(self, seq, op_data, config, priority_rule):
        def safe_get_op_data(x, idx):
            try:
                return op_data[x // config.n_machine][x % config.n_machine][idx]
            except IndexError:
                return float('inf') if idx == 1 else 0

        if priority_rule is None:
            return seq  # 우선순위 규칙을 적용하지 않은 경우

        if priority_rule == 'SPT':
            sorted_seq = sorted(seq, key=lambda x: safe_get_op_data(x, 1))
        elif priority_rule == 'LPT':
            sorted_seq = sorted(seq, key=lambda x: -safe_get_op_data(x, 1))
        elif priority_rule == 'MWR':
            sorted_seq = sorted(seq, key=lambda x: -sum(safe_get_op_data(x, 1) for i in range(config.n_machine)))
        elif priority_rule == 'LWR':
            sorted_seq = sorted(seq, key=lambda x: sum(safe_get_op_data(x, 1) for i in range(config.n_machine)))
        elif priority_rule == 'MOR':
            sorted_seq = sorted(seq, key=lambda x: -len([op for op in op_data[x // config.n_machine] if op[1] > 0]))
        elif priority_rule == 'LOR':
            sorted_seq = sorted(seq, key=lambda x: len([op for op in op_data[x // config.n_machine] if op[1] > 0]))
        elif priority_rule == 'EDD':
            sorted_seq = sorted(seq, key=lambda x: safe_get_op_data(x, 2))
        return sorted_seq

    def create_new_individual(self, individual, new_seq, config):
        new_individual = copy.deepcopy(individual)
        new_individual.seq = new_seq
        new_individual.job_seq = new_individual.get_repeatable()
        new_individual.feasible_seq = new_individual.get_feasible()
        new_individual.machine_order = new_individual.get_machine_order()
        new_individual.makespan, new_individual.mio_score = new_individual.evaluate(new_individual.machine_order)
        return new_individual
