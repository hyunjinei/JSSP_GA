import copy
import random

import copy
import random

class GifflerThompson:
    def __init__(self, priority_rule=None):
        self.priority_rule = priority_rule

    def optimize(self, individual, config):
        print(f"GifflerThompson 시작")
        if self.priority_rule is None:
            priority_rules = ['SPT', 'LPT', 'MWR', 'LWR', 'MOR', 'LOR', 'EDD']
            selected_rule = random.choice(priority_rules)
        else:
            selected_rule = self.priority_rule
        
        schedule = self.giffler_thompson(individual.seq, individual.op_data, config, selected_rule)
        optimized_individual = copy.deepcopy(individual)
        optimized_individual.seq = schedule
        optimized_individual.calculate_fitness(config.target_makespan)
        print(f"GifflerThompson 종료")
        return optimized_individual

    def giffler_thompson(self, seq, op_data, config, selected_rule):
        sorted_seq = self.apply_priority_rule(seq, op_data, config, selected_rule)
        return sorted_seq

    def apply_priority_rule(self, seq, op_data, config, rule):
        if rule == 'SPT':
            sorted_seq = sorted(seq, key=lambda x: op_data[x // config.n_machine][x % config.n_machine][1])
        elif rule == 'LPT':
            sorted_seq = sorted(seq, key=lambda x: -op_data[x // config.n_machine][x % config.n_machine][1])
        elif rule == 'MWR':
            sorted_seq = sorted(seq, key=lambda x: -sum(op_data[x // config.n_machine][i][1] for i in range(config.n_machine)))
        elif rule == 'LWR':
            sorted_seq = sorted(seq, key=lambda x: sum(op_data[x // config.n_machine][i][1] for i in range(config.n_machine)))
        elif rule == 'MOR':
            sorted_seq = sorted(seq, key=lambda x: -len([op for op in op_data[x // config.n_machine] if op[1] > 0]))
        elif rule == 'LOR':
            sorted_seq = sorted(seq, key=lambda x: len([op for op in op_data[x // config.n_machine] if op[1] > 0]))
        elif rule == 'EDD':
            sorted_seq = sorted(seq, key=lambda x: op_data[x // config.n_machine][x % config.n_machine][2])
        else:
            sorted_seq = seq  # Default to no sorting
        return sorted_seq
