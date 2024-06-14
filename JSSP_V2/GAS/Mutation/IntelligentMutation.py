# Mutation/IntelligentMutation.py
import random
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from GAS.Mutation.base import Mutation
from GAS.Individual import Individual

class IntelligentMutation(Mutation):
    def __init__(self, pm):
        self.pm = pm

    def mutate(self, individual):
        if random.random() < self.pm:
            machine_loads = [0] * individual.config.n_machine
            for op in individual.seq:
                job_idx = op // individual.config.n_machine
                op_idx = op % individual.config.n_machine
                machine = individual.op_data[job_idx][op_idx][0]
                processing_time = individual.op_data[job_idx][op_idx][1]
                machine_loads[machine] += processing_time
            
            overloaded_machine = machine_loads.index(max(machine_loads))
            for i, op in enumerate(individual.seq):
                job_idx = op // individual.config.n_machine
                op_idx = op % individual.config.n_machine
                if individual.op_data[job_idx][op_idx][0] == overloaded_machine:
                    alternative_machines = [m for m in range(individual.config.n_machine) if m != overloaded_machine]
                    new_machine = random.choice(alternative_machines)
                    individual.op_data[job_idx][op_idx] = (new_machine, individual.op_data[job_idx][op_idx][1])
                    break
            return Individual(config=individual.config, seq=individual.seq, op_data=individual.op_data)
        return individual

        