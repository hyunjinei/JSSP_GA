# Crossover/JBX.py
import sys
import os
import random
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from GAS.Crossover.base import Crossover
from GAS.Individual import Individual

# Job-based order crossover
class JBX(Crossover):
    def __init__(self, pc):
        self.pc = pc

    def cross(self, parent1, parent2):
        if random.random() > self.pc:
            return parent1, parent2

        size = len(parent1.seq)
        jobs = list(set(parent1.seq))

        def get_jobs_by_machine(parent, machine_count):
            jobs_by_machine = [[] for _ in range(machine_count)]
            for i, job in enumerate(parent.seq):
                machine = i % machine_count
                jobs_by_machine[machine].append(job)
            return jobs_by_machine

        jobs_by_machine_p1 = get_jobs_by_machine(parent1, len(jobs))
        jobs_by_machine_p2 = get_jobs_by_machine(parent2, len(jobs))

        def fill_unfixed_positions(proto_child, reference):
            fixed_jobs = set(proto_child)
            unfixed_jobs = [job for job in reference if job not in fixed_jobs]
            return [job if job is not None else unfixed_jobs.pop(0) for job in proto_child]

        offspring1_seq = []
        offspring2_seq = []

        for machine in range(len(jobs)):
            proto_child1 = jobs_by_machine_p1[machine][:2] + [None]
            proto_child2 = jobs_by_machine_p2[machine][:2] + [None]

            proto_child1 = fill_unfixed_positions(proto_child1, jobs_by_machine_p2[machine])
            proto_child2 = fill_unfixed_positions(proto_child2, jobs_by_machine_p1[machine])

            offspring1_seq.extend(proto_child1)
            offspring2_seq.extend(proto_child2)

        return Individual(config=parent1.config, seq=offspring1_seq, op_data=parent1.op_data), Individual(config=parent1.config, seq=offspring2_seq, op_data=parent1.op_data)
