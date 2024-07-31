import os
import pandas as pd
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random

class RLDataset_FJSSP_generate:
    def __init__(self, num_jobs, num_machines, flexibility, max_operations_per_job, max_machine_options):
        self.n_job = num_jobs
        self.n_machine = num_machines
        self.flexibility = flexibility
        self.op_data = []
        self.total_operations = 0
        self.max_op_counts = []

        self.generate_random_dataset(max_operations_per_job, max_machine_options)

    def generate_random_dataset(self, max_operations_per_job, max_machine_options):
        for job_index in range(self.n_job):
            num_operations = max(1, int(self.flexibility * random.randint(1, max_operations_per_job)))
            self.max_op_counts.append(num_operations - 1)
            self.total_operations += num_operations
            self.op_data.append([])

            for _ in range(num_operations):
                machine_options = []
                num_machine_options = max(1, int(self.flexibility * random.randint(1, max_machine_options)))
                for _ in range(num_machine_options):
                    machine = random.randint(0, self.n_machine - 1)
                    time = random.randint(1, 10)
                    machine_options.append((machine, time))
                self.op_data[job_index].append(machine_options)

        print(f"Generated random dataset with {self.n_job} jobs, {self.n_machine} machines, and flexibility {self.flexibility}.")
        print(f"Total operations: {self.total_operations}")
        print(f"Max operation counts: {self.max_op_counts}")
        print(f"op data: {self.op_data}")

    def save_to_file(self, filename):
        with open(filename, 'w') as file:
            file.write(f"{self.n_job} {self.n_machine} {self.flexibility}\n")
            for job_operations in self.op_data:
                job_line = f"{len(job_operations)}"
                for operation in job_operations:
                    job_line += f" {len(operation)}"
                    for machine, time in operation:
                        job_line += f" {machine + 1} {time}"
                file.write(f"{job_line}\n")
