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
        total_operations_required = self.n_job * self.n_machine
        operations_per_job = total_operations_required // self.n_job
        remaining_operations = total_operations_required % self.n_job

        for job_index in range(self.n_job):
            num_operations = operations_per_job + (1 if job_index < remaining_operations else 0)
            self.max_op_counts.append(num_operations - 1)
            self.total_operations += num_operations
            self.op_data.append([])

            for _ in range(num_operations):
                machine_options = []
                num_machine_options = max(1, int(self.flexibility * max_machine_options))
                num_machine_options = min(num_machine_options, self.n_machine)  # Ensure we do not exceed the number of machines

                selected_machines = random.sample(range(self.n_machine), num_machine_options)
                for machine in selected_machines:
                    time = random.randint(1, 100)  # You can adjust the range as needed
                    machine_options.append((machine, time))
                self.op_data[job_index].append(machine_options)

        print(f"Generated random dataset with {self.n_job} jobs, {self.n_machine} machines, and flexibility {self.flexibility}.")
        print(f"Total operations: {self.total_operations}")
        print(f"Max operation counts: {self.max_op_counts}")
        print(f"op data: {self.op_data}")

    def save_to_file(self, filename):
        with open(filename, 'w') as file:
            file.write(f"{self.n_job}\t{self.n_machine}\t{self.flexibility}\n")
            for job_operations in self.op_data:
                job_line = f"{len(job_operations)}"
                for operation in job_operations:
                    job_line += f"\t{len(operation)}"
                    for machine, time in operation:
                        job_line += f"\t{machine + 1}\t{time}"
                file.write(f"{job_line}\n")

def generate_datasets(num_datasets, num_jobs, num_machines, flexibility, max_operations_per_job, max_machine_options, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for i in range(num_datasets):
        dataset = RLDataset_FJSSP_generate(num_jobs, num_machines, flexibility, max_operations_per_job, max_machine_options)
        filename = os.path.join(output_folder, f"dataset_{i + 1}.fjs")
        dataset.save_to_file(filename)
        print(f"Saved dataset to {filename}")

def main():
    # 무작위로 생성된 데이터셋
    num_datasets = 20
    num_jobs = 10
    num_machines = 5
    flexibility = 1.15
    max_operations_per_job = 10
    max_machine_options = 3

    output_folder = "fjssprandom_10_5"
    generate_datasets(num_datasets, num_jobs, num_machines, flexibility, max_operations_per_job, max_machine_options, output_folder)


if __name__ == "__main__":
    main()
