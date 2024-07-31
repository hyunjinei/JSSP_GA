import sys
import os
import math
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import simpy
from environment.Source import Source
from environment.Sink import Sink
from environment.Part import Job, Operation
from environment.Process import Process
from environment.Resource import Machine
from environment.Monitor import Monitor
from postprocessing.PostProcessing import *
from visualization.Gantt import *
from visualization.GUI import GUI
from MachineInputOrder.utils import kendall_tau_distance, spearman_footrule_distance, spearman_rank_correlation, bubble_sort_distance, MSE

def calculate_score(x_array, y_array):
    score = [0.0 for i in range(6)]
    for i in range(len(x_array)):
        score[0] += kendall_tau_distance(x_array[i], y_array[i])
        score[1] += spearman_rank_correlation(x_array[i], y_array[i])
        score[2] += spearman_footrule_distance(x_array[i], y_array[i])
        score[3] += MSE(x_array[i], y_array[i])
        score[4] += bubble_sort_distance(x_array[i])
        correlation_matrix = np.corrcoef(x_array[i], y_array[i])
        score[5] += correlation_matrix[0, 1]
    return score

def swap_digits(num):
    if num < 10:
        return num * 10
    else:
        units = num % 10
        tens = num // 10
        return units * 10 + tens

class Individual:
    def __init__(self, config=None, seq=None, op_data=None):
        self.config = config
        self.op_data = op_data
        self.seq = seq if seq else random.sample(range(config.n_job), config.n_job)
        self.makespan = None
        self.fitness = None
        self.monitor = None
        print(f"Current chromosome: {self.seq}")

    def __str__(self):
        return f"Individual(makespan={self.makespan}, fitness={self.fitness})"

    def evaluate(self):
        env = simpy.Environment()
        self.monitor = Monitor(self.config)
        model = dict()

        # Initialize machines
        for j in range(self.config.n_machine):
            model['M' + str(j)] = Machine(env, j)

        # Initialize processes
        for j in range(self.config.n_machine):
            model['Process' + str(j)] = Process(env, 'Process' + str(j), model, self.monitor, self.seq, self.config)

        # Initialize sources
        for i in range(self.config.n_job):
            model['Source' + str(i)] = Source(env, 'Source' + str(i), model, self.monitor, job_order=self.seq, op_data=self.op_data, config=self.config)

        model['Sink'] = Sink(env, self.monitor, self.config)

        # Run the simulation to initialize sources
        env.run(self.config.simul_time)

        # Perform post-processing to ensure sequential operations
        self.post_process(model)

        if self.config.save_log:
            self.monitor.save_event_tracer(self.config.filename['log'])
            if self.config.save_machinelog:
                machine_log_ = generate_machine_log(self.config)
                if self.config.save_machinelog and self.config.show_gantt:
                    gantt = Gantt(machine_log_, len(machine_log_), self.config)
                    if self.config.show_gui:
                        gui = GUI(gantt)

        self.makespan = model['Sink'].last_arrival
        return self.makespan

    def post_process(self, model):
        machine_end_times = [0] * self.config.n_machine

        for job in self.seq:
            for step in range(self.config.n_machine):
                machine = model['M' + str(step)]
                process_time = self.op_data[job][step][1]

                start_time = max(machine_end_times[step], model['Process' + str(step)].get_last_end_time(job, step))
                end_time = start_time + process_time

                machine_end_times[step] = end_time

                self.monitor.record(start_time, 'Process' + str(step), machine='M' + str(step),
                                    part_name=f'Part{job}_{step}', event="Started")
                self.monitor.record(end_time, 'Process' + str(step), machine='M' + str(step),
                                    part_name=f'Part{job}_{step}', event="Finished")

    def calculate_makespan(self):
        num_machines = self.config.n_machine
        completion_times = [[0 for _ in range(num_machines)] for _ in range(len(self.seq))]

        for i, job in enumerate(self.seq):
            for j in range(num_machines):
                machine, processing_time = self.op_data[job][j]
                if i == 0 and j == 0:
                    completion_times[i][j] = processing_time
                elif i == 0:
                    completion_times[i][j] = completion_times[i][j - 1] + processing_time
                elif j == 0:
                    completion_times[i][j] = completion_times[i - 1][j] + processing_time
                else:
                    completion_times[i][j] = max(completion_times[i - 1][j], completion_times[i][j - 1]) + processing_time

        return completion_times[-1][-1]

    def calculate_fitness(self, target_makespan):
        if self.makespan is None:
            raise ValueError("Makespan is not calculated. Run evaluate() first.")
        self.fitness = 1 / (self.makespan / target_makespan)
        return self.fitness