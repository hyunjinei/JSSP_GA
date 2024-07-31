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

import random

class Individual:
    def __init__(self, config=None, seq=None, op_data=None):
        self.config = config
        self.op_data = op_data
        self.seq = seq if seq else random.sample(range(config.n_job), config.n_job)
        self.makespan = None
        self.fitness = None
        print(f"Current chromosome: {self.seq}")

    def __str__(self):
        return f"Individual(makespan={self.makespan}, fitness={self.fitness})"

    def evaluate(self):
        self.makespan = self.calculate_makespan()
        self.fitness = 1 / self.makespan
        return self.makespan

    def calculate_makespan(self):
        num_machines = self.config.n_machine
        completion_times = [[0 for _ in range(num_machines)] for _ in range(len(self.seq))]
        
        for i, job in enumerate(self.seq):
            for j in range(num_machines):
                machine, processing_time = self.op_data[job][j]
                if i == 0 and j == 0:
                    completion_times[i][j] = processing_time
                elif i == 0:
                    completion_times[i][j] = completion_times[i][j-1] + processing_time
                elif j == 0:
                    completion_times[i][j] = completion_times[i-1][j] + processing_time
                else:
                    completion_times[i][j] = max(completion_times[i-1][j], completion_times[i][j-1]) + processing_time
        
        return completion_times[-1][-1]

    def calculate_fitness(self, target_makespan):
        self.fitness = 1 / (self.makespan / target_makespan)
        return self.fitness

