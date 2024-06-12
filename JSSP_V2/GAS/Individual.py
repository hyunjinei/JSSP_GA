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
    def __init__(self, config=None, seq=None, solution_seq=None, op_data=None):
        self.fitness = None
        # self.scaled_fitness = None # 새로 넣은것. fitness때매
        self.monitor = None  # Add monitor attribute
        if solution_seq is not None:
            self.seq = self.interpret_solution(solution_seq)
        else:
            self.seq = seq

        self.config = config
        self.op_data = op_data
        self.MIO = []
        self.MIO_sorted = []
        self.job_seq = self.get_repeatable()
        self.feasible_seq = self.get_feasible()
        self.machine_order = self.get_machine_order()
        self.makespan, self.mio_score = self.evaluate(self.machine_order)
        self.score = calculate_score(self.MIO, self.MIO_sorted)
        self.calculate_fitness(config.target_makespan)  # Ensure target_makespan is passed
        '''
        오리지날 
        self.calculate_fitness()
        '''

    def __str__(self):
        return f"Individual(makespan={self.makespan}, fitness={self.fitness})"
        # return f"Individual(seq={self.seq}, makespan={self.makespan}, fitness={self.fitness})"

    def calculate_fitness(self, target_makespan):
        if self.makespan == 0:
            raise ValueError("Makespan is zero, which will cause division by zero error.")
        self.fitness = 1 / (self.makespan / target_makespan)
        # print(f"Calculated fitness: {self.fitness} for makespan: {self.makespan} and target_makespan: {target_makespan}")
        return self.fitness
        
        
        '''
        fitness 수정 본 (적용은 잘되었지만 1300이 target_makespan인데 makespan이 1380인데도 1.0으로 도출)
    def calculate_fitness(self, target_makespan):
        if self.makespan == 0:
            raise ValueError("Makespan is zero, which will cause division by zero error.")
        self.fitness = 1 / (self.makespan / target_makespan)
        return self.fitness
        '''

        '''
        오리지날 calculate_fitness

        if self.makespan == 0:
            raise ValueError("Makespan is zero, which will cause division by zero error.")
        self.fitness = 1 / self.makespan
        return self.fitness
        '''

    def interpret_solution(self, s):
        modified_list = [swap_digits(num) for num in s]
        return modified_list

    def get_repeatable(self):
        cumul = 0
        sequence_ = np.array(self.seq)
        for i in range(self.config.n_job):
            for j in range(self.config.n_machine):
                sequence_ = np.where((sequence_ >= cumul) & (sequence_ < cumul + self.config.n_machine), i, sequence_)
            cumul += self.config.n_machine
        return sequence_.tolist()

    def get_feasible(self):
        temp = 0
        cumul = 0
        sequence_ = np.array(self.seq)
        for i in range(self.config.n_job):
            idx = np.where((sequence_ >= cumul) & (sequence_ < cumul + self.config.n_machine))[0]
            for j in range(min(len(idx), self.config.n_machine)):
                sequence_[idx[j]] = temp
                temp += 1
            cumul += self.config.n_machine
        return sequence_

    def get_machine_order(self):
        m_list = []
        for num in self.feasible_seq:
            idx_j = num % self.config.n_machine
            idx_i = num // self.config.n_machine
            m_list.append(self.op_data[idx_i][idx_j][0])
        m_list = np.array(m_list)

        m_order = []
        for num in range(self.config.n_machine):
            idx = np.where((m_list == num))[0]
            job_order = [self.job_seq[o] for o in idx]
            m_order.append(job_order)
        return m_order

    def evaluate(self, machine_order):
        env = simpy.Environment()
        self.monitor = Monitor(self.config)
        model = dict()
        for i in range(self.config.n_job):
            model['Source' + str(i)] = Source(env, 'Source' + str(i), model, self.monitor, part_type=i, op_data=self.op_data, config=self.config)

        for j in range(self.config.n_machine):
            model['Process' + str(j)] = Process(env, 'Process' + str(j), model, self.monitor, machine_order[j], self.config)
            model['M' + str(j)] = Machine(env, j)

        model['Sink'] = Sink(env, self.monitor, self.config)
        
        env.run(self.config.simul_time)
        
        if self.config.save_log: 
            self.monitor.save_event_tracer()
            if self.config.save_machinelog:
                machine_log_ = generate_machine_log(self.config)
                if self.config.save_machinelog and self.config.show_gantt:
                    gantt = Gantt(machine_log_, len(machine_log_), self.config)
                    if self.config.show_gui:
                        gui = GUI(gantt)

        for i in range(self.config.n_machine):
            mio = model['M' + str(i)].op_where
            self.MIO.append(mio)
            self.MIO_sorted.append(np.sort(mio))

        # mio_score = np.sum(np.abs(np.subtract(np.array(mio), np.array(sorted(mio)))))
        # return model['Sink'].last_arrival, mio_score
        mio_score = np.sum(np.abs(np.subtract(np.array(mio), np.array(sorted(mio)))))
        makespan = model['Sink'].last_arrival
        # print(f"Calculated makespan: {makespan} and mio_score: {mio_score}")
        return makespan, mio_score        
            