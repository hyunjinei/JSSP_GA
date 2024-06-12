import sys
import os
import math
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import simpy
from environment_multi.Source import Source
from environment_multi.Sink import Sink
from environment_multi.Part import Job, Operation
from environment_multi.Process import Process
from environment_multi.Resource import Machine
from environment_multi.Monitor import Monitor
from postprocessing.PostProcessing import *
from visualization.Gantt import *
from visualization.GUI import GUI
from MachineInputOrder.utils import kendall_tau_distance, spearman_footrule_distance, spearman_rank_correlation, bubble_sort_distance, MSE

class MachineTask:
    def __init__(self, machine_id):
        self.machine_id = machine_id
        self.tasks = []

    def add_task(self, job_id, op_id, process_time):
        self.tasks.append({
            'job_id': job_id,
            'op_id': op_id,
            'process_time': process_time,
            'start_time': None,
            'end_time': None
        })

    def start_task(self, job_id, op_id, start_time):
        for task in self.tasks:
            if task['job_id'] == job_id and task['op_id'] == op_id:
                task['start_time'] = start_time
                task['end_time'] = start_time + task['process_time']
                break

    def calculate_idle_time_and_makespan(self):
        # None을 제외하고 정렬
        tasks_with_start_time = [task for task in self.tasks if task['start_time'] is not None]
        tasks_with_start_time.sort(key=lambda x: x['start_time'])
        
        idle_time = 0
        prev_end_time = 0
        for task in tasks_with_start_time:
            if task['start_time'] > prev_end_time:
                idle_time += task['start_time'] - prev_end_time
            prev_end_time = task['end_time']
        makespan = prev_end_time
        return idle_time, makespan




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
        self.monitor = None  # Add monitor attribute
        if solution_seq is not None:
            self.seq = self.interpret_solution(solution_seq)
        else:
            self.seq = seq

        self.config = config
        self.op_data = op_data
        self.MIO = []
        self.MIO_sorted = []
        self.makespan = 0  # 추가
        self.idle_time = 0  # 추가
        self.fitness = 0  # 추가         
        self.job_seq = self.get_repeatable()
        self.feasible_seq = self.get_feasible()
        self.machine_order = self.get_machine_order()
        self.makespan, self.idle_time, self.mio_score = self.evaluate(self.machine_order)  # idle_time 추가
        self.score = calculate_score(self.MIO, self.MIO_sorted)
        self.calculate_fitness(config.target_makespan, self.idle_time, multi_objective=True)  # idle_time 추가

    def __str__(self):
        return f"Individual(seq={self.seq}, makespan={self.makespan}, fitness={self.fitness})"
        # return f"Individual(makespan={self.makespan}, fitness={self.fitness})"

    def calculate_fitness(self, target_makespan, idle_time=None, multi_objective=False):
        if self.makespan == 0:
            raise ValueError("Makespan is zero, which will cause division by zero error.")
        
        if not multi_objective:
            self.fitness = 1 / (self.makespan / target_makespan)
            print(f'fitness: {self.fitness}, makespan: {self.makespan}, target_makespan: {target_makespan}')
        else:
            if idle_time is None:
                raise ValueError("For multi-objective fitness calculation, idle_time must be provided.")
            
            combined_time = self.makespan + idle_time
            self.fitness = 1 / (combined_time / target_makespan)
            print(f'fitness: {self.fitness}, combined_time: {combined_time}, makespan: {self.makespan}, idle_time: {idle_time}, target_makespan: {target_makespan}')
        
        return self.fitness

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

        machine_tasks = {i: MachineTask(i) for i in range(self.config.n_machine)}

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

        for job_id in range(self.config.n_job):
            for op_id in range(self.config.n_machine):
                machine_id = self.op_data[job_id][op_id][0]
                process_time = self.op_data[job_id][op_id][1]
                machine_tasks[machine_id].add_task(job_id, op_id, process_time)

        for event in self.monitor.event_log:
            if event['event'] == 'Started':
                try:
                    job_part = event['part_name'].split('_')
                    job_id = int(job_part[0].replace('Part', ''))
                    op_id = int(job_part[1].replace('Op', ''))
                    machine_id = int(event['machine'].replace('M', ''))
                    start_time = event['time']
                    machine_tasks[machine_id].start_task(job_id, op_id, start_time)
                except (IndexError, ValueError) as e:
                    print(f"Error parsing part_name: {event['part_name']}, error: {e}")

        max_makespan = 0
        total_idle_time = 0
        for machine in machine_tasks.values():
            idle_time, makespan = machine.calculate_idle_time_and_makespan()
            print(f'Machine {machine.machine_id}: idle_time = {idle_time}, makespan = {makespan}')
            total_idle_time += idle_time
            if makespan > max_makespan:
                max_makespan = makespan

        self.idle_time = total_idle_time
        mio_score = np.sum(np.abs(np.subtract(np.array(mio), np.array(sorted(mio)))))

        if max_makespan == 0:
            print("Debug: max_makespan is zero")
        
        return max_makespan, total_idle_time, mio_score  # total_idle_time 추가





