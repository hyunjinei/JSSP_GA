import numpy as np

from environment.Source import Source
from environment.Sink import Sink
from environment.Part import Job, Operation
from environment.Process import Process
from environment.Resource import Machine
from environment.Monitor import Monitor
from postprocessing.PostProcessing import *
from visualization.Gantt import *
from visualization.GUI import GUI
import simpy
from MachineInputOrder.utils import kendall_tau_distance, spearman_footrule_distance, spearman_rank_correlation, \
    bubble_sort_distance, MSE


def swap_digits(num):
    if num < 10:
        return num * 10
    else:
        units = num % 10
        tens = num // 10
        return units * 10 + tens


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
        # score[3] += bubble_sort_distance(x_array[i])
    return score


class Individual:
    def __init__(self, config=None, seq=None, solution_seq=None, op_data=None):
        if solution_seq != None:
            """
            5는 Job5의 0번째 operation
            15는 Job5의 1번째 operation
            이런 식으로 작성되어 있음
            사실상 [5 15 25 35 45 ... 95]까지가 한 Job을 나타냄
            """
            self.seq = self.interpret_solution(solution_seq)
        else:
            self.seq = seq  # 0부터 시작하는 값, op들의 순서

        self.config = config

        self.config.n_machine = self.config.n_machine
        self.config.n_job = self.config.n_job
        self.op_data = op_data
        self.MIO = []
        self.MIO_sorted = []
        self.job_seq = self.get_repeatable()
        self.feasible_seq = self.get_feasible()
        self.machine_seq = self.get_machine_order()
        self.makespan, self.mio_score = self.evaluate(self.machine_seq)
        self.score = calculate_score(self.MIO, self.MIO_sorted)

    def interpret_solution(self, s):
        # 리스트의 각 원소에 대해 숫자 바꾸기
        modified_list = [swap_digits(num) for num in s]
        return modified_list

    def get_repeatable(self):
        cumul = 0
        sequence_ = np.array(self.seq)
        for i in range(self.config.n_job):
            for j in range(self.config.n_machine):
                sequence_ = np.where((sequence_ >= cumul) &
                                     (sequence_ < cumul + self.config.n_machine), i, sequence_)
            cumul += self.config.n_machine
        sequence_ = sequence_.tolist()
        return sequence_

    def get_feasible(self):
        temp = 0
        cumul = 0
        sequence_ = np.array(self.seq)
        for i in range(self.config.n_job):
            idx = np.where((sequence_ >= cumul) & (sequence_ < cumul + self.config.n_machine))[0]
            for j in range(self.config.n_machine):
                sequence_[idx[j]] = temp
                temp += 1
            cumul += self.config.n_machine
        return sequence_

    def get_machine_order(self):
        m_list = []
        for num in self.feasible_seq:
            idx_j = num % self.config.n_machine  # job의 번호
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
        monitor = Monitor(self.config)
        model = dict()
        for i in range(self.config.n_job):
            model['Source' + str(i)] = Source(env, 'Source' + str(i), model, monitor,
                                              part_type=i, op_data=self.op_data, config=self.config)

        for j in range(self.config.n_machine):
            model['Process' + str(j)] = Process(env, 'Process' + str(j), model, monitor, machine_order[j],
                                                self.config)
            model['M' + str(j)] = Machine(env, j)

        model['Sink'] = Sink(env, monitor, self.config)

        # In case of the situation where termination of the simulation greatly affects the machine utilization time,
        # it is necessary to terminate all the process at (SIMUL_TIME -1) and add up the process time to all machines
        env.run(self.config.simul_time)

        if self.config.save_log :
            monitor.save_event_tracer()
            if self.config.save_machinelog:
                machine_log_ = machine_log(self.config)
                if self.config.save_machinelog & self.config.show_gantt:
                    gantt = Gantt(machine_log_, len(machine_log_), self.config)
                    if self.config.show_gui:
                        gui = GUI(gantt)

        for i in range(self.config.n_machine):
            mio = model['M' + str(i)].op_where
            self.MIO.append(mio)
            self.MIO_sorted.append(np.sort(mio))

        mio_score = np.sum(np.abs(np.subtract(np.array(mio), np.array(sorted(mio)))))
        return model['Sink'].last_arrival, mio_score


def rearrange_sequence(sequence):
    index = sorted(range(len(sequence)), key=lambda k: sequence[k], reverse=False)
    result_sequence = [0 for i in range(len(sequence))]
    for i, idx in enumerate(index):
        result_sequence[idx] = i

    return result_sequence

# # 테스트를 위한 임의의 수열 생성
# original_sequence = [0, 2, 8, 6, 5, 1, 9, 2, 3, 4]
#
# # 함수 호출
# result_sequence = rearrange_sequence(original_sequence)
#
# # 결과 출력
# print(result_sequence)
