import os
import sys
import random
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

import simpy
# from Data.FT.ft10.ft10 import Dataset
from Data.Adams.abz7.abz7 import Dataset
# from Data.Dataset.Dataset import Dataset
from Config.Run_Config import Run_Config
import random
from GA_geneticpython.objects import Individual

print_console = False


class Machine:
    def __init__(self, n_machine):
        self.op_ready = []
        self.op_by_order = [[] for _ in range(n_machine)]
        self.current_position = 0
        self.finished = False


class Operation:
    def __init__(self, i, j, k, n_machine):
        self.job = i
        self.precedence = j
        self.machine = k
        self.idx = n_machine * i + j
        if j == 0:
            self.job_ready, self.machine_ready = True, True
        else:
            self.job_ready, self.machine_ready = False, False

        self.op_prior = None
        self.op_following = None


class JSSP:
    def __init__(self, dataset):
        self.dataset = dataset
        self.op_data = dataset.op_data
        self.op_list = [[] for _ in range(self.dataset.n_job)]
        self.machine_list = [Machine(dataset.n_machine) for _ in range(self.dataset.n_machine)]

        # Initialization
        for i in range(self.dataset.n_job):
            for j in range(self.dataset.n_machine):
                self.op_list[i].append(Operation(i, j, self.machine_list[self.op_data[i][j][0]], dataset.n_machine))

        for i in range(self.dataset.n_job):
            for j in range(self.dataset.n_machine):
                self.machine_list[self.op_data[i][j][0]].op_by_order[j].append(self.op_list[i][j])

        for i in range(self.dataset.n_machine):
            while len(self.machine_list[i].op_by_order[self.machine_list[i].current_position]) == 0:
                self.machine_list[i].current_position += 1
            self.machine_list[i].op_ready = self.machine_list[i].op_by_order[self.machine_list[i].current_position]
            for op in self.machine_list[i].op_ready:
                op.machine_ready = True

        # 연결 관계 수립
        for i in range(self.dataset.n_job):
            for j in range(1, self.dataset.n_machine):
                self.op_list[i][j].op_prior = self.op_list[i][j - 1]
                self.op_list[i][j - 1].op_following = self.op_list[i][j]

    def get_seq(self):
        self.ready = []
        self.seq = []
        for i in range(self.dataset.n_job):
            for j in range(self.dataset.n_machine):
                if self.op_list[i][j].job_ready and self.op_list[i][j].machine_ready:
                    self.ready.append(self.op_list[i][j])

        while len(self.seq) < self.dataset.n_op:
            if print_console: print('1. 현재 대기중인 작업 : ', [op.idx for op in self.ready])
            random.shuffle(self.ready)
            op = self.ready.pop()
            if print_console: print('2. 결정된 작업 : ', (op.job, op.precedence))
            self.seq.append(op)

            if op.precedence != self.dataset.n_machine - 1:
                if print_console: print('3. 현재까지 형성된 sequence : ', [op.idx for op in self.seq])
                if print_console: print('3-1. sequence 길이 :', len(self.seq))

                op.op_following.job_ready = True
                if print_console: print('4. 같은 job의 다음 operation의 작업 가능 현황 : ',
                                        (op.op_following.job_ready, op.op_following.machine_ready))

                if print_console: print('5-1. machine의 ready list 수정 전 : ', [op.idx for op in op.machine.op_ready])
                
                # Check if op is in op_ready before removing
                if op in op.machine.op_ready:
                    op.machine.op_ready.remove(op)
                    if print_console: print('5-2. machine의 ready list 수정 후 : ', [op.idx for op in op.machine.op_ready])
                else:
                    print(f"Error: Operation {op.idx} not found in machine {op.machine} op_ready list")
                    print(f"Current op_ready list: {[op.idx for op in op.machine.op_ready]}")
                    break  # Stop the process to prevent further errors

                while (not op.machine.finished) and len(op.machine.op_ready) == 0:
                    if print_console: print('5-3. 현재 machine의 ready list가 모두 비었으므로 다음 order로 이동')
                    op.machine.current_position += 1
                    if op.machine.current_position == self.dataset.n_machine:
                        if print_console: print('5-3-2. 해당 machine은 모든 operation이 배정되었습니다.')
                        op.machine.finished = True
                    else:
                        if print_console: print('5-4. 현재 machine의 작업 order : ', op.machine.current_position)
                        op.machine.op_ready = op.machine.op_by_order[op.machine.current_position]
                        if print_console: print('5-5. 바뀐 machine의 ready list :', op.machine.op_ready)
                if not len(op.machine.op_ready) == 0:
                    for x in op.machine.op_ready:
                        x.machine_ready = True

                # check
                if op.op_following.job_ready and op.op_following.machine_ready:
                    if op.op_following not in self.ready:
                        self.ready.append(op.op_following)
                        if print_console: print('6. Job 진행으로 인해 새롭게 ready list에 추가되는 작업 : ',
                                                (op.op_following.job, op.op_following.precedence))

                for x in op.machine.op_ready:
                    if x.job_ready and x.machine_ready:
                        if x not in self.ready:
                            self.ready.append(x)
                            if print_console: print('7. Machine 진행으로 인해 새롭게 ready list에 추가되는 작업 : ',
                                                    (x.idx))

        s = [op.idx for op in self.seq]
        self.__init__(self.dataset)
        return s



if __name__ == "__main__":
    dataset = Dataset()
    op_data = dataset.op_data
    config = Run_Config(dataset.n_job, dataset.n_machine, dataset.n_op,
                        False, False, False,
                        False, False, False)
    jssp = JSSP(dataset)
    for i in range(10):
        seq = jssp.get_seq()
        ind = Individual(config=config, seq=seq, op_data=op_data)
        print(ind.MIO)
        # print(ind.makespan)

    print()