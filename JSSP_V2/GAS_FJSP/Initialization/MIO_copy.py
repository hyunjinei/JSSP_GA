import os
import sys
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

class JSSP:
    def __init__(self, dataset):
        self.dataset = dataset

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
                if self.op_list[i][j].job_ready == True and self.op_list[i][j].machine_ready == True:
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
                op.machine.op_ready.remove(op)
                if print_console: print('5-2. machine의 ready list 수정 후 : ', [op.idx for op in op.machine.op_ready])

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
                if (op.op_following.job_ready == True) and (op.op_following.machine_ready == True):
                    if not op.op_following in self.ready:
                        self.ready.append(op.op_following)
                        if print_console: print('6. Job 진행으로 인해 새롭게 ready list에 추가되는 작업 : ',
                                                (op.op_following.job, op.op_following.precedence))

                for x in op.machine.op_ready:
                    if x.job_ready == True and x.machine_ready == True:
                        if not x in self.ready:
                            self.ready.append(x)
                            if print_console: print('7. Machine 진행으로 인해 새롭게 ready list에 추가되는 작업 : ',
                                                    (x.idx))

        s = [op.idx for op in self.seq]
        self.__init__(self.dataset)
        return s

def initialize_population_MIO(config, op_data):
    jssp = JSSP(op_data)
    individuals = []
    for _ in range(config.population_size):
        seq = jssp.get_seq()
        individual = Individual(config=config, seq=seq, op_data=op_data)
        individuals.append(individual)
    return individuals

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
        print(ind.makespan)

    print()