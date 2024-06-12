import os
import sys
import random
import numpy as np
import simpy

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from GA_geneticpython.objects import Individual

# from environment.Part import Operation  # Import the Operation class
from environment.Resource import Machine as BaseMachine  # Import the original Machine class

class Operation:
    def __init__(self, job, precedence, machine, n_machine):
        self.job = job
        self.precedence = precedence
        self.machine = machine
        self.idx = n_machine * job + precedence
        self.job_ready = precedence == 0
        self.machine_ready = precedence == 0
        self.op_prior = None
        self.op_following = None

class MIOMachine(BaseMachine):
    def __init__(self, id, n_machine):
        super().__init__(None, id)  # Call the base class constructor with id
        self.op_ready = []
        self.op_by_order = [[] for _ in range(n_machine)]
        self.current_position = 0
        self.finished = False

class JSSP:
    def __init__(self, dataset):
        self.dataset = dataset
        self.op_data = dataset.op_data
        self.op_list = [[] for _ in range(self.dataset.n_job)]
        self.machine_list = [MIOMachine(idx, dataset.n_machine) for idx in range(self.dataset.n_machine)]

        for i in range(self.dataset.n_job):
            for j in range(self.dataset.n_machine):
                op_data = self.op_data[i][j]
                self.op_list[i].append(Operation(i, j, op_data[0], dataset.n_machine))

        for i in range(self.dataset.n_job):
            for j in range(self.dataset.n_machine):
                self.machine_list[self.op_data[i][j][0]].op_by_order[j].append(self.op_list[i][j])

        for i in range(self.dataset.n_machine):
            while len(self.machine_list[i].op_by_order[self.machine_list[i].current_position]) == 0:
                self.machine_list[i].current_position += 1
            self.machine_list[i].op_ready = self.machine_list[i].op_by_order[self.machine_list[i].current_position]
            for op in self.machine_list[i].op_ready:
                op.machine_ready = True

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
            if len(self.ready) == 0:
                raise ValueError("Ready list is empty. Cannot pop from empty list.")
            random.shuffle(self.ready)
            op = self.ready.pop()
            self.seq.append(op)

            if op.precedence != self.dataset.n_machine - 1:
                op.op_following.job_ready = True
                op.machine.op_ready.remove(op)

                while not op.machine.finished and len(op.machine.op_ready) == 0:
                    op.machine.current_position += 1
                    if op.machine.current_position == self.dataset.n_machine:
                        op.machine.finished = True
                    else:
                        op.machine.op_ready = op.machine.op_by_order[op.machine.current_position]

                if len(op.machine.op_ready) != 0:
                    for x in op.machine.op_ready:
                        x.machine_ready = True

                if op.op_following.job_ready and op.op_following.machine_ready:
                    if op.op_following not in self.ready:
                        self.ready.append(op.op_following)

                for x in op.machine.op_ready:
                    if x.job_ready and x.machine_ready:
                        if x not in self.ready:
                            self.ready.append(x)

        s = [op.idx for op in self.seq]
        return s



