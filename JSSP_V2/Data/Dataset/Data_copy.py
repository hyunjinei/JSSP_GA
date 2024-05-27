import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
from GAS.Individual import Individual  # GA_pyGAD 대신 GAS 모듈 사용

class Dataset:
    def __init__(self, filename):
        self.name, _ = os.path.splitext(filename)
        self.path = 'Data\\Dataset\\'        
        if __name__ == "__main__":
            file_path = os.path.join(os.getcwd(), filename)
        else:
            file_path = os.path.join(os.path.dirname(__file__), filename)

        with open(file_path, 'r') as file:
            first_line = file.readline()

        self.n_job, self.n_machine = map(int, first_line.strip().split('\t'))
        self.n_op = self.n_job * self.n_machine

        self.op_data = []
        data = pd.read_csv(file_path, sep="\t", engine='python', encoding="cp949", skiprows=[0], header=None)
        for i in range(self.n_job):
            self.op_data.append([])
            for j in range(self.n_machine):
                self.op_data[i].append((data.iloc[self.n_job + i, j] - 1, data.iloc[i, j]))

        self.n_solution = 0

    def print_op_data(self):
        for i, job in enumerate(self.op_data):
            print(f"Job {i}:")
            for j, (machine, time) in enumerate(job):
                print(f"  Operation {j}: Machine {machine}, Time {time}")

# 예시 호출
if __name__ == "__main__":
    dataset = Dataset('test_33.txt')
    dataset.print_op_data()
