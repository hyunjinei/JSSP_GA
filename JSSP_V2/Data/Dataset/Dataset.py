import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from GAS.Individual import Individual  
# class Solution():


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
            print("First line from file:", first_line)  # 디버깅 출력 추가

        self.n_job, self.n_machine = map(int, first_line.strip().split('\t'))
        self.n_op = self.n_job * self.n_machine

        self.op_data = []
        data = pd.read_csv(file_path, sep="\t", engine='python', encoding="cp949", skiprows=[0], header=None)
        print("Loaded data:\n", data)  # 디버깅 출력 추가

        for i in range(self.n_job):
            self.op_data.append([])
            for j in range(self.n_machine):
                process_type = int(data.iloc[self.n_job + i, j]) - 1
                process_time = float(data.iloc[i, j])
                self.op_data[i].append((process_type, process_time))
                print(f"Job {i}, Machine {j}: Process {process_type}, Time {process_time}")  # 디버깅 출력 추가

        self.n_solution = 0
