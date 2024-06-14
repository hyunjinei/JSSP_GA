import os
import pandas as pd
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class RLDataset:
    def __init__(self, filename):
        self.name, _ = os.path.splitext(filename)
        self.path = 'Data/Dataset/'
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

        print(f"First few lines of data:\n{data.head()}\n")
        print(f"Number of jobs: {self.n_job}, Number of machines: {self.n_machine}")

        for i in range(self.n_job):
            self.op_data.append([])
            for j in range(self.n_machine):
                print(f"Processing job {i}, operation {j}")
                print(f"Machine: {data.iloc[self.n_job + i, j] - 1}, Duration: {data.iloc[i, j]}")
                
                self.op_data[i].append((data.iloc[self.n_job + i, j] - 1, data.iloc[i, j]))

        print(f"Operation data: {self.op_data}")

    def get_process_times(self):
        return [[op[1] for op in job] for job in self.op_data]

    def get_machine_sequence(self):
        return [[op[0] for op in job] for job in self.op_data]

def load_solutions(filename):
    with open(filename, 'r') as file:
        solutions = []
        for line in file:
            solutions.append([int(x) - 1 for x in line.strip().split()])
    return solutions

# 사용 예제
if __name__ == "__main__":
    dataset = RLDataset('la02.txt')
    process_times = dataset.get_process_times()
    machine_sequence = dataset.get_machine_sequence()
    solutions = load_solutions('la02_solutions.txt')
    print(f"Process times: {process_times}")
    print(f"Machine sequence: {machine_sequence}")
    print(f"Solutions: {solutions}")
