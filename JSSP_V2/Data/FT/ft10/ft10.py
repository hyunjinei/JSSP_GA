import os
import pandas as pd
import sys
from GA_pyGAD.GA import Individual
# class Solution():

class Dataset:
    def __init__(self):
        self.name = 'ft10'
        self.path = 'Data\\FT\\ft10\\'
        if __name__ == "__main__":
            file_path = os.path.join(os.getcwd(),'ft10.txt')
        else:
            file_path = os.path.join(os.path.dirname(__file__),'ft10.txt')

        # 파일 열기 및 첫 번째 줄 읽기
        with open(file_path, 'r') as file:
            first_line = file.readline()

        # Tab으로 구분된 값을 분리하여 변수에 저장
        self.n_job, self.n_machine = map(int, first_line.strip().split('\t'))
        self.n_op = self.n_job * self.n_machine

        # Set Problem Data
        self.op_data = []
        data = pd.read_csv(file_path, sep="\t", engine='python', encoding="cp949", skiprows=[0], header=None)
        for i in range(self.n_job):
            self.op_data.append([])
            for j in range(self.n_machine):
                # machine이 1부터 시작하기 때문에 M0부터 시작하게 하기 위해서 1씩 빼줌
                self.op_data[i].append((data.iloc[10 + i, j] - 1, data.iloc[i, j]))

        # Set Solution Data
        self.solution_list = []
        solution_path = os.path.join(os.path.dirname(__file__), 'ft10_solutions.txt')
        with open(solution_path, 'r') as file:
            for line in file:
                # 각 줄을 tab으로 구분하여 숫자들을 리스트로 변환
                line_values = list(map(int, line.strip().split('\t')))
                sol = [idx - 1 for idx in line_values]

                # 리스트를 solution에 추가
                # self.solution_list.append(line_values)
                self.solution_list.append(sol)

        self.n_solution = len(self.solution_list)



if __name__ == "__main__":
    from Config.Run_Config import Run_Config

    config = Run_Config(10, 10, 100, False, False, False, False, False, False)

    dataset = Dataset()
    num_solution = dataset.n_solution

    makespan = [0 for i in range(num_solution)]
    score = [[0 for i in range(num_solution)] for j in range(4)]
    for idx, s in enumerate(dataset.solution_list):
        ind = Individual(config, solution_seq=s, op_data=dataset.op_data)
        makespan[idx] = ind.makespan
        score[0][idx] = ind.score[0]
        score[1][idx] = ind.score[1]
        score[2][idx] = ind.score[2]
        score[3][idx] = ind.score[3]
    print(score)