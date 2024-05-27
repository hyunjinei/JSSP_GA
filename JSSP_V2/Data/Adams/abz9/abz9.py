import os
import pandas as pd
import sys
# class Solution():

class Dataset:
    def __init__(self):
        self.name = 'abz9'
        self.path = 'Data\\Adams\\abz9\\'
        if __name__ == "__main__":
            file_path = os.path.join(os.getcwd(),'abz9.txt')
        else:
            file_path = os.path.join(os.path.dirname(__file__),'abz9.txt')

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
                self.op_data[i].append((data.iloc[self.n_job + i, j] - 1, data.iloc[i, j]))

        # Set Solution Data - No solution currently available!
    
