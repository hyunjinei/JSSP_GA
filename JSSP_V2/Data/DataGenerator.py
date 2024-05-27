import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from itertools import permutations
import numpy as np
from statistics import *
from Dataset.Dataset import Dataset


def generate_JSSP_data(num_job, num_machine, prefix):
    filename = prefix + str(num_job) + str(num_machine) + '.txt'
    first_line = f"{num_job}\t{num_machine}"
    df = pd.DataFrame(np.random.randint(11, 41, size=(num_job, num_machine)))
    # 각 행의 숫자를 1부터 num_machine까지의 permutation으로 변경
    for i in range(num_job):
        permutation = np.random.permutation(np.arange(1, num_machine + 1)).astype(int)
        df.loc[df.shape[0]] = permutation

    # 파일 작성
    with open(filename, 'w') as f:
        # 첫번째 줄 작성
        f.write(first_line + '\n')
        # 데이터프레임을 파일에 작성
        df.to_csv(f, sep='\t', index=False, header=False, lineterminator='\n')  # Updated lineterminator

num_job = 3
num_machine = 3
generate_JSSP_data(num_job, num_machine, './Dataset/test_')
filename = 'test_'+str(num_job) +str(num_machine)+'.txt'
dataset = Dataset(filename)

# Assuming show_machine_distribution and show_pt_distribution are defined elsewhere
# show_machine_distribution(dataset)
# show_pt_distribution(dataset)
