import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# from Adams.abz5.abz5 import Dataset
# from FT.ft10.ft10 import Dataset
from Dataset.la import Dataset
# from Taillard.ta01.ta01 import Dataset
import matplotlib.pyplot as plt
import numpy as np
# filename = 'la01.txt'
dataset = Dataset()
# dataset = Dataset(filename)
op_data = np.array(dataset.op_data)
print(dataset.op_data)

def show_machine_distribution(dataset):
    op_data = np.array(dataset.op_data)

    machines = [[] for _ in range(dataset.n_machine)]
    machines_pt = [[] for _ in range(dataset.n_machine)]

    for i in range(len(op_data)):
        for j in range(len(op_data[i])):
            m = op_data[i][j][0]
            machines[m].append(np.where(op_data[i, :, 0] == m)[0][0])  # 자신의 위치
            machines_pt[m].append(op_data[i][j][1])

    plt.figure(figsize=(8, 6))
    for i in range(dataset.n_machine):
        plt.scatter(np.ones(len(machines[i])) * i, machines[i], color='black', alpha=0.2, s=machines_pt[i])

    plt.title(dataset.name + ' Data Distribution')
    plt.xlabel('Machine')
    plt.ylabel('Job Index')
    plt.savefig(os.path.join(dataset.path, dataset.name + '_Data_Distribution.png'))
    plt.show()


def show_pt_distribution(dataset):
    op_data = np.array(dataset.op_data)
    pt = op_data[:, :, 1].reshape(-1).tolist()
    plt.figure(figsize=(8, 6))
    plt.hist(pt, bins=30)
    plt.title(dataset.name + ' Processing Time Distribution')
    plt.xlabel('Processing Time')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(dataset.path, dataset.name + '_PT_Distribution.png'))
    plt.show()

if __name__ == "__main__":
    show_machine_distribution(dataset)
    show_pt_distribution(dataset)