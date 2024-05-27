from Adams.abz5.abz5 import Dataset
# from FT.ft10.ft10 import Dataset
# from Dataset.Dataset import Dataset
# from Taillard.ta01.ta01 import Dataset
import matplotlib.pyplot as plt
import numpy as np
# filename = 'test_2020.txt'
dataset = Dataset()
# dataset = Dataset(filename)
op_data = np.array(dataset.op_data)

def show_machine_distribution(dataset):
    op_data = np.array(dataset.op_data)

    machines = [[] for _ in range(dataset.n_machine)]
    machines_pt = [[] for _ in range(dataset.n_machine)]

    for i in range(len(op_data)):
        for j in range(len(op_data[i])):
            # op_data[i][j][0] : machine
            m = op_data[i][j][0]
            machines[m].append(np.where(op_data[i, :, 0] == m))  # 자신의 위치
            machines_pt[m].append(op_data[i][j][1])

    plt.figure(figsize=(8, 6))
    for i in range(dataset.n_machine):
        for j in range(dataset.n_job):
            plt.scatter(i, machines[i][j], color='black', alpha=0.2, s=(machines_pt[i][j]))

    plt.title(dataset.name + ' Data Distribution')
    plt.savefig('..\\' + dataset.path + dataset.name + '_Data_Distribution.png')
    # plt.show()

def show_pt_distribution(dataset):
    op_data = np.array(dataset.op_data)
    pt = op_data[:,:,1].reshape(-1).tolist()
    plt.figure()
    plt.hist(pt)
    plt.title(dataset.name+' Processing Time Distribution')
    plt.savefig('..\\'+dataset.path + dataset.name+'_PT_Distribution.png')
    # plt.show()
