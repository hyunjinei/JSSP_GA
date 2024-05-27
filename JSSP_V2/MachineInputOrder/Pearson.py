import numpy as np
import matplotlib.pyplot as plt
from GA_pyGAD.GA import Individual
# from Data.Adams.abz5.abz5 import Dataset
from Data.Adams.abz6.abz6 import Dataset
# from Data.FT.ft10.ft10 import Dataset
from Config.Run_Config import Run_Config

dataset = Dataset()
op_data = dataset.op_data

config = Run_Config(dataset.n_job, dataset.n_machine, dataset.n_op, False, False, False, False, False, False)

NUM_ITERATION = 1000
num_solution = 50

makespan = [0 for i in range(num_solution+NUM_ITERATION)]
score = [[0 for i in range(num_solution+NUM_ITERATION)] for j in range(6)]
# for idx, s in enumerate(dataset.solution_list):
for idx, s in enumerate(dataset.solution_list[:num_solution]):
    ind = Individual(config, solution_seq=s, op_data=op_data)
    makespan[idx] = ind.makespan
    score[0][idx] = ind.score[0]
    score[1][idx] = ind.score[1]
    score[2][idx] = ind.score[2]
    score[3][idx] = ind.score[3]
    score[4][idx] = ind.score[4]
    score[5][idx] = ind.score[5]
print('Solution Finished')
print('Generating New solutions...')
popul = []
for i in range(num_solution, num_solution+NUM_ITERATION):
    seq = np.random.permutation(np.arange(dataset.n_op))
    individual = Individual(config, seq=seq, op_data=op_data)
    popul.append(individual)
    makespan[i] = individual.makespan
    score[0][i] = individual.score[0]
    score[1][i] = individual.score[1]
    score[2][i] = individual.score[2]
    score[3][i] = individual.score[3]
    score[4][i] = individual.score[4]
    score[5][i] = individual.score[5]

fig = plt.figure(figsize=(8,6))
ax1 = fig.add_subplot(3, 2, 1)
ax2 = fig.add_subplot(3, 2, 2)
ax3 = fig.add_subplot(3, 2, 3)
ax4 = fig.add_subplot(3, 2, 4)
ax5 = fig.add_subplot(3, 2, 5)
ax6 = fig.add_subplot(3, 2, 6)

color = 'tab:red'

ax1.scatter(makespan[:num_solution], score[0][:num_solution], color='tab:red', s=4, alpha=0.2)
ax2.scatter(makespan[:num_solution], score[1][:num_solution], color='tab:red', s=4, alpha=0.2)
ax3.scatter(makespan[:num_solution], score[2][:num_solution], color='tab:red', s=4, alpha=0.2)
ax4.scatter(makespan[:num_solution], score[3][:num_solution], color='tab:red', s=4, alpha=0.2)
ax5.scatter(makespan[:num_solution], score[4][:num_solution], color='tab:red', s=4, alpha=0.2)
ax6.scatter(makespan[:num_solution], score[5][:num_solution], color='tab:red', s=4, alpha=0.2)

ax1.scatter(makespan[num_solution:], score[0][num_solution:], color='tab:blue', s=4, alpha=0.2)
ax2.scatter(makespan[num_solution:], score[1][num_solution:], color='tab:blue', s=4, alpha=0.2)
ax3.scatter(makespan[num_solution:], score[2][num_solution:], color='tab:blue', s=4, alpha=0.2)
ax4.scatter(makespan[num_solution:], score[3][num_solution:], color='tab:blue', s=4, alpha=0.2)
ax5.scatter(makespan[num_solution:], score[4][num_solution:], color='tab:blue', s=4, alpha=0.2)
ax6.scatter(makespan[num_solution:], score[5][num_solution:], color='tab:blue', s=4, alpha=0.2)

pearson_corr = [0.0 for i in range(6)]
for i in range(6):
    correlation_matrix = np.corrcoef(makespan, score[i])
    pearson_corr[i] = correlation_matrix[0, 1]
print('Pearson Correlation Coefficient')
print('Kendall Tau :', pearson_corr[0])
print('Spearman Rank :', pearson_corr[1])
print('Spearman Footrule :', pearson_corr[2])
print('MSE :', pearson_corr[3])
print('Bubble Sort :', pearson_corr[4])
print('Pearson :', pearson_corr[5])

ax1.set_title('Kendall Tau '+ str(round(pearson_corr[0],4)))
ax2.set_title('spearman_rank '+ str(round(pearson_corr[1],4)))
ax3.set_title('spearman_footrule '+ str(round(pearson_corr[2],4)))
ax4.set_title('MSE '+ str(round(pearson_corr[3],4)))
ax5.set_title('Bubble Sort '+ str(round(pearson_corr[4],4)))
ax6.set_title('Pearson '+ str(round(pearson_corr[5],4)))
# print('bubble_sort :',pearson_corr[3])
plt.tight_layout()
plt.suptitle(dataset.name)
plt.show()