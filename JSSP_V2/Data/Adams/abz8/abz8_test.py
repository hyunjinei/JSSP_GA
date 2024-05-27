
import numpy as np
import matplotlib.pyplot as plt
from GA_pyGAD.GA import *
from abz8 import Dataset
from Config.Run_Config import Run_Config
config = Run_Config(20, 15, 300, False, False, False, False, False, False)

from MachineInputOrder.utils import kendall_tau_distance, spearman_footrule_distance, spearman_rank_correlation, \
    bubble_sort_distance, MSE

dataset = Dataset()
op_data = dataset.op_data
popul = []
num_repeat = 10
makespan = [0.0 for i in range(num_repeat)]
score = [[0.0 for i in range(num_repeat)] for j in range(6)]
for i in range(num_repeat):
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

fig = plt.figure(figsize=(8, 6))
ax1 = fig.add_subplot(3, 2, 1)
ax2 = fig.add_subplot(3, 2, 2)
ax3 = fig.add_subplot(3, 2, 3)
ax4 = fig.add_subplot(3, 2, 4)
ax5 = fig.add_subplot(3, 2, 5)
ax6 = fig.add_subplot(3, 2, 6)

color = 'tab:red'
ax1.scatter(makespan, score[0], color='tab:blue', s=4, alpha=0.2)
ax2.scatter(makespan, score[1], color='tab:blue', s=4, alpha=0.2)
ax3.scatter(makespan, score[2], color='tab:blue', s=4, alpha=0.2)
ax4.scatter(makespan, score[3], color='tab:blue', s=4, alpha=0.2)
ax5.scatter(makespan, score[4], color='tab:blue', s=4, alpha=0.2)
ax6.scatter(makespan, score[5], color='tab:blue', s=4, alpha=0.2)

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

ax1.set_title('Kendall Tau ' + str(round(pearson_corr[0], 4)))
ax2.set_title('spearman_rank ' + str(round(pearson_corr[1], 4)))
ax3.set_title('spearman_footrule ' + str(round(pearson_corr[2], 4)))
ax4.set_title('MSE ' + str(round(pearson_corr[3], 4)))
ax5.set_title('Bubble Sort ' + str(round(pearson_corr[4], 4)))
ax6.set_title('Pearson ' + str(round(pearson_corr[5], 4)))
# print('bubble_sort :',pearson_corr[3])
plt.tight_layout()
plt.suptitle(dataset.name)
plt.show()
plt.savefig(dataset.name + '.png')
