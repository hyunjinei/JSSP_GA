"""
Abz5, Abz6 데이터에 대해 SpearmanFootrule 점수와 makespan과의 상관관계를 보여주는 그림그리기용 코드
"""

import matplotlib.pyplot as plt
from Data.Adams.abz5.abz5 import Dataset as abz5
from Data.Adams.abz6.abz6 import Dataset as abz6
import numpy as np
from Config.Run_Config import Run_Config
from objects import *

if __name__ == '__main__':
    abz5data = abz5()
    abz6data = abz6()
    # n_sol_abz5 = abz5data.n_solution
    # n_sol_abz6 = abz6data.n_solution
    n_sol_abz5 = 400
    n_sol_abz6 = 400

    makespan = [0 for i in range(n_sol_abz5 * 5)]
    score = [0 for i in range(n_sol_abz5  * 5)]
    config = Run_Config(10, 10, 100, False, False, False, False, False, False)
    for idx, s in enumerate(abz5data.solution_list[:400]):
        ind = Individual(config, solution_seq=s, op_data=abz5data.op_data)
        makespan[idx] = ind.makespan
        score[idx] = ind.score[2]
    for i in range(n_sol_abz5 * 1, n_sol_abz5 * 5):
        s = np.random.permutation(100)
        ind = Individual(config, seq=s, op_data=abz5data.op_data)
        makespan[i] = ind.makespan
        score[i] = ind.score[2]

    makespan2 = [0 for i in range(n_sol_abz6 * 5)]
    score2 = [0 for i in range(n_sol_abz6 * 5)]

    for idx, s in enumerate(abz6data.solution_list[:400]):
        ind = Individual(config, solution_seq=s, op_data=abz6data.op_data)
        makespan2[idx] = ind.makespan
        score2[idx] = ind.score[2]
    for i in range(n_sol_abz6 * 1, n_sol_abz6 * 5):
        s = np.random.permutation(100)
        ind = Individual(config, seq=s, op_data=abz6data.op_data)
        makespan2[i] = ind.makespan
        score2[i] = ind.score[2]

    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams.update({'font.size': 22})

    fig = plt.figure(figsize=(6, 4))
    axes = [fig.add_subplot(2, 1, i + 1) for i in range(2)]
    makespan = np.array(makespan)
    score = np.array(score)

    makespan2 = np.array(makespan2)
    score2 = np.array(score2)
    color = ['red', 'blue', 'green']
    axes[0].scatter(makespan[:n_sol_abz5], score[:n_sol_abz5], color='red', alpha=0.1, s=4, label='Optimal')
    axes[0].scatter(makespan[n_sol_abz5:], score[n_sol_abz5:], color='blue', alpha=0.1, s=4, label='Random')
    axes[1].scatter(makespan2[:n_sol_abz6], score2[:n_sol_abz6], color='red', alpha=0.1, s=4, label='Optimal')
    axes[1].scatter(makespan2[n_sol_abz6:], score2[n_sol_abz6:], color='blue', alpha=0.1, s=4, label='Random')


    correlation_matrix = np.corrcoef(makespan, score)
    pearson_corr = correlation_matrix[0, 1]

    correlation_matrix2 = np.corrcoef(makespan2, score2)
    pearson_corr2 = correlation_matrix2[0, 1]


    axes[0].set_title('abz5(10×10)')
    axes[1].set_title('abz6(10×10)')
    axes[0].legend()
    axes[1].legend()
    axes[0].set_xlabel('Makespan')
    axes[1].set_xlabel('Makespan')
    axes[0].set_ylabel('MIO score')
    axes[1].set_ylabel('MIO score')

    leg = axes[0].legend()
    for lh in leg.legendHandles:
        lh.set_alpha(1)

    leg2 = axes[1].legend()
    for lh in leg2.legendHandles:
        lh.set_alpha(1)
    fig.tight_layout()
    print('Pearson Correlation:'+str(round(pearson_corr, 4)))
    print('Pearson Correlation2:'+str(round(pearson_corr2, 4)))
    # plt.text(0.7, 0.1, 'Pearson Correlation:'+str(round(pearson_corr, 4)), ha='center', va='center', transform=axes[0].transAxes,fontsize=8)
    # plt.text(0.7, 0.1, 'Pearson Correlation:'+str(round(pearson_corr2, 4)), ha='center', va='center', transform=axes[1].transAxes,fontsize=8)
    plt.savefig('Validation.png')
    plt.show()