import matplotlib.pyplot as plt
import numpy as np

def show_evolution(n_division, n_generation,makespan, score, title):
    """
    n_division : 몇 개로 구분해 보여줄 것인가?
    n_generation : 한 구분 당 몇 개의 세대를 보여줄 것인가?
    """
    fig = plt.figure(figsize=(8, 6))
    axes = [fig.add_subplot(3, 2, i + 1) for i in range(6)]



    color = ['red', 'blue', 'green']
    for d in range(n_division):
        for i in range(6):
            for j in range(n_generation):
                axes[i].scatter(makespan[j + 20 * d], score[i][j + 20 * d], color=color[d], alpha= ((1./(n_generation))*j))

    pearson_corr = [0.0 for i in range(6)]
    for i in range(6):
        correlation_matrix = np.corrcoef(makespan, score[i])
        pearson_corr[i] = correlation_matrix[0, 1]
    print(makespan)
    print('Pearson Correlation Coefficient')
    print('Kendall Tau :', pearson_corr[0])
    print('Spearman Rank :', pearson_corr[1])
    print('Spearman Footrule :', pearson_corr[2])
    print('MSE :', pearson_corr[3])
    print('Bubble Sort :', pearson_corr[4])
    print('Pearson :', pearson_corr[5])

    axes[0].set_title('Kendall Tau ' + str(round(pearson_corr[0], 4)))
    axes[1].set_title('spearman_rank ' + str(round(pearson_corr[1], 4)))
    axes[2].set_title('spearman_footrule ' + str(round(pearson_corr[2], 4)))
    axes[3].set_title('MSE ' + str(round(pearson_corr[3], 4)))
    axes[4].set_title('Bubble Sort ' + str(round(pearson_corr[4], 4)))
    axes[5].set_title('Pearson ' + str(round(pearson_corr[5], 4)))
    # print('bubble_sort :',pearson_corr[3])
    plt.tight_layout()
    plt.suptitle(title)
    plt.show()