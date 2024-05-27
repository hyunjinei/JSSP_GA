import matplotlib.pyplot as plt
import numpy as np

def show_evolution(makespan, score, title, text=None):
    """
    n_division : 몇 개로 구분해 보여줄 것인가?
    n_generation : 한 구분 당 몇 개의 세대를 보여줄 것인가?
    """
    fig = plt.figure(figsize=(12, 8))
    axes = [fig.add_subplot(3, 2, i + 1) for i in range(6)]


    score = np.array(score)
    color = ['red', 'blue', 'green']
    for i in range(6):
         axes[i].scatter(makespan, score[i,:], color='red', alpha=0.1, s=4)

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
    if text is not None:
        plt.text(0.8, 0.5, text, ha='center', va='center', transform=axes[1].transAxes,fontsize=12)
    plt.savefig(title+'.png')
    plt.show()