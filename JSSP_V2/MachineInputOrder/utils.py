import numpy as np

from scipy.stats import kendalltau, spearmanr


def kendall_tau_distance(x, y):
    tau, _ = kendalltau(x, y)
    return tau

def spearman_rank_correlation(x, y):
    rho, _ = spearmanr(x, y)
    return rho

def spearman_footrule_distance(x, y):
    distance = np.sum(np.abs(np.argsort(x) - np.argsort(y)))
    return distance

def bubble_sort_distance(x):
    n = len(x)
    sorted_x = np.sort(x)
    distance = np.sum(np.where(x != sorted_x)[0])
    return distance

def positional_distance(x, y):
    distance = np.sum(np.abs(np.where(x != y)[0] - np.where(x != y)[0]))
    return distance

def MSE(x, y):
    # 2차원 리스트를 NumPy 배열로 변환
    x = np.array(x)
    y = np.array(y)

    # 각 원소 간의 평균 제곱 오차 계산
    mse = np.square(np.subtract(x, y)).mean()

    return mse

