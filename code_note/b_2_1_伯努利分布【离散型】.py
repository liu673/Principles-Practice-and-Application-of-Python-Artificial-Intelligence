# -*-coding:utf-8-*-
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


"""
伯努利分布（Bernoulli Distribution）又称两点分布或0-1分布，
其样本空间中只有两个点，一般取为{0，1}，不同的伯努利分布只是取到这两个值的概率不同。
伯努利分布只有一个参数 ，记作 X～Bernoulli（p），或 X～B（1， p），读作X 服从参数为 p的伯努利分布
"""

def bernoulli_pmf(p=0.0):
    """
    抛硬币  描述离散型随机变量的概率质量分布函数（Probability Mass Function，PMF）
    :param p: 硬币正面朝上的概率
    :return:
    """
    ber_dist = stats.bernoulli(p)
    x = [0, 1]
    x_name = ['0', '1']
    pmf = [ber_dist.pmf(x[0]), ber_dist.pmf(x[1])]
    plt.bar(x, pmf, width=0.15)
    plt.xticks(x, x_name)
    plt.ylabel('Probability')
    plt.title('PMF of bernoulli distribution')
    plt.show()

bernoulli_pmf(p=0.3)












