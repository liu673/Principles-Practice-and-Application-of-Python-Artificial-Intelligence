# -*-coding:utf-8-*-
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


def diff_f_dis():
    """
    不同参数下的F分布
    分布经常被用来对两个样本方差进行比较。它是方差分析的一个基本分布，也被用于回归分析中的显著性检验。
    分布有两个参数：dfn和dfd，分别代表分子上的第一自由度和分母上的第二自由度。
    :return:
    """
    # f_dis_10_1 = stats.f(dfn=10, dfd=1)
    f_dis_1_30 = stats.f(dfn=1, dfd=30)
    f_dis_30_5 = stats.f(dfn=30, dfd=5)
    f_dis_30_30 = stats.f(dfn=30, dfd=30)
    f_dis_30_100 = stats.f(dfn=30, dfd=100)
    f_dis_100_100 = stats.f(dfn=100, dfd=100)

    # x1 = np.linspace(f_dis_10_1.ppf(0.01), f_dis_10_1.ppf(0.99), 100)
    x2 = np.linspace(f_dis_1_30.ppf(0.2),     f_dis_1_30.ppf(0.99), 100)
    x3 = np.linspace(f_dis_30_5.ppf(0.00001),    f_dis_30_5.ppf(0.99), 100)
    x4 = np.linspace(f_dis_30_30.ppf(0.00001),   f_dis_30_30.ppf(0.999), 100)
    x6 = np.linspace(f_dis_30_100.ppf(0.0001),   f_dis_30_100.ppf(0.999), 100)
    x5 = np.linspace(f_dis_100_100.ppf(0.0001),   f_dis_100_100.ppf(0.9999), 100)

    fig, ax = plt.subplots(1, 1)
    # ax.plot(x1, f_dis_10_1.pdf(x1), 'b-', lw=2, label='F(10, 1)')
    ax.plot(x2, f_dis_1_30.pdf(x2), 'g-', lw=2, label='t(1, 30)')
    ax.plot(x3, f_dis_30_5.pdf(x3), 'r-', lw=2, label='t(30, 5)')
    ax.plot(x4, f_dis_30_30.pdf(x4), 'm-', lw=2, label='t(30, 30)')
    ax.plot(x6, f_dis_30_100.pdf(x5), 'c-', lw=2, label='t(30, 100)')
    ax.plot(x5, f_dis_100_100.pdf(x5), 'y-', lw=2, label='t(100, 100)')

    plt.ylabel('Probability')
    plt.title('PDF of F Distribution')
    ax.legend(loc='best', frameon=False)
    # plt.savefig('f_diff_pdf.png', dip=500)
    plt.show()


diff_f_dis()
