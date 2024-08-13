# -*-coding:utf-8-*-
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


def diff_t_dis():
    """
    不同参数下的t分布
    当自由度df等于1或2时，函数图像都呈单调递减的趋势；当df大于或等于3时，呈先增后减的趋势。从定义上来看，df的值只能取正整数
    与标准正态分布之间的差别还是比较大的，
    但是当自由度 趋近于无穷大时， 分布与标准正态分布没有差别（公式上的形式将变得完全相同，这里没有列出概率密度函数的公式）。
    较大的区别在于，当自由度 较小时， 分布比标准正态分布的尾部（Fatter Tails）更宽，因此也比正态分布更慢地趋近于0。
    :return:
    """
    norm_dis = stats.norm()
    t_dis_1 = stats.t(df=1)
    t_dis_4 = stats.t(df=4)
    t_dis_10 = stats.t(df=10)
    t_dis_20 = stats.t(df=20)

    x1 = np.linspace(norm_dis.ppf(0.000001), norm_dis.ppf(0.999999), 1000)
    x2 = np.linspace(t_dis_1.ppf(0.04), t_dis_1.ppf(0.96), 1000)
    x3 = np.linspace(t_dis_4.ppf(0.001), t_dis_4.ppf(0.999), 1000)
    x4 = np.linspace(t_dis_10.ppf(0.001), t_dis_10.ppf(0.999), 1000)
    x5 = np.linspace(t_dis_20.ppf(0.001), t_dis_20.ppf(0.999), 1000)

    fig, ax = plt.subplots(1, 1)
    ax.plot(x1, norm_dis.pdf(x1), 'r-', lw=2, label='N(0, 1)')
    ax.plot(x2, t_dis_1.pdf(x2), 'b-', lw=2, label='t(1)')
    ax.plot(x3, t_dis_4.pdf(x3), 'g-', lw=2, label='t(4)')
    ax.plot(x4, t_dis_10.pdf(x4), 'm-', lw=2, label='t(10)')
    ax.plot(x5, t_dis_20.pdf(x5), 'y-', lw=2, label='t(20)')

    plt.ylabel('Probability')
    plt.title('PDF of t Distribution')
    ax.legend(loc='best', frameon=False)
    plt.show()


diff_t_dis()




