# -*-coding:utf-8-*-
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


def diff_chi2_dis():
    """
    不同参数下的卡方分布
    当自由度df等于1或2时，函数图像都呈单调递减的趋势；当df大于或等于3时，呈先增后减的趋势。从定义上来看，df的值只能取正整数
    :return:
    """
    # chi2_dis_0_5 = stats.chi2(df=0.5)
    chi2_dis_1 = stats.chi2(df=1)
    chi2_dis_4 = stats.chi2(df=4)
    chi2_dis_10 = stats.chi2(df=10)
    chi2_dis_20 = stats.chi2(df=20)

    # x1 = np.linspace(chi2_dis_0_5.ppf(0.01), chi2_dis_1.ppf(0.99), 100)
    x2 = np.linspace(chi2_dis_1.ppf(0.65), chi2_dis_1.ppf(0.9999999), 100)
    x3 = np.linspace(chi2_dis_4.ppf(0.000001), chi2_dis_4.ppf(0.999999), 100)
    x4 = np.linspace(chi2_dis_10.ppf(0.000001), chi2_dis_10.ppf(0.99999), 100)
    x5 = np.linspace(chi2_dis_20.ppf(0.00000001), chi2_dis_20.ppf(0.9999), 100)

    fig, ax = plt.subplots(1, 1)
    # ax.plot(x1, chi2_dis_0_5.pdf(x1), 'b-', lw=2, label='df = 0.5')
    ax.plot(x2, chi2_dis_1.pdf(x2), 'g-', lw=2, label='df = 1')
    ax.plot(x3, chi2_dis_4.pdf(x3), 'r-', lw=2, label='df = 4')
    ax.plot(x4, chi2_dis_10.pdf(x4), 'b-', lw=2, label='df = 10')
    ax.plot(x5, chi2_dis_20.pdf(x5), 'y-', lw=2, label='df = 20')

    plt.ylabel('Probability')
    plt.title('PDF of $\chi^2$ Distribution')
    ax.legend(loc='best', frameon=False)
    plt.show()


diff_chi2_dis()
