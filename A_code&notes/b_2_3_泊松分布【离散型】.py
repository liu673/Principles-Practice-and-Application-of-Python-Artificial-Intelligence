# -*-coding:utf-8-*-
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

"""
如果某事件以固定强度 随机且独立地出现，该事件在单位时间内出现的次数（个数）可以看成是服从泊松分布。
我们把一个随机变量 X服从参数为 lambda 的泊松分布，记作 X～Poisson（ lambda），
或 X～P（ lambda）。泊松分布适合于描述单位时间内随机事件发生次数的概率分布

"""


def poisson_pmf(mu=3):
    poisson_dis = stats.poisson(mu)
    # print(poisson_dis.ppf(0.001))
    # print(poisson_dis.ppf(0.999))
    # print(poisson_dis.cdf(0.001))
    x = np.arange(poisson_dis.ppf(0.001), poisson_dis.ppf(0.999))
    # print(type(x))
    # print(poisson_dis.pmf(x))
    print(x)  # [ 1.  2.  3.  4.  5.  6.  7.  8.  9. 10. 11. 12. 13. 14. 15. 16. 17.]
    fig, ax = plt.subplots(1, 1)            # 子图行列数为 1
    ax.plot(x, poisson_dis.pmf(x), 'bo', ms=8, label='Poisson pmf')
    # matplotlib库的axiss模块中的Axes.vlines()函数用于在从ymin到ymax的每个x处绘制垂直线。
    ax.vlines(x, 0, poisson_dis.pmf(x), colors='b', lw=5, alpha=0.5)
    # 图例位置  图例边框
    ax.legend(loc='best', frameon=False)
    plt.ylabel('Probability')
    plt.title('PMF of poisson distribution(mu = {})'.format(mu))
    plt.show()


poisson_pmf(mu=8)
