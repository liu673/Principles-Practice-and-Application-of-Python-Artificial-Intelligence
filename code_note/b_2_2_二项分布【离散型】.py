# -*-coding:utf-8-*-
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

"""
如果把一个伯努利分布独立地重复 n次，就得到了一个二项分布。
二项分布是最重要的离散型概率分布之一。随机变量 要满足这个分布有两个重要条件：
①各次试验的条件是稳定的；
②各次试验之间是相互独立的。
"""

"""
利用抛硬币的例子来比较伯努利分布和二项分布的区别。
如果将抛一次硬币看作一次伯努利实验，且将正面朝上记为1，反面朝上记为0。那么抛 n次硬币，记录正面朝上的次数Y， 就服从二项分布。
假如硬币是均匀的， 的取值应该大部分集中在 n/2附近，而非常大或非常小的值都很少。
由此可见，二项分布关注的是计数，而伯努利分布关注的是比值（正面朝上的计数/ n）。
一个随机变量 服从参数为 n和 p的二项分布，记作 X～Binomial（n ， p）或 X～B（ n， p）。
"""


def binom_dis(n=1, p=1):
    """
    n=20， =0.6的二项分布，表示每次试验抛硬币，该硬币正面朝上的概率大于背面朝上的概率，共抛20次并记录正面朝上的次数。
    :param n: 次数
    :param p: 正面朝上的概率
    :return:
    """
    binom_dis = stats.binom(n, p)
    # ppf 累积分布函数（cdf）的反函数
    x = np.arange(binom_dis.ppf(0.0001), binom_dis.ppf(0.9999))
    print(x)  # [ 4.  5.  6.  7.  8.  9. 10. 11. 12. 13. 14. 15. 16. 17. 18.]
    fig, ax = plt.subplots(1, 1)
    ax.plot(x, binom_dis.pmf(x), 'bo', label='binom pmf')
    ax.vlines(x, 0, binom_dis.pmf(x), colors='b', lw=5, alpha=0.5)
    ax.legend(loc='best', frameon=False)

    plt.ylabel('Probability')
    plt.title('PMF of binomial distribution(n = {}, p = {})'.format(n, p))
    plt.show()


binom_dis(n=20, p=0.6)
