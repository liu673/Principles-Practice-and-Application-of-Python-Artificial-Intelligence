# -*-coding:utf-8-*-
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

"""
均匀分布（Uniform Distribution）是最简单的连续型概率分布，因为其概率密度是一个常数，不随随机变量 取值的变化而变化。
如果连续型随机变量 具有如下的概率密度函数，则称 服从[ a， b]上的均匀分布，记作 X～U（ a， b）或 X～Unif（ a， b）。
f(x)= 1 / (b - a), a< x < b
    = 0 , x< a 或 x > b
"""


def uniform_distribution(loc=0, scale=1):
    """
    直接传入参数和先冻结一个分布，画出来均匀分布的概率分布函数，此外还从该分布中取了10000个值作直方图
    :param loc: location表示起点，
    :param scale: 表示区间长度
    :return:
    """
    uniform_dis = stats.uniform(loc=loc, scale=scale)
    x = np.linspace(uniform_dis.ppf(0.01), uniform_dis.ppf(0.99), 100)
    fig, ax = plt.subplots(1, 1)
    # 直接传入参数
    ax.plot(x, stats.uniform.pdf(x, loc=2, scale=4), 'r-', lw=5, alpha=0.6, label='uniform pdf')
    # 从冻结的均匀分布取值
    ax.plot(x, uniform_dis.pdf(x), 'k-', lw=2, label='frozen pdf')
    # 计算pdf分别等于0.001、0.5和0.999是的x值
    vals = uniform_dis.ppf([0.001, 0.5, 0.999])
    print(vals)  # [2.004 4.    5.996]
    # 检测cdf和pdf的精确度
    print(np.allclose([0.001, 0.5, 0.999], uniform_dis.cdf(vals)))
    # True

    r = uniform_dis.rvs(size=10000)
    ax.hist(r, density=True, histtype='stepfilled', alpha=0.2)
    plt.ylabel('Probability')
    plt.title('PDF of Unif({}, {})'.format(loc, scale))
    ax.legend(loc='best', frameon=False)
    plt.show()


uniform_distribution(loc=2, scale=4)

