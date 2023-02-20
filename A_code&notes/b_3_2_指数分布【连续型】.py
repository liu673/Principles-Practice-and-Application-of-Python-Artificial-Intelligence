# -*-coding:utf-8-*-
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

"""
指数分布和离散型的泊松分布之间有很大的关系。
泊松分布表示单位时间（或单位面积）内随机事件的平均发生次数，
指数分布则可以用来表示独立随机事件发生的时间间隔。
由于发生次数只能是自然数，所以泊松分布自然就是离散型的随机变量，而时间间隔则可以是任意的实数，因此其定义域是（0，+∞）
"""

"""
如果一个随机变量 的概率密度函数满足以下形式，
f(x) = lambda * e ^(-lambda * e), x > 0
     = 0, 其他
就称 X为服从参数 lambda 的指数分布（Exponential Distribution），记作 X～E（lambda ）
或 X～Exp（lambda ）。指数分布只有一个参数 lambda，且 lambda ＞0。

指数分布的一个显著的特点是其具有无记忆性。
"""

def exponential_dis(loc=0, scale=1.0):
    """
    指数分布， exponential continuous random variable
    按照定义，指数分布只有一个参数lambda， 这里的scale = 1 / lambda
    :param loc: 定义与的左端点，相当于将整体分布沿x轴平移loc
    :param scale: lambda的倒数，loc + scale表示该分布的均值， scale ^ 2表示该分布的方差
    :return:
    """
    exp_dis = stats.expon(loc=loc, scale=scale)
    x = np.linspace(exp_dis.ppf(0.000001), exp_dis.ppf(0.999999), 100)
    fig, ax = plt.subplots(1, 1)

    # 直接传入参数
    ax.plot(x, stats.expon.pdf(x, loc=loc, scale=scale), 'r-', lw=5, alpha=0.6, label='exponential pdf')
    # 从冻结的均匀分布取值
    ax.plot(x, exp_dis.pdf(x), 'k-', lw=2, label='frozen pdf')
    # 计算pdf分别等于0.001、0.5和0.999是的x值
    vals = exp_dis.ppf([0.001, 0.5, 0.999])
    print(vals)  # [2.00100067e-03 1.38629436e+00 1.38155106e+01]
    # 检测cdf和pdf的精确度
    print(np.allclose([0.001, 0.5, 0.999], exp_dis.cdf(vals)))
    # True

    r = exp_dis.rvs(size=10000)
    ax.hist(r, density=True, histtype='stepfilled', alpha=0.2)
    plt.ylabel('Probability')
    plt.title('PDF of Exp(0.5)')
    ax.legend(loc='best', frameon=False)
    plt.show()

exponential_dis(loc=0, scale=2)






