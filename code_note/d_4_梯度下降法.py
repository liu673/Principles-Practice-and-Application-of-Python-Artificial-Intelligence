# -*-coding:utf-8-*-
from random import random

"""
梯度下降法是求解无约束优化问题的一种简单而有效的优化方法，它是一种利用目标函数的Taylor展式构造搜索方向的方法。
梯度下降法三要素：出发点、下降方向、下降步长

梯度的方向就是函数上升最快的方向，那么梯度的反方向就是给定函数在给定位置下降最快的方向，
所以我们沿着梯度相反的方向一直走，每走一段，重复上面的方法，最后成功抵达山谷，也就可求得函数的最小值。
"""

"""
梯度下降时使用数据量的不同，梯度下降法可以分为3类：
（1）批量梯度下降法（Batch Gradient Descent，BGD）
# 每次都使用训练集中的所有样本来更新参数，因此每次更新都会朝着正确的方向进行，最后能够保证收敛到极值点，凸函数收敛到全局最优解，非凸函数收敛到局部最优解。
# 当样本数据集很大时，批量梯度下降法的速度就会非常慢，学习时间太长，消耗大量内存。

（2）随机梯度下降法（Stochastic Gradient Descent，SGD）
# 使用全部样本数据可能会造成训练过程过慢，随机梯度下降法（SGD）每轮迭代只从样本中选择一条数据进行梯度下降，这样经过足够多的迭代次数，SGD也可以发挥作用。
# SGD的缺点在于每次更新可能并不会按照正确的方向进行，参数更新具有高方差，从而导致损失函数剧烈波动，
# 不过，SGD可以使优化方向从一个极小点跳到另一个极小点，对于非凸函数而言，可能会找到#全局最优点


（3）小批量梯度下降法（Mini-Batch Gradient Descent，MBGD）
综合了了SGD和BGD的优点，同时弱化了缺点。
# BGD和SGD收敛速度快，但是收敛性不稳定。MBGD是BGD和SGD的折中方案，MBGD每次迭代多个样本。
# MBGD降低了SGD训练过程的杂乱程度，同时也保证了速度。并且如果Batch Size选择合理，不仅收敛速度比SGD更快、更稳定，
# 而且在最优解附近的跳动也不会很大，甚至得到比Batch Gradient Descent更好的解。
"""


# 梯度下降法
def gradient_decent(fn, partial_derivatives, n_variables,
                    lr=0.1, max_iter=10000, tolerance=1e-5):
    theta = [random() for _ in range(n_variables)]
    y_cur = fn(*theta)
    for i in range(max_iter):
        # 计算梯度
        gradient = [f(*theta) for f in partial_derivatives]
        # 更新theta通过梯度
        for j in range(n_variables):
            theta[j] -= gradient[j] * lr
        y_cur, y_pre = fn(*theta), y_cur
        if abs(y_pre - y_cur) < tolerance:
            break
    return theta, y_cur


def f(x, y):
    return (x + y - 3) ** 2 + (x + 2 * y - 5) ** 2 + 2


def df_dx(x, y):
    return 2 * (x + y - 3) + 2 * (x + 2 * y - 5)


def df_dy(x, y):
    return 2 * (x + y - 3) + 4 * (x + 2 * y - 5)


def main():
    n_variables = 2

    theat, f_theat = gradient_decent(f, [df_dx, df_dy], n_variables)
    theat = [round(x, 3) for x in theat]
    print('结果如下： theta={} , f(theta)={:.2f}'.format(theat, f_theat))


if __name__ == '__main__':
    main()
