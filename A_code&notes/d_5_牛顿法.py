# -*-coding:utf-8-*-
import numpy as np

"""
求解无约束非线性规划问题的牛顿法（Newton法）是利用目标函数的二次Taylor展式构造搜索方向的方法。
从本质上看，牛顿法是二阶收敛，梯度下降法是一阶收敛，所以牛顿法更快。

通俗地说，例如你想找一条最短的路径走到一个盆地的最底部，梯度下降法每次只从你当前所处位置选一个坡度最大
的方向走一步，而牛顿法在选择方向时，不仅会考虑坡度是否够大，还会考虑你走了一步之后，坡度是否会变得更大。所以，可以说牛顿
法比梯度下降法看得更远一点，能更快地走到最底部。牛顿法目光更加长远，所以少走弯路；相对而言，梯度下降法只考虑了局部的最优，而没有全局思想
"""


# 牛顿法求解无约束优化问题
def fd(x):
    t = np.asarray([2, 4])
    # y = np.dot(x.T, t)
    y = x.T * t
    return y


def fdd():
    # ys = 12 * x ** 2 - 24 * x - 12
    a = np.asarray([[2, 0], [0, 4]])
    A = np.matrix(a)
    return A.I


fdd()
i = 1
x0 = np.asarray([1, 2])  # 3.00000
ans = pow(10, -6)
fd0 = fd(x0)
fdd0 = fdd()
while np.linalg.norm(fd0) > ans:
    x1 = x0 - (fd0 * fdd0)
    x0 = x1
    print('次数： {}， 所得的值x：{}'.format(i, x1))
    i += 1
    fd0 = fd(x0)
    fdd0 = fdd()
else:
    print('运算结束，找到最优值')
    print('最优值：X={}'.format(x0))
