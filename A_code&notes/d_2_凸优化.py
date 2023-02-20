# -*-coding:utf-8-*-
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.optimize as spo

"""
凸集的几何意义：假如集合中有两个点，则这两个点的连线上的任意一点依然属于此集合
投影定理和凸集分离定理是研究约束规划最优性条件和对偶理论的重要工具。
"""
"""
设集合C为非空的凸集，函数f（x）定义在集合C上，如果∀ X1，X2∈ C，lambda ∈[0，1]，
有 f（ X1+（1-lambda ）X2）≤ lambda f（X1）+（1- lambda）f（X2），则称 f为 C上的凸函数

凸函数的集合解释告诉我们，一个凸函数的图形总是位于相应弦的下方。
"""
"""
凸函数有以下性质：
（1）设f是定义在凸集 C上的凸函数，实数a ≥0，则 af 也是定义在C上的凸函数。
（2）设 f1， f2是定义在凸集 C上的凸函数，则 f1+ f2也是定义在C上的凸函数。
（3）设 f1， f2，…，fm 是定义在凸集 C上的凸函数，实数 a1， a2，…， am≥0，则 sum(ai * fi)也是定义在C 上的凸函数
"""

"""
在金融学和经济学中，凸优化起着重要作用，这方面的例子包括市场数据校准和期权定价模型，或者效用函数的优化
"""


def fm(*args):
    """
    效用函数优化
    :param args:
    :return:
    """
    return (np.sin(args[0]) + 0.05 * args[0] ** 2 + np.sin(args[1]) + 0.05 * args[1] ** 2)


# x = np.linspace(-10, 10, 50)
# y = np.linspace(-10, 10, 50)
# # 将x中每一个数据和y中每一个数据组合生成很多点,然后将这些点的x坐标放入到X中,y坐标放入Y中,并且相应位置是对应的
# x, y = np.meshgrid(x, y)
# z = fm(x, y)
#
# fig = plt.figure(figsize=(9, 6))  # 指定figure的宽和高，单位为英寸
# ax = fig.gca(projection='3d')  # 建立一个3d坐标系
# surf = ax.plot_surface(x, y, z, rstride=2, cstride=2,  # rstride:行之间的跨度  cstride:列之间的跨度
#                        cmap=mpl.cm.coolwarm, linewidth=0.5, antialiased=True)
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# fig.colorbar(surf, shrink=0.5, aspect=5)
# # plt.show()


# 全局优化
def fo(*args):
    x = args[0][0]
    y = args[0][1]
    z = np.sin(x) + 0.05 * x ** 2 + np.sin(y) + 0.05 * y ** 2
    # print(x, y, z)
    return z


opt = spo.brute(fo, ((-10, 10, 0.1), (-10, 10, 0.1)), finish=None)
print(opt)                      # [-1.4 -1.4]
print(fm(opt[0], opt[1]))       # -1.7748994599769203

# 局部优化
opt2 = spo.fmin(fo, (2.0, 2.0), maxiter=250)
print(opt2)                     # [4.2710728  4.27106945]
print(fm(opt2[0], opt2[1]))     # 0.0158257532746805
