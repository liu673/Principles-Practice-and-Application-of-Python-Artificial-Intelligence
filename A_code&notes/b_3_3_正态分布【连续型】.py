# -*-coding:utf-8-*-
import math

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

"""
正态分布（Normal Distribution），也称常态分布，又名高斯分布（Gaussian Distribution），
最早由A.棣莫弗在求二项分布的渐近公式中得到。C.F.高斯在研究测量误差时从另一个角度导出了它。
P.S.拉普拉斯和高斯研究了它的性质。正态分布是一个在数学、物理及工程等领域都非常重要的概率分布，
在统计学的许多方面有着重大的影响力。正态曲线呈钟形，两头低，中间高，左右对称因其曲线呈钟形，
因此人们又经常称其为钟形曲线

"""

"""
若随机变量X 服从一个数学期望为μ 、方差为 σ^2的正态分布，记为 N（ μ， σ^2）。
其概率密度函数为正态分布的期望值 μ 决定了其位置，其标准差 σ决定了分布的幅度。
当 μ =0， σ=1时的正态分布是标准正态分布
概率密度函数为 f(x) = (1 / σ * sqrt(2π)) * e ^ (-(x-μ)^2 / (2σ^2)) 
"""


"""
正态分布中的两个参数含义如下：
当固定 σ，改变 μ 的大小时， f（x）图形的形状不变，只是沿着轴作平移变换，因此 μ 被称为位置参数（决定对称轴的位置）；
当固定 μ，改变 σ 的大小时， f（x）图形的对称轴不变，形状改变，σ越小，图形尖峰越陡峭。 σ越大，图形越平坦，因此 σ被称为尺度参数，决定曲线的分散程度。
"""

u = 0  # 均值μ
sig = math.sqrt(0.2)  # 标准差σ

x = np.linspace(u - 3 * sig, u + 3 * sig, 50)
y_sig = np.exp(-(x - u) ** 2 / (2 * sig ** 2)) / (math.sqrt(2 * math.pi) * sig)
print(x)
print('=' * 20)
print(y_sig)
plt.plot(x, y_sig, 'r-', linewidth=2)
plt.grid(True)
plt.show()
