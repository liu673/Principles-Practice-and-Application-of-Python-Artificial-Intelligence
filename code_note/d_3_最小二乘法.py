# -*-coding:utf-8-*-
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from scipy.optimize import leastsq
from sklearn.linear_model import LinearRegression
from scipy import sparse
import numpy as np

"""
最小二乘法是一种数学优化技术，用来作为函数拟合或者求函数极值的方法。
主要思想是最小化误差二次方和寻找数据的最佳匹配函数，利用最小二乘法求解未知参数，使得理论值与观测值之差（即误差，或称为残差）的二次方和达到最小，
"""


# 拟合函数
def func(a, x):
    k, b = a
    return k * x + b


# 残差
def dist(a, x, y):
    return func(a, x) - y


# 设置标头属性 中文字体显示
font = FontProperties()

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.sans-serif'] = ['Droid Sans Fallback']  # 指定默认字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
plt.figure()
plt.title(u'女生的身高体重数据 ')
plt.xlabel(u'体重/kg')
plt.ylabel(u'身高/cm')
plt.axis([40, 80, 140, 200])
plt.grid(True)
x = np.array([48.0, 57.0, 50.0, 54.0, 64.0, 61.0, 43.0, 59.0])

y = np.array([165.0, 165.0, 157.0, 170.0, 175.0, 165.0, 155.0, 170.0])
plt.plot(x, y, 'k.')
param = [0, 0]
# SciPy子函数库optimize已经提供了实现最小二乘拟合算法的函数leastsq。下面的例子使用leastsq，实现最小二乘拟合。
var = leastsq(dist, param, args=(x, y))
k, b = var[0]
print(k, b)
plt.plot(x, k * x + b, 'o-')
plt.show()
