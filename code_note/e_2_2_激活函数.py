# -*-coding:utf-8-*-
import numpy as np

"""
激活函数又叫激励函数，主要作用是对神经元所获得的输入进行非线性变换，以此反映神经元的非线性特性。
"""
"""
(1)线性激活函数               f(x) = kx + b
(2)符号激活函数               f(x) = 1,   x>=0
                                 = 0,   x<0
(3)Sigmoid激活函数           f(x) = 1 / (1 + e^(-x))
    S形函数，区间（-∞，+∞）映射到（0，1）的连续区间。
(4)双曲正切激活函数            f(x) = (e^x - e^(-x)) / (e^x + e^(-x))    
    区间（-∞，+∞）映射到（-1，1）的连续区间。
(5)高斯激活函数               f(x) = e^[-1/2 * ((x-c)/σ)^2]
(6)ReLU激活函数              f(x) = x, x>0
                                 =0, x<=0
     f(x) = max(0, x)
"""
"""
卷积神经网络中，选择ReLU函数的原因：
（1）与Sigmoid函数必须计算指数和导数比较，ReLU代价小，而速度更快
（2）对于梯度计算公式▽= σ`，其中 σ`是Sigmoid的导数，在使用BP算法求梯度下降的时候，每经过一层Sigmoid神经元，都要乘以 σ`，
但是σ` 的最大值为1/4，所以会导致梯度越来越小，这对于训练深层网络是一个大问题，
但是ReLU函数的导数为1，不会出现梯度下降，以及梯度消失问题，从而更易于训练深层网络
（3）有研究表明，人脑在工作时只有大概5%的神经元被激活，而Sigmoid函数大概有50%的神经元被激活，
而人工神经网络在理想状态时有15%～30%的激活率，所以ReLU函数在小于0的时候是完全不激活的，所以可以适应理想网络的激活率要求
"""


def sigmoid(z):
    return 1. / (1 + np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))


def ReLU(z):
    return (z * (z > 0))


def ReLU_prime(z):
    return 1 * (z >= 0)


def lReLU(z):
    """
    leaky ReLU
    """
    return np.maximum(z / 100, z)


def lReLU_prime(z):
    z = 1 * (z >= 0)
    z[z == 0] = 1 / 100
    return z

def tanh(z):
    return np.tanh(z)

def tanh_prime(z):
    return (1 - tanh(z) ** 2)




























