# -*-coding:utf-8-*-
import random
import matplotlib.pyplot as plt


"""
大数定律是一种描述当试验次数很大时所呈现的概率性质的定律，它由概率统计定义“频率收敛于概率”引申而来
统计学中常采用大数定律用样本均值来估计总体的期望。
"""

def flip_plot(minExp, maxExp):
    """
    模拟抛硬币
    (2 ** maxExp - 2 ** minExp)的批次实验，每批次重复抛硬币2 ** n次
    随着实验次数的增加，硬币正反面出现次数之比越来越接近于1
    :param minExp: 抛硬币次数为2的minExp次方
    :param maxExp: 抛硬币次数为2的maxExp次方
    :return:
    """

    ratios = []
    xAxis = []
    for exp in range(minExp, maxExp + 1):
        xAxis.append(2 ** exp)
    for numFlips in xAxis:
        numHeads = 0                                # 初始化，硬币正面朝上的计数为0
        for n in range(numFlips):
            if random.random() < 0.5:               # random.random()从[0, 1]随机的取出数
                numHeads += 1                       # 当随机取出的数小于0.5，正面朝上的计数+1
        numTails = numFlips - numHeads              # 得到本次实验中反面朝上的次数
        ratios.append(numHeads / float(numTails))   # 正方面计数的比值

    plt.title('Heads/Tails Ratios')
    plt.xlabel('Number of Flips')
    plt.ylabel('Heads/Tails')
    plt.plot(xAxis, ratios)
    plt.hlines(1, 0, xAxis[-1], linestyles='dashed', colors='r')
    plt.show()


flip_plot(4, 16)
















