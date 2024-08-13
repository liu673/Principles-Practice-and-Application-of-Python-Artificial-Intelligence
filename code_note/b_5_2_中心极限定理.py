# -*-coding:utf-8-*-
import numpy as np
import matplotlib.pyplot as plt
"""
大数定律揭示了大量随机变量的平均结果，但没有涉及随机变量的分布问题。
而中心极限定理说明的是在一定条件下，大量独立随机变量的平均数是以正态分布为极限的。

中心极限定理指出大量的独立随机变量之和具有近似于正态的分布。
因此，它不仅提供了计算独立随机变量之和的近似概率的简单方法，而且有助于解释为什么有很多自然群体的经验频率呈现出钟形（即正态）曲线这一事实，
因此中心极限定理这个结论使正态分布在数理统计中具有很重要的地位，也使正态分布有了广泛的应用。

中心极限定理有辛钦中心极限定理、德莫佛-拉普拉斯中心极限定理、李亚普洛夫中心极限定理、林德贝尔格定理等表现形式
"""

random_data = np.random.randint(1, 7, 10000)
print(random_data.mean())  # 平均值 3.4964
print(random_data.std())  # 标准差 1.7109024051651807
"""
平均值接近3.5很好理解，因为每次掷出来的结果是1、2、3、4、5、6。每个结果的概率是1/6，所以加权平均值就是3.5。
"""

# 10个样本数 样本均值（4.1）离总体均值（3.5）有所偏差
sample1 = []
for i in range(0, 10):
    sample1.append(random_data[int(np.random.random() * len(random_data))])
print(sample1)
print(np.mean(sample1))  # 均值    4.1
print(np.std(sample1))  # 标准差  1.7

# 抽取1000组，每组50个样本，并且把每组的平均值都算出来。
samples = []
samples_mean = []
samples_std = []

for i in range(0, 1000):
    sample = []
    for j in range(0, 50):
        sample.append(random_data[int(np.random.random() * len(random_data))])
    sample_np = np.array(sample)
    samples_mean.append(sample_np.mean())
    samples_std.append(sample_np.std())
    samples.append(sample_np)


samples_mean_np = np.array(samples_mean)
samples_std_np = np.array(samples_std)

print(samples_mean_np.mean())       # 3.50728
print(samples_std_np.std())         # 0.10153241608282683

# plt.hist(samples_mean_np, bins=10, histtype='stepfilled')
plt.hist(samples_mean_np, bins=10, edgecolor='k')
plt.show()

