# -*-coding:utf-8-*-
import numpy as np

"""
卷积神经网络每层之间的节点没有连接。

循环神经网络（RNN）用来处理和预测序列数据
RNN隐藏层之间的节点是有连接的，隐藏层的输入不仅包括输入层的输出，还包括上一时刻隐藏层的输出。
"""
"""
卷积神经网络在不同的空间位置共享参数，
循环神经网络是在不同时间位置共享参数，从而能够使用有限的参数处理任意长度的序列。
"""
"""
RNN可以看作同一神经网络结构（输入层→隐藏层→输出层）在时间序列上被复制多次的结果，这个被复制多次的结构被称为循环体。
循环体中的神经网络的输入有两部分，一部分为上一时刻的状态，另一部分为当前时刻的输入样本。

RNN中的状态是通过一个向量来表示的，这个向量的维度也称为RNN隐藏层的大小，假设其为 n。
假设输入向量的维度为x ，隐藏状态的维度为 n，全连接层神经网络的输入大小为 n+x。
将上一时刻的状态与当前时刻的输入拼接成一个大的向量作为循环体中神经网络的输入。
因为该全连接层的输出为当前时刻的状态，于是输出层的节点个数也为n ，循环体中的参数个数为（n+x）× n+n个。
"""

"""
RNN与其他网络唯一的区别在于它每个时刻都有一个输出，所以RNN的总损失为所有时刻或者部分时刻上的损失函数的总和
"""

X = [1, 2]
state = [0.0, 0.0]
# 分开定义不同输入部分的权重以方便操作
w_cell_state = np.asarray([[0.1, 0.2], [0.3, 0.4]])
w_cell_input = np.asarray([0.5, 0.6])
b_cell = np.asarray([0.1, -0.1])

# 定义用于输出的全连接层参数
w_output = np.asarray([[1.0], [2.0]])
b_output = 0.1

# 执行前向传播过程
for i in range(len(X)):
    before_activation = np.dot(state, w_cell_state) + X[i] * w_cell_input + b_cell
    state = np.tanh(before_activation)
    final_output = np.dot(state, w_output) + b_output
    print('before activation:', before_activation)
    print('state:', state)
    print('output:', final_output)




# print(w_cell_state)
# print(w_cell_state.shape)
# print(state)
# print(np.dot(state, w_cell_state))
# state = np.asarray([0.0, 0.0])
# print(state.shape)
# print(np.dot(state, w_cell_state))
# print(np.dot(state, w_cell_state).shape)

# import numpy as np
# np.random.seed(4)
# a = np.random.randint(0, 5, size=(2, 2))
# b = np.array([1, 2])
# print(a)
# print(b)
# print("the shape of a is " + str(a.shape))
# print("the shape of b is " + str(b.shape))
# print(np.dot(a, b))
# print(np.dot(a, b).shape)
# print(np.dot(b, a))
# print(np.dot(b, a).shape)


