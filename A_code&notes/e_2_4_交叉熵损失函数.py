# -*-coding:utf-8-*-

"""
交叉熵（Binary Cross Entropy）作为损失函数，
设p表示真实标记的分布，q则为训练后的模型的预测标记分布，
交叉熵损失函数可以衡量p与q的相似性。交叉熵作为损失函数还有一个好处，
使用Sigmoid函数在梯度下降时能避免均方误差损失函数学习速率降低的问题，
因为学习速率可以被输出的误差所控制。
"""

# def binary_crossentropy(t,o):
#  return-(t?tf.log(o+eps)+(1.0-t)?tf.log(1.0-o+eps))






