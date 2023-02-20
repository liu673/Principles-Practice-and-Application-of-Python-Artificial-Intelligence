# -*-coding:utf-8-*-

"""
TensorFlow作为分布式机器学习平台
远程过程调用（Remote Procedure Call，RPC）和远程直接数据存取（Remote Direct Memory Access，RDMA）为网络层，主要负责传递神经网络算法参数。
CPU、GPU等为设备层，主要负责神经网络算法中具体的运算操作。

"""

""""
TensorFlow的数据流图是由节点（Node）和边（Edge）组成的有向无环图。Tensor（张量）代表了数据流图中的边，Flow（流动）这个动作就代表了数据流图中节点所做的操作，
TensorFlow将程序分为两个独立的部分：
（1）定义并构建神经网络结构图，包括激活函数定义、损失函数定义、分类模型定义等；
（2）执行设计好的神经网络模型等，包括数据集输入、初始赋值及通过会话（Session）编译运行等。
由于神经网络结构图的定义和执行分开设计，所以TensorFlow能够多平台工作以并行执行
"""
# 传统统程序 设 计一 般采用 先赋 值 后运 行的 编程 方 式。TensorFlow先定义各种张量结构的变量，然后建立一个数据流图，
# 在数据流图中规定各个变量之间的计算关系，最后需要对数据流图进行编译，但此时的数据流图还是一个空壳，里面没有任何实际数据，只
# 有把需要计算的输入放进去后，才能在整个模型中形成数据流，从而形成输出值，
# 传统编程方式
t = 8 + 9
# 定义了t的运算,在运行时就执行了,并输出17
print(t)
# TensorFlow编程方式
# import tensorflow as tf
# 解决TensorFlow的2.0版本与1.0版本之间的冲突
import tensorflow.compat.v1 as tf

t = tf.add(8, 9)
# 输出Tensor(＂Add_1:0＂,shape=(),dtype=int32)
print(t)
# 数据流图中的节点,实际上对应的是TensorFlow API中的一个操作,并没有真正去运行


# TensorFlow中涉及的运算都要放在图中，而图的运行只发生在会话（Session）中，开启会话后，就可以用数据去填充节点，进行运算，关闭会话后就不能进行计算了。
# import tensorflow as tf
import tensorflow.compat.v1 as tf

tf.compat.v1.disable_eager_execution()

# 创建图
a = tf.constant([1.0, 2.0])
b = tf.constant([3.0, 4.0])
c = a * b
# 创建会话
sess = tf.Session()
# with tf.Session() as sess:
# 计算c
print(sess.run(c))
# sess.close()
