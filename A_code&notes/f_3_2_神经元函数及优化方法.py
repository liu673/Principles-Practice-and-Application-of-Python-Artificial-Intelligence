# -*-coding:utf-8-*-
import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_eager_execution()

"""
激活函数运行时激活神经网络中某一部分神经元，将激活信息向后传入下一层的神经网络。神经网络之所以能解决非线性问题（如语
音、图像识别等），本质上就是激活函数加入了非线性因素，弥补了线性模型的表达力，把“激活的神经元的特征”通过函数保留并映射到下一层
常见的激活函数有
Sigmoid、Tanh、ReLU和softplus
"""

# sigmoid函数
"""
sigmoid函数的优点在于他的输出映射在（0,1）内，单调连续，非常适合用于作输出层，并且求导比较容易
缺点在于其软饱和性。容易造成梯度消失
软饱和性指激活函数 h（x）在取值趋于无穷大时，它的一阶导数趋于0。硬饱和是指当|x|＞c时，其中c为常数， f‘（x）=0），一旦落入
软饱和区， f'（x）就会变得接近于0，很容易产生梯度消失。梯度消失指在更新模型参数时采用链式求导法则反向求导，越往前梯度越
小。最终的结果是到达一定深度后梯度对于模型的更新就没有任何贡献了
"""
a = tf.constant([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])
# print(tf.sigmoid(a))
with tf.compat.v1.Session() as sess:
    # print(sess.run(a))
    print('sigmoid', sess.run(tf.sigmoid(a)))

# ReLu函数
"""
softplus函数可以看作ReLU函数的平滑版本。ReLU函数定义为 f（x）=max（x，0）。
softplus函数定义为 f（x）=log（1+exp（x））。
"""
a = tf.constant([-1.0, 2.0])
with tf.compat.v1.Session() as sess:
    b = tf.nn.relu(a)
    print('relu', sess.run(b))

# dropout函数
"""
一个神经元将以概率keep_prob决定是否被抑制。如果被抑制，则该神经元的输出为0；如果不被抑制，那么该神经元的输出值将被放大到原来的1/keep_prob倍。
在默认情况下，每个神经元是否被抑制是相互独立的。但是否被抑制也可以通过noise_shape来调节。当noise_shape[i]==shape（x）[i]时， x中的元素是相互独立的。
如果shape（x）=[k，l ，m ，n]， x 中的维度的顺序分别为批、行、列和通道；如果noise_shape=[ k，l ，m ，n]，那么每个批和通道相互独立，行跟列相互关联，也就是说，要么都是0，要么都还是原来的值。
"""
a = tf.constant([[-1.0, 2.0, 3.0, 4.0]])
with tf.compat.v1.Session()as sess:
    b = tf.nn.dropout(a, rate=1 - 0.5, noise_shape=[1, 4])
    # b = tf.nn.dropout(a, rate=1 - 0.5, seed=1)
    print('dropout', sess.run(b))
    b = tf.nn.dropout(a, rate=1 - 0.5, noise_shape=[1, 1], seed=1)
    print('dropout', sess.run(b))

# 卷积函数
"""
是在一批图像上扫描的二维过滤器
tf.nn.convolution(input ,filter,padding,strides=None , dilation_rate=None ,name=None ,data_format=None）
这个函数计算 N 维卷积的和。
tf.nn.conv2d(input ,filter,padding,strides=None , dilation_rate=None ,name=None ,data_format=None）
这个函数的作用是对一个四维的输入数据input和四维的卷积核filter进行操作，然后对输入数据进行一个二维的卷积操作，
最后得到卷积之后的结果，
（1）tf.nn.convolution
（2）tf.nn.conv2d
（3）tf.nn.depthwise_conv2d 
（4）tf.nn.separable_conv2d 
（5）tf.nn.atrous_conv2d
（6）tf.nn.conv2d_transpose
（7）tf.nn.conv1d
（8）tf.nn.conv3d
（9）tf.nn.conv3d_transpose
"""
"""# tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, data_format=None, name=None)
# 输入:
# input:一个tensor.数据类型必须是float32或者float64
# filter:一个tensor.数据类型必须与input相同
# strides:一个长度是4的一维整数类型数组,每一维度对应的是input中每一维的对应移动步数
# padding:一个字符串,取值为SA M E或VALID
# padding='SAME':仅适用于全尺寸操作,即输入数据维度和输出数据维度相同
# padding='VALID':适用于部分窗口,即输入数据维度和输出数据维度不同
# use_cudnn_on_gpu:一个可选布尔值,默认是True
# name:(可选)为该操作取一个名字
input_data = tf.Variable(np.random.rand(10, 9, 9, 3), dtype=np.float32)
filter_data = tf.Variable(np.random.rand(2, 2, 3, 2), dtype=np.float32)

y = tf.nn.conv2d(input_data, filter_data, strides=[1, 1, 1, 1], padding='SAME')
# print(np.random.rand(4, 3, 2, 2))  # 一个大矩阵里包含4个矩阵，然后每个矩阵又有三个矩阵，这三个矩阵分别是2行2列

# tf.nn.depthwise_conv2d(input ,filter, strides,padding,rate=None,name=None,data_format=None）
# 这个函数输入张量的数据维度是[batch，in_height，in_width，in_channels]，
# 卷积 核 的 维 度 是 [filter_height ， filter_width ， in_channels ，channel_multiplier]，
# 在通道in_channels上面的卷积深度是1，depthwise_conv2d函数将不同的卷积核独立地应用在in_channels的每个通道上（从通道1到通道channel_multiplier），
# 然后把所有的结果进行汇总。最后输出通道的总数是 in_channels×channel_multiplier。
input_data = tf.Variable(np.random.rand(10, 9, 9, 3), dtype=np.float32)
filter_data = tf.Variable(np.random.rand(2, 2, 3, 5), dtype=np.float32)

y = tf.nn.depthwise_conv2d(input_data, filter_data, strides=[1, 1, 1, 1], padding='SAME')

# tf.nn.separable_conv2d （ input ， depthwise_filter ，pointwise_filter ， strides ，padding， rate=None，name=None，data_format=None）
# 是利用几个分离的卷积核去做卷积。在这个API中 ， 将应用一个二维的卷积核 ，在每通道上 ，以深度channel_multiplier进行卷积

# tf.nn.separable_conv2d （ input ， depthwise_filter ，pointwise_filter, strides, padding, rate = None, name = None, data_format = None)
# 特殊参数:
# depthwise_filter:一个张量.数据维度是四维[filter_height, filter_width, in_channels,channel_multiplier].
# 其中, in_channels的卷积深度是1
# pointwise_filter:一个张量,数据维度是四维[1, 1, channel_multiplier?in_channels, out_channels].
# 其中, pointwise_filter是在depthwise_filter卷积之后的混合卷积
input_data = tf.Variable(np.random.rand(10, 9, 9, 3), dtype=np.float32)
depthwise_filter = tf.Variable(np.random.rand(2, 2, 3, 5), dtype=np.float32)
pointwise_filter = tf.Variable(np.random.rand(1, 1, 15, 20), dtype=np.float32)

y = tf.nn.separable_conv2d(input_data, depthwise_filter, pointwise_filter, strides=[1, 1, 1, 1], padding='SAME')
"""

# 池化函数
"""
池化函数（Pooling Function）一般跟在卷积函数的下一层，池化操作是利用一个矩阵窗口在张量上进行扫描，将每个矩阵窗口中的值通过取最大值或平均值来减少元素个数。每个池化
操作的矩阵窗口大小是由ksize指定的，并且根据步长strides决定移动步长
（1）tf.nn.avg_pool   平均池化
（2）tf.nn.max_pool   最大池化
（3）tf.nn.max_pool_with_argmax   计算池化区域中元素的最大值和该最大值所在的位置。
（4）tf.nn.avg_pool3d（）和tf.nn.max_pool3d（）
（5）tf.nn.fractional_avg_pool （） 和tf.nn.fractional_max_pool（）    池化后的图片大小可以成非整倍缩小
（6）tf.nn.pool
"""

# 分类函数
"""
（1）tf.nn.sigmoid_cross_entropy_with_logits(logits, targets, name=None)
（2）tf.nn.softmax
（3）tf.nn.log_softmax
（4）tf.nn.softmax_cross_entropy_with_logits
（5）tf.nn.sparse_softmax_cross_entropy_with_logits
"""

# 优化方法
"""
目前加速训练的优化方法基本基于梯度下降法，只是细节上有差异。梯度下降法是求函数极值的一种方法，学习到最后就是求损失函数的极值问题
（1）BGD
BGD的全称是Batch Gradient Descent，即批梯度下降法。
这种方法是利用现有参数对训练集中的每一个输入生成一个估计输出yi，然后与实际输出yi比较，统计所有误差，求平均值后得到平均误差，以此作为更新参数的依据
（2）SGD
SGD的全称是Stochastic Gradient Descent，即随机梯度下降法。 
因为这种方法的主要思想是将数据集拆分成一个个批次（batch），随机抽取一个批次来计算并更新参数。
（3）Momentum法
Momentum法模拟物理学中动量的概念，更新时会在一定程度上保留之前的更新方向，利用当前批次再微调本次的更新参数，因此引入了一个新的变量 （速度），作为前几次梯度的累加。
（4）Nesterov Momentum法
Momentum法首先计算一个梯度，然后在加速更新梯度的方向进行一个大的跳跃；
Nesterov法首先在原来加速的梯度方向进行一个大的跳跃，然后在该位置计算梯度值，最后用这个梯度值修正最终的更新方向
（5）Adagrad法
Adagrad法能够自适应地为各个参数分配不同的学习率，能够控制每个维度的梯度方向。
（6）Adadelta法
其学习率单调递减使在训练的后期学习率非常小，并且需要手动设置一个全局的初始学习率
（7）RMSProp法
RMSProp法与Momentum法类似，通过引入一个衰减系数，使每一回合都衰减一定比例，在实践中，对循环神经网络（RNN）效果很好
（8）Adam法
Adam法根据损失函数针对每个参数的梯度的一阶矩估计和二阶矩估计动态调整每个参数的学习率。矩估计就是利用样本来估计总体中相应的参数

"""

