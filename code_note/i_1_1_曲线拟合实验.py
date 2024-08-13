# -*-coding:utf-8-*-


"""

"""
"""
TensorFlow进行深度学习模型训练流程图主要分为5个步骤：
（1）对模型参数进行初始化，通常采用对参数随机赋值的方法。对于比较简单的模型可以将各参数的初值均设为0，然后读取已经分配好的训练数据inputs（），包括每个数据样本及其期望输出。
（2）在训练数据上执行推断模型，在当前模型参数配置下，每个训练样本都会得到一个输出值，然后计算损失loss（ X,Y ），依据训练数据 X 及期望输出 Y 计算损失
（3）不断调整模型参数train（total_loss）。在给定损失函数的约束下，通过大量训练步骤改善各参数的值，从而将损失最小化。本书选用TensorFlowᨀ供的梯度下降算法tf.gradients进行学习。
tf.gradients通过符号计算推导出指定的流图步骤的梯度，并将其以张量形式输出，由于TensorFlow已经实现了大量优化方法，因此不需要手工调用这个梯度计算函数，只需通过大量循环不断重复上述过程即可
（4）当训练结束后便进入模型评估阶段evaluate（sess， X，Y）。在这一阶段中，需要对一个同样含有期望输出信息的不同测试集
依据模型进行推断，并评估模型在该数据集上的损失。由于测试集拥有与训练集完全不同的样本，通过评估可以了解所训练的模型在训练集之外的识别能力
（5）当对模型的准确率满意后，例如达到预设的98%以上的准确率，就可以直接导出模型了。
"""

# 绘制标准的sin曲线
# 导入相应的Python包和模块
# import tensorflow.compat.v1 as tf
import tensorflow as tf
import math
import numpy as np
import matplotlib.pyplot as plt
import pylab
import warnings

warnings.filterwarnings('ignore')


# tf.compat.v1.disable_eager_execution()
# tf.disable_v2_behavior()


# 定义draw_sin_line()函数,该函数用来绘制标准的sin曲线
def draw_sin_line():
    # 绘制标准的sin曲线
    x = np.arange(0, 2 * np.pi, 0.01)
    x = x.reshape((len(x), 1))
    y = np.sin(x)

    # 解决中文显示问题
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    pylab.plot(x, y, label='标准的sin曲线')
    plt.axhline(linewidth=1, color='r')
    plt.axvline(x=np.pi, linestyle='--', linewidth=1, color='g')
    # plt.show()


# draw_sin_line()
# 创建训练样本
def get_train_data():
    """返回一个训练样本(train_x,train_y)其中train_x是随机的自变量,train_y是train_x的sin函数值"""
    train_x = np.random.uniform(0.0, 2 * np.pi, (1))
    train_y = np.sin(train_x)
    return train_x, train_y


# 定义推理函数inference
# 构建了3个隐藏层，每个隐藏层包含16个节点，连接节点的参数weight和bias采用初始化均值为0、方差为1的随机初始化，每个隐藏层的单位采用tf.sigmoid（）作为激活函数，输出层中没有增加
# sigmoid（）函数，这是因为前面的几层非线性变换已经提取了足够充分的特征，使用这些特征已经可以让模型用最后一个线性分类函数来分类。
def inference_data(input_data):
    """
    定义前向计算的网络结构,args:输人的值,单个值
    初始器 initializer
    """
    # with tf.variable_creator_scope('hidden1'):
    with tf.compat.v1.variable_scope('hidden1'):
        # 第1个隐藏层，采用16个隐藏节点
        weights = tf.compat.v1.get_variable("weight", [1, 16], tf.float32,
                                            initializer=tf.random_normal_initializer(0.0, 1))
        biases = tf.compat.v1.get_variable("bias", [1, 16], tf.float32,
                                           initializer=tf.random_normal_initializer(0.0, 1))
        hidden1 = tf.sigmoid(tf.multiply(input_data, weights) + biases)
    # with tf.variable_creator_scope("hidden2"):
    with tf.compat.v1.variable_scope('hidden2'):
        # 第2个隐藏层,采用16个隐藏节点
        weights = tf.compat.v1.get_variable("weight", [16, 16], tf.float32,
                                            initializer=tf.random_normal_initializer(0.0, 1))
        biases = tf.compat.v1.get_variable("bias", [16], tf.float32, initializer=tf.random_normal_initializer(0.0, 1))
        mul = tf.matmul(hidden1, weights)
        hidden2 = tf.sigmoid(mul + biases)
    # with tf.variable_creator_scope("hidden3"):
    with tf.compat.v1.variable_scope('hidden3'):
        # 第3个隐藏层,采用16个隐藏节点
        weights = tf.compat.v1.get_variable("weight", [16, 16], tf.float32,
                                            initializer=tf.random_normal_initializer(0.0, 1))
        biases = tf.compat.v1.get_variable("bias", [16], tf.float32, initializer=tf.random_normal_initializer(0.0, 1))
        mul = tf.matmul(hidden2, weights)
        hidden3 = tf.sigmoid(mul + biases)
    with tf.compat.v1.variable_scope('output_layer'):
        # 输出层
        weights = tf.compat.v1.get_variable("weight", [16, 1], tf.float32,
                                            initializer=tf.random_normal_initializer(0.0, 1))
        biases = tf.compat.v1.get_variable("bias", [1], tf.float32, initializer=tf.random_normal_initializer(0.0, 1))
        output = tf.matmul(hidden3, weights) + biases
        return output


# 定义训练函数
# 用TensorFlow实现神经网络时，需要定义网络结构、参数、数据的输入和输出、采用损失函数和优化方法。
# 通过梯度下降法将损失最小化
def train_data():
    # 学习率
    learning_rate = 0.01
    # x = tf.placeholder(tf.float32)
    # y = tf.placeholder(tf.float32)
    x = tf.compat.v1.placeholder(tf.float32)
    y = tf.compat.v1.placeholder(tf.float32)
    # 基于训练好的模型推理,获取推理结果
    net_out = inference_data(x)
    # 定义损失函数的op
    loss_op = tf.square(net_out - y)
    # 采用随机梯度下降的优化函数
    opt = tf.train.GradientDescentOptimizer(learning_rate)
    # 定义训练操作
    train_op = opt.minimize(loss_op)
    # 变量初始化
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        # 执行变量的初始化操作
        sess.run(init)
        print("开始训练...")
        for i in range(100001):
            # 获取训练数据
            train_x, train_y = get_train_data()
            sess.run(train_op, feed_dict={x: train_x, y: train_y})
            # 定时输出当前的状态
            if i % 10000 == 0:
                times = int(i / 10000)
                # 每执行 10000 次训练后,试一下结果,测试结果用 pylab.plot()函数在界面上绘制出来
                test_x_ndarray = np.arange(0, 2 * np.pi, 0.01)
                test_y_ndarray = np.zeros([len(test_x_ndarray)])
                ind = 0
                for test_x in test_x_ndarray:
                    test_y = sess.run(net_out, feed_dict={x: test_x, y: 1})
                    # 对数组中指定的索引值指向的元素替换成指定的值
                    np.put(test_y_ndarray, ind, test_y)
                    ind += 1
                # 先绘制标准正弦函数的曲线,再用虚线绘制出模拟正弦函数的曲线
                draw_sin_line()
                pylab.plot(test_x_ndarray, test_y_ndarray, '--', label=str(times) + ' times')
                pylab.legend(loc='upper right')
                pylab.show()
                print("=== DONE===")


train_data()

