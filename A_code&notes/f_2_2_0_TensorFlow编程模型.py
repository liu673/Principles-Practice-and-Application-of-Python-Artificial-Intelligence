# -*-coding:utf-8-*-


"""
（1）边
TensorFlow数据流图的边有两种连接关系：数据依赖和控制依赖。
（2）节点
节点又称为算子，它代表一个操作，一般用来表示施加的数学运算，也可以表示数据输入的起点及输出的终点，或者是读取／写入持久变量的终点。
（3）会话
启动图的第一步是创建一个Session对象，会话（Session）提供在途中执行操作的一些方法。一般的模式是，建立会话，此时会生成一张空图，在会话中添加节点和边，形成一张图，然后执行。
（4）设备
设备是指一块可以用来运算并且拥有自己的地址空间的硬件，如GPU和CPU，TensorFlow为了实现分布式执行操作，充分利用资源，可以明确指定操作在哪个设备上执行。
（5）变量
变量是一种特殊的数据，它在图中有固定的位置，不像普通张量那样可以流动。
创建一个变量张量，使用tf.Variable（）构造函数
填充机制，tf.placeholder（）临时替代任意操作的张量
"""
# import tensorflow as tf
import tensorflow.compat.v1 as tf

tf.compat.v1.disable_eager_execution()

# 创建一个常量运算操作,产生一个1×2矩阵
matrix1 = tf.constant([[3., 3.]])
# 创建另外一个常量运算操作,产生一个2×1矩阵
matrix2 = tf.constant([[2.], [2.]])
# 创建一个矩阵乘法运算,把matrix1和matrix2作为输入
# 返回值result代表矩阵乘法的结果
result = tf.matmul(matrix1, matrix2)

print(result)
with tf.Session() as sess:
    with tf.device('gpu:0'):
        result = sess.run(result)
print(result)

# 创建一个变量,初始化为标量0
state = tf.Variable(0, name='counter')
print(state)

# 变量填充
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
output = tf.multiply(input2, input2)
with tf.Session() as sess:
    print(sess.run([output], feed_dict={input1: [7.], input2: [2.]}))
