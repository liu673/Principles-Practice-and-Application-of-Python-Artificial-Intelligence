# -*-coding:utf-8-*-

"""

"""
"""
目前常用的TensorFlow、PyTorch深度学习框架的底层大多基于cuDNN库。
CUDA将CPU和GPU（所谓的主机和设备）统一到异构计算系统中
CUDA编程框架可分为3个部分：编程接口（API）、运行时需要的RunTime库和设备驱动，
"""

"""
CUDA 调用API方式：驱动API、运行API
在CUDA的计算模型中，程序分为两部分：主机（Host）端和设备（Device）端。
主机端是在CPU上执行的程序部分，设备端是在GPU上执行的程序部分。内核（Kernel）是设备端程序的另外一种叫法。

一般情况下，CPU执行主机端的程序会准备好数据并将其复制进显卡内存中，然后设备端的程序由GPU执行完后，主机端程序会将生成的数据结果从显卡的内存中取回
"""

"""
CUDA-C包含3种类型的函数：
（1）主机函数：调用、执行都仅由主机端来完成
（2）内核函数：定义时必须要加上_global_限定符，它由主机端调用，设备端执行
（3）设备函数：定义时必须要加上_device_限定符，仅由设备端调用、执行。
一次内核调用将会在GPU上并行执行大量的线程。

CUDA计算结构中分为3个层次：块内本地内存、共享内存、全局内存
"""

# 线程层次结构
"""
GPU中要执行的线程，根据最有效的数据共享来创建块（Block），其类型有一维、二维或三维。

一个块中的所有线程都必须位于同一个处理器核心中
一个内核可由多个大小相同的线程块同时执行，因而线程总数应等于每个块的线程数乘以块的数量。
"""
# 存储器层次结构
"""
CUDA拥有多个独立的存储空间：全局存储器、本地存储器、共享存储器、常量存储器、纹理存储器和寄存器
"""
# 主机和设备
"""
CUDA假设主机和设备均维护自己的DRAM（主机存储器、设备存储器）
"""
#
import torch as t

print(t.cuda.is_available())

# import tensorflow.compat.v1 as tf
#
# sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# # 查看日志信息若包含gpu信息，就是使用了gpu)

import tensorflow as tf

print(tf.version.__name__)
# gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
# cpus = tf.config.experimental.list_physical_devices(device_type='CPU')
# print(gpus, cpus)

# print(len(tf.config.list_physical_devices('GPU')))
# print(tf.test.gpu_device_name())







