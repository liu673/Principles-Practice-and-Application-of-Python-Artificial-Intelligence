# -*-coding:utf-8-*-

"""
TensorFlow的计算表现为数据流图，所以tf.Graph类中包含一系
列表示计算的操作对象（tf.Operation），以及在操作之间流动的数据——张量对象（tf.Tensor）。、

tf.Operation类代表图中的一个节点，用于计算张量数据。与操作相关的API均位于tf.Operation类中，

tf.Tensor类是操作输出的符号句柄，它不包含操作输出的值，而是提供了一种在tf.Session中计算这些值的方法。
与张量相关的API均位于tf.Tensor类中
"""

"""
TensorFlow有两个作用域：
一个是name_scope，
一个是variable_scope。
"""






