# -*-coding:utf-8-*-
import numpy as np
import tensorflow.compat.v1 as tf
# import tensorflow as tf
import matplotlib.pyplot as plt

num_points = 100
vectors_set = []
for i in range(num_points):
    x1 = np.random.normal(0., 0.55)
    y1 = x1 * 0.1 + 0.3 + np.random.normal(0.0, 0.03)
    vectors_set.append([x1, y1])

    x_data = [v[0] for v in vectors_set]
    y_data = [v[1] for v in vectors_set]
    # plt.scatter(x_data, y_data)
    # plt.show()

    w = tf.Variable(tf.random_uniform([1], -1., 1.), name='myw')
    b = tf.Variable(tf.zeros([1]), name='myb')

    y = w * x_data + b

    loss = tf.reduce_mean(tf.square(y - y_data, name='mysquare'), name='myloss')
    optimizer = tf.train.GradientDescentOptimizer(0.5)
    train = optimizer.minimize(loss, name='mytrain')

    with tf.compat.v1.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        print('W=', sess.run(w), 'b=', sess.run(b), sess.run(loss))
        for step in range(20):
            sess.run(train)
            print('w=', sess.run(w), 'b=', sess.run(b), sess.run(loss))
            # writer = tf.summary.create_file_writer('./mytemp', sess.graph)
            writer = tf.summary.File_Writer('./mytemp', sess.graph)
    plt.scatter(x_data, y_data, c='b')
    plt.plot(x_data, sess.run(w) * x_data + sess.run(b))
    plt.show()
"""
在cmd中通过“cd目录”切换到该目录下，输入dir命令显示该目录下刚才运行的日志文件，最后输入：
tensorboard--logdir=C:\Users\ybx\Desktop\mytmp
打开Chrome浏览器，在浏览器地址栏中输入，http://laptop11et0a5m：6006/#graphs&run=，就会展示刚才程序设计的神经网络的
图形显示，
"""





