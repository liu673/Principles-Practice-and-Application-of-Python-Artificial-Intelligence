# -*-coding:utf-8-*-
import time

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

"""
（1）tensorflow 同步预测：同步预测是指使用当前时刻的500只个股股价，预测当前时刻的大盘指数，即一个回归问题，输入共500维特征，输出一维，
    即[None，500]=＞[None，1]。
    使用TensorFlow实现同步预测，主要用到多层感知机（Multi-Layer Perceptron，MLP），损失函数用均方误差（Mean Square Error，MSE）。
（2）Keras 同步预测
（3）异步预测：异步预测是指使用历史若干时刻的大盘指数，预测当前时刻的大盘指数，这样才更加符合预测的定义。
    使用Keras实现异步预测，主要用到循环神经网络即RNN（Recurrent Neural Network）中的LSTM（ Long Short-Term Memory）

"""
data = pd.read_csv('data_stocks.csv')
# print(data.describe())
# print(data.info())
# print(data.head())

# 查看时间跨度
# print(time.strftime('%Y-%m-%d', time.localtime(data['DATE'].max())),
#       time.strftime('%Y-%m-%d', time.localtime(data['DATE'].min())))

# 绘制大盘趋势折线图
plt.plot(data['SP500'])
# plt.show()

# 去掉DATE一列，训练集与测试集分割
data.drop('DATE', axis=1, inplace=True)
data_train = data.iloc[:int(data.shape[0] * 0.8), :]
data_test = data.iloc[int(data.shape[0] * 0.8):, :]
# print(data_train.shape, data_test.shape)

# 数据归一化，使用data_trian进行拟合fit
scaler = MinMaxScaler(feature_range=(-1, 1))
scaler.fit(data_train)
data_train = scaler.transform(data_train)
data_test = scaler.transform(data_test)

# 同步预测
"""
同步预测是指使用当前时刻的500只个股股价，预测当前时刻的大盘指数，即一个回归问题，输入共500维特征，输出一维，即[None，500]=＞[None，1]。
使用TensorFlow实现同步预测，主要用到多层感知机（Multi-Layer Perceptron，MLP），损失函数用均方误差（Mean,Square Error，MSE）。
"""
"""
X_train = data_train[:, 1:]
Y_train = data_train[:, 0]
X_test = data_test[:, 1:]
Y_test = data_test[:, 0]
input_dim = X_train.shape[1]
hidden_1 = 1024
hidden_2 = 512
hidden_3 = 256
hidden_4 = 128
output_dim = 1
batch_size = 256
epochs = 10
tf.reset_default_graph()
X = tf.placeholder(shape=[None, input_dim], dtype=tf.float32)
Y = tf.placeholder(shape=[None], dtype=tf.float32)

W1 = tf.get_variable('W1', [input_dim, hidden_1], initializer=tf.contrib.layers.xavier_initializer(seed=1))
b1 = tf.get_variable('b1', [hidden_1], initializer=tf.zeros_initializer())
W2 = tf.get_variable('W2', [hidden_1, hidden_2], initializer=tf.contrib.layers.xavier_initializer(seed=1))
b2 = tf.get_variable('b2', [hidden_2], initializer=tf.zeros_initializer())
W3 = tf.get_variable('W3', [hidden_2, hidden_3], initializer=tf.contrib.layers.xavier_initializer(seed=1))
b3 = tf.get_variable('b3', [hidden_3], initializer=tf.zeros_initializer())
W4 = tf.get_variable('W4', [hidden_3, hidden_4], initializer=tf.contrib.layers.xavier_initializer(seed=1))
b4 = tf.get_variable('b4', [hidden_4], initializer=tf.zeros_initializer())
W5 = tf.get_variable('W5', [hidden_4, output_dim], initializer=tf.contrib.layers.xavier_initializer(seed=1))
b5 = tf.get_variable('b5', [output_dim], initializer=tf.zeros_initializer())

hl = tf.nn.relu(tf.add(tf.matmul(X, W1), b1))
h2 = tf.nn.relu(tf.add(tf.matmul(hl, W2), b2))
h3 = tf.nn.relu(tf.add(tf.matmul(h2, W3), b3))
h4 = tf.nn.relu(tf.add(tf.matmul(h3, W4), b4))
out = tf.transpose(tf.add(tf.matmul(h4, W5), b5))
cost = tf.reduce_mean(tf.squared_difference(out, Y))
optimizer = tf.train.AdamOptimizer().minimize(cost)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for e in range(epochs):
        shuffle_indices = np.random.permutation(np.arange(Y_train.shape[0]))
        X_train = X_train[shuffle_indices]
        Y_train = Y_train[shuffle_indices]
        for i in range(Y_train.shape[0] // batch_size):
            start = i * batch_size
            batch_x = X_train[start: start + batch_size]
            batch_y = Y_train[start: start + batch_size]
            sess.run(optimizer, feed_dict={X: batch_x, Y: batch_y})
            if i % 50 == 0:
                print('MSE Train:', sess.run(cost, feed_dict={X: X_train, Y: Y_train}))
                print('MSE Test:', sess.run(cost, feed_dict={X: X_test, Y: Y_test}))
                y_pred = sess.run(out, feed_dict={X: X_test})
                y_pred = np.squeeze(y_pred)
                plt.plot(Y_test, label='test')
                plt.plot(y_pred, label='pred')
                plt.title('Epoch' + str(e) + ', Batch' + str(i))
                plt.legend()
                plt.show()
"""

# 使用Keras进行同步预测
"""
from keras.layers import Input, Dense
from keras.models import Model

X_train = data_train[:, 1:]
Y_train = data_train[:, 0]
X_test = data_test[:, 1:]
Y_test = data_test[:, 0]
input_dim = X_train.shape[1]
hidden_1 = 1024
hidden_2 = 512
jidden_3 = 256
hidden_4 = 128
output_dim = 1
batch_size = 256
epochs = 10
X = Input(shape=[input_dim, ])
h = Dense(hidden_1, activation='relu')(X)
h = Dense(hidden_2, activation='relu')(h)
h = Dense(hidden_3, activation='relu')(h)
h = Dense(hidden_4, activation='relu')(h)
Y = Dense(output_dim, activation='sigmoid')(h)

model = Model(X, Y)
model.compile(loss='mean squared error', optimizer='adam')
model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, shuffle=False)
y_pred = model.predict(X_test)
print('MSE Train:', model.evaluate(X_train, Y_train, batch_size=batch_size))
print('MSE Test:', model.evaluate(X_test, Y_test, batch_size=batch_size))
plt.plot(Y_test, label='test')
plt.plot(y_pred, label='pred')
plt.legend()
plt.show()
"""

# 异步预测

from keras.layers import Input, Dense, LSTM
from keras.models import Model

output_dim = 1
batch_size = 256
epochs = 10
seq_len = 5
hidden_size = 128

X_train = np.array([data_train[i: i + seq_len, 0] for i in range(data_train.shape[0] - seq_len)])[:, :, np.newaxis]
Y_train = np.array([data_train[i + seq_len, 0] for i in range(data_train.shape[0] - seq_len)])
X_test = np.array([data_test[i: i + seq_len, 0] for i in range(data_test.shape[0] - seq_len)])[:, :, np.newaxis]
Y_test = np.array([data_test[i + seq_len, 0] for i in range(data_test.shape[0] - seq_len)])
print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)

X = Input(shape=[X_train.shape[1], X_train.shape[2], ])
h = LSTM(hidden_size, activation='relu')(X)
Y = Dense(output_dim, activation='sigmoid')(h)
model = Model(X, Y)
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, shuffle=False)
Y_pred = model.predict(X_test)
print('MSE Train:', model.evaluate(X_train, Y_train, batch_size=batch_size))

print('MSE Test:', model.evaluate(X_test, Y_test, batch_size=batch_size))
plt.plot(Y_test, label='test')
plt.plot(Y_pred, label='pred')
plt.legend()
plt.show()
