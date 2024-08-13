# -*-coding:utf-8-*-


import numpy as np
import tflearn
# Download the Titanic dataset
from tflearn.datasets import titanic

# titanic.download_dataset('titanic dataset.csv')
# Load CSV file,indicate that the first column represents labels
from tflearn.data_utils import load_csv

data, labels = load_csv('titanic dataset.csv', target_column=0, categorical_labels=True, n_classes=2)


# 预处理 数据
def preprocess(data, columns_to_ignore):
    # Sort by descending id and delete columns
    for id in sorted(columns_to_ignore, reverse=True):
        [r.pop(id) for r in data]
    for i in range(len(data)):
        # Converting 'sex' field to float (id is 1 after removing labels column)
        data[i][1] = 1. if data[i][1] == 'female' else 0.
    return np.array(data, dtype=np.float32)


# 忽略 'name'and 'ticket'columns (id 1 & 6 of data array)
to_ignore = [1, 6]
data = preprocess(data, to_ignore)
print(data)

# 建立深度神经网络
# 一个3层神经网络，需要规定输入数据的形态，每个样本有6个特征，按批次处理可以节省内存，数据输入形态是[None，6]，其中None代码不知道维度，因此可以改变批处理中被处理后的样本总数量。
net = tflearn.input_data(shape=[None, 6])
net = tflearn.fully_connected(net, 32)
net = tflearn.fully_connected(net, 32)
net = tflearn.fully_connected(net, 2, activation='softmax')
net = tflearn.regression(net)

# 训练
# 供DNN包装器自动执行神经网络分类任务
# 训练10次，神经网络训练10次会看到全部数据，每次批处理大小是16。
model = tflearn.DNN(net)
# Start training(apply gradient descent algorithm)

model.fit(data, labels, n_epoch=10, batch_size=16, show_metric=True)

# 验证
# 预测杰克和露丝的存活率
jack = [3, "Mr.Bernt", 'male', 0, 0, 0, 65306, 8.1125]
rose = [1, "Allen,Miss.Elisabeth Walton", 'female', 29, 0, 0, 24160, 211.3375]
# Preprocess data
jack, rose = preprocess([jack, rose], to_ignore)
# Predict surviving chances(class 1 results)
pred = model.predict([jack, rose])
print("Jack Surviving Rate:", pred[0][1])
print("Rose Surviving Rate:", pred[1][1])
