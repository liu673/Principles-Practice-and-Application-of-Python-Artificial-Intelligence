# -*-coding:utf-8-*-


"""
基于TensorFlow构建seq2seq模型，并加入Attention机制，encoder和decoder为3层的RNN。

（1）清洗数据、ᨀ取ask数据和answer数据、ᨀ取词典、为每个字生成唯一的数字ID、ask和answer用数字ID表示。
（2）TensorFlow中Translate Demo，由于出现deepcopy错误，这里对seq2seq进行了稍微改动。
（3）训练seq2seq模型。
（4）进行聊天。
"""

# 清洗数据：generate_chat.py
"""
原始数据中，每次对话是M开头，前一行是E，并且每次对话都是一问一答的形式。将原始数据分为ask、answer两份数据。
两种词袋：“汉字=＞数字”“数字=＞汉字”，根据第一个词袋将ask、answer数据转化为数字表示
answer数据每句末尾添加EOS作为结束符号。
"""
# from i_7_1_聊天机器人_seq2seq.generate_chat import *

# 模型学习：seq2seq.py,seq2seq_model.py

# 训练模型：train_model.py

# 聊天测试：predict_chat.py

# from i_7_1_聊天机器人_seq2seq import predict_chat
















