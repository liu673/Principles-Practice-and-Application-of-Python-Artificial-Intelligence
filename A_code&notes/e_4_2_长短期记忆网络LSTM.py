# -*-coding:utf-8-*-
import random
import numpy as np
import math

"""
RNN失效场景
(1)当预测位置和相关信息之间的文本间隔不断增大时(长期依赖问题)
(2)在复杂语言场景中，有用信息的间隔有大有小、长短不一，

长短时记忆网络（Long Short-Term Memory，LSTM），本质上，LSTM是一种时间循环神经网络。
LSTM够解决循环神经网络RNN存在的长期依赖问题，还能够解决神经网络中常见的梯度爆炸或梯度消失等问题。
"""
"""
LSTM靠一些“门”的结构让信息有选择性地影响循环神经网络中每个时刻的状态。
所谓“门”结构就是一个使用Sigmoid神经网络和一个按位做乘法计算的操作，这两个操作合在一起就是一个“门”结构。

使用Sigmoid作为激活函数的全连接神经网络层会输出一个0～1之间的数值，描述当前输入有多少信息量可以通过这个结构。
当门打开时，如果Sigmoid神经网络层输出为1，则全部信息都可以通过。当门关上时，如果Sigmoid神经网络层输出为0，则任何信息都无法通过。

“遗忘门”和“输入门”可以使神经网络更有效地保存需长期记忆的信息。
遗忘门的作用是让循环神经网络“遗忘”之前没有用到的信息。“遗忘门”会根据当前输入Xt 和上一时刻输出 Ht-1决定哪一部分记忆需要被遗忘。
"""
"""
LSTM有4个网络层
"""


# 声明sigmoid函数
def sigmoid(x):
    return 1. / (1 + np.exp(-x))


# 生成随机矩阵，取值范围是[a, b), shape用args指定
def rand_arr(a, b, *args):
    np.random.seed(0)
    return np.random.rand(*args) * (b - a) + a


class LstmParam:
    # 用于传递相关参数
    def __init__(self, mem_cell_ct, x_dim):
        self.mem_cell_ct = mem_cell_ct  # lstm的神经元数目
        self.x_dim = x_dim  # x_dim是输入数据的维度
        concat_len = x_dim + mem_cell_ct  # mem_cell_ct与x_dim的长度和
        # 按照矩阵维度初始化，并把损失矩阵归零
        # 权重矩阵
        self.wg = rand_arr(-0.1, 0.1, mem_cell_ct, concat_len)  # wg是输入节点的权重矩阵(这里的g不要理解为gate，原始论文里有解释)
        self.wi = rand_arr(-0.1, 0.1, mem_cell_ct, concat_len)  # wi是输入门的权重矩阵
        self.wf = rand_arr(-0.1, 0.1, mem_cell_ct, concat_len)  # wf是忘记门的权重矩阵
        self.wo = rand_arr(-0.1, 0.1, mem_cell_ct, concat_len)  # wo是输出门的权重矩阵
        # 偏置项  bg、bi、bf、bo分别是输入节点、输入门、忘记门、输出门的偏置
        self.bg = rand_arr(-0.1, 0.1, mem_cell_ct)
        self.bi = rand_arr(-0.1, 0.1, mem_cell_ct)
        self.bf = rand_arr(-0.1, 0.1, mem_cell_ct)
        self.bo = rand_arr(-0.1, 0.1, mem_cell_ct)
        # 权重损失  wg_diff、wi_diff、wf_diff、wo_diff分别是输入节点、输入门、忘记门、输出门的权重损失
        self.wg_diff = np.zeros((mem_cell_ct, concat_len))
        self.wi_diff = np.zeros((mem_cell_ct, concat_len))
        self.wf_diff = np.zeros((mem_cell_ct, concat_len))
        self.wo_diff = np.zeros((mem_cell_ct, concat_len))
        # 偏置损失  bg_diff、bi_diff、bf_diff、bo_diff分别是输入节点、输入门、忘记门、输出门的偏置损失
        self.bg_diff = np.zeros(mem_cell_ct)
        self.bi_diff = np.zeros(mem_cell_ct)
        self.bf_diff = np.zeros(mem_cell_ct)
        self.bo_diff = np.zeros(mem_cell_ct)

    def apply_diff(self, lr=1):
        # 权重更新过程 先减掉损失，再把损失矩阵归零
        self.wg -= lr * self.wg_diff
        self.wi -= lr * self.wi_diff
        self.wf -= lr * self.wf_diff
        self.wo -= lr * self.wo_diff

        self.bg -= lr * self.bg_diff
        self.bi -= lr * self.bi_diff
        self.bf -= lr * self.bf_diff
        self.bo -= lr * self.bo_diff

        self.wg_diff = np.zeros_like(self.wg)
        self.wi_diff = np.zeros_like(self.wi)
        self.wf_diff = np.zeros_like(self.wf)
        self.wo_diff = np.zeros_like(self.wo)

        self.bg_diff = np.zeros_like(self.bg)
        self.bi_diff = np.zeros_like(self.bi)
        self.bf_diff = np.zeros_like(self.bf)
        self.bo_diff = np.zeros_like(self.bo)


class LstmState:
    # 用于存储LSTM神经元的状态，这里包括g、i、f、o、s、h，其中s是内部状态矩阵(可以理解为记忆)，h是隐藏层神经元的输出矩阵
    def __init__(self, mem_cell_ct, x_dim):
        self.g = np.zeros(mem_cell_ct)
        self.i = np.zeros(mem_cell_ct)
        self.f = np.zeros(mem_cell_ct)
        self.o = np.zeros(mem_cell_ct)
        self.s = np.zeros(mem_cell_ct)
        self.h = np.zeros(mem_cell_ct)

        self.bottom_diff_h = np.zeros_like(self.h)
        self.bottom_diff_s = np.zeros_like(self.s)
        # self.bottom_diff_x = np.zeros(x_dim)


class LstmNode:
    # 对应一个样本的输入
    def __init__(self, lstm_param, lstm_state):
        self.state = lstm_state
        self.param = lstm_param
        # self.x = None  # x就是输入样本的x
        self.xc = None  # xc是用hstack把x和递归的输入节点拼接出来的矩阵（hstack是横着拼成矩阵，vstack就是纵着拼成矩阵）

    def bottom_data_is(self, x, s_prev=None, h_prev=None):
        # 输入样本的过程，首先把x和先前的输入拼接成矩阵，然后用公式wx+b分别计算g、i、f、o的值的值，这里面的激活函数有tanh和sigmoid
        # 判断是不是网络中的第一个lstm节点
        if s_prev is None: s_prev = np.zeros_like(self.state.s)
        if h_prev is None: h_prev = np.zeros_like(self.state.h)
        # 保存数据以用于反向操作
        self.s_prev = s_prev
        self.h_prev = h_prev

        xc = np.hstack((x, h_prev))
        # C~t 就是 g
        self.state.g = np.tanh(np.dot(self.param.wg, xc) + self.param.bg)
        self.state.i = sigmoid(np.dot(self.param.wi, xc) + self.param.bi)  # 输入门
        self.state.f = sigmoid(np.dot(self.param.wf, xc) + self.param.bf)  # 遗忘门
        self.state.o = sigmoid(np.dot(self.param.wo, xc) + self.param.bo)  # 输出门
        # C 就是 s
        self.state.s = self.state.g * self.state.i + s_prev * self.state.f
        self.state.h = self.state.s * self.state.o
        # self.x = x  # x就是输入样本的x
        self.xc = xc  # xc是用hstack把x和递归的输入节点拼接出来的矩阵（hstack是横着拼成矩阵，vstack就是纵着拼成矩阵）

    def top_diff_is(self, top_diff_h, top_diff_s):
        # top表示神经网络层的输出, bottom表示神经网络层的输入
        # top_diff_h表示当前t时序的dL(t)/dh(t), top_diff_s表示t+1时序记忆单元的dL(t)/ds(t)
        # 前缀d表达的是误差L对某一项的导数(directive)
        ds = self.state.o * top_diff_h + top_diff_s  # ds一行是在根据上面的公式dL(t)/ds(t)计算当前t时序的dL(t)/ds(t)
        do = self.state.s * top_diff_h  # do一行是计算dL(t)/do(t)，因为h(t) = s(t) * o(t)，所以dh(t)/do(t) = s(t)，所以dL(t)/do(t) = (dL(t)/dh(t)) * (dh(t)/do(t)) = top_diff_h * s(t)
        di = self.state.g * ds  # di一行是计算dL(t)/di(t)，考虑到换个符号表示就是s(t) = f(t) * s(t-1) + i(t) * g(t)，所以dL(t)/di(t) = (dL(t)/ds(t)) * (ds(t)/di(t)) = ds * g(t)
        dg = self.state.i * ds  # dg一行是计算dL(t)/dg(t)，同上有dL(t)/dg(t) = (dL(t)/ds(t)) * (ds(t)/dg(t)) = ds * i(t)
        df = self.s_prev * ds  # df一行是计算dL(t)/df(t)，同上有dL(t)/df(t) = (dL(t)/ds(t)) * (ds(t)/df(t)) = ds * s(t-1)

        # sigmoid函数导数
        # 表示当i神经元的输入发生单位变化时输出值有多大变化，那么再乘以di就表示当i神经元的输入发生单位变化时误差L(t)发生多大变化，也就是dL(t)/d i_input(t)
        di_input = (1. - self.state.i) * self.state.i * di
        df_input = (1. - self.state.f) * self.state.f * df
        do_input = (1. - self.state.o) * self.state.o * do
        # tanh函数导数
        dg_input = (1. - self.state.g ** 2) * dg
        # w*_diff是权重矩阵的误差，用于更新
        self.param.wi_diff += np.outer(di_input, self.xc)
        self.param.wf_diff += np.outer(df_input, self.xc)
        self.param.wo_diff += np.outer(do_input, self.xc)
        self.param.wg_diff += np.outer(dg_input, self.xc)
        # b*_diff是偏置的误差，用于做更新
        self.param.bi_diff += di_input
        self.param.bf_diff += df_input
        self.param.bo_diff += do_input
        self.param.bg_diff += dg_input
        # 累加输入x的diff，因为x在四处起作用，所以四处的diff加和之后才算作x的diff
        dxc = np.zeros_like(self.xc)
        dxc += np.dot(self.param.wi.T, di_input)
        dxc += np.dot(self.param.wf.T, df_input)
        dxc += np.dot(self.param.wo.T, do_input)
        dxc += np.dot(self.param.wg.T, dg_input)

        # bottom_diff_s是在t-1时序上s的变化和t时序上s的变化时f倍的关系
        self.state.bottom_diff_s = ds * self.state.f
        # dxc是x和h横向合并出来的矩阵，所以分别取出两部分的diff信息就是bottom_diff_x和bottom_diff_h（这里面的bottom_diff_x代码里面没有真正作用）
        # self.state.bottom_diff_x = dxc[:self.param.x_dim]
        self.state.bottom_diff_h = dxc[self.param.x_dim:]


class LstmNetwork():
    def __init__(self, lstm_param):
        self.lstm_param = lstm_param
        self.lstm_node_list = []
        self.x_list = []  # 输入序列

    def y_list_is(self, y_list, loss_layer):
        """
        通过设置具有相应损失层的目标序列来更新差异。不会更新参数。要更新参数，请调用self.lstm_param.apply_diff（）
        """
        assert len(y_list) == len(self.x_list)
        idx = len(self.x_list) - 1
        loss = loss_layer.loss(self.lstm_node_list[idx].state.h, y_list[idx])
        diff_h = loss_layer.bottom_diff(self.lstm_node_list[idx].state.h, y_list[idx])
        # 这里s不影响h（t+1）引起的损失，因此我们将其设置为零
        diff_s = np.zeros(self.lstm_param.mem_cell_ct)
        self.lstm_node_list[idx].top_diff_is(diff_h, diff_s)
        idx -= 1
        # …以下节点也会从下一个节点获得diff，因此我们将diff添加到diff_h中，我们也使用diff_s沿着恒定错误反向传播错误
        while idx >= 0:
            loss += loss_layer.loss(self.lstm_node_list[idx].state.h, y_list[idx])
            # diff_h(表达的是预测结果误差发生单位变化时损失L是多少，也就相当于公式中的dL(t)/dh(t)的一个数值计算)
            # 由idx从T往前遍历到1，计算loss_layer.bottom_diff和下一个时序的bottom_diff_h的和作为diff_h(其中第一次遍历即T时不加bottom_diff_h和公式一样)
            diff_h = loss_layer.bottom_diff(self.lstm_node_list[idx].state.h, y_list[idx])
            diff_h += self.lstm_node_list[idx + 1].state.bottom_diff_h
            diff_s = self.lstm_node_list[idx + 1].state.bottom_diff_s
            self.lstm_node_list[idx].top_diff_is(diff_h, diff_s)
            idx -= 1

        return loss

    def bottom_diff(self, pred, label):
        # l(t) = f(h(t), y(t)) = ||h(t) - y(t)||^2的导数l'(t) = 2 * (h(t) - y(t))
        diff = np.zeros_like(pred)
        diff[0] = 2 * (pred[0] - label)
        return diff

    def x_list_clear(self):
        self.x_list = []

    def x_list_add(self, x):
        self.x_list.append(x)
        if len(self.x_list) > len(self.lstm_node_list):
            # 需要添加新的lstm节点，创建新的状态
            lstm_state = LstmState(self.lstm_param.mem_cell_ct, self.lstm_param.x_dim)
            self.lstm_node_list.append(LstmNode(self.lstm_param, lstm_state))

        idx = len(self.x_list) - 1
        if idx == 0:
            self.lstm_node_list[idx].bottom_data_is(x)
        else:
            s_prev = self.lstm_node_list[idx - 1].state.s
            h_prev = self.lstm_node_list[idx - 1].state.h
            self.lstm_node_list[idx].bottom_data_is(x, s_prev, h_prev)

