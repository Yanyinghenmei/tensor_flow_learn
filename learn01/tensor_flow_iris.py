#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'Daniel'

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
from sklearn.datasets import load_iris

import matplotlib as mpl
mpl.use('TKAgg')
import matplotlib.pyplot as plt


'''数据集导入'''
x_data = load_iris().data
t_data = load_iris().target

'''数据集乱序'''
np.random.seed(116) # 使用 相同的种子, 使乱序后数据与标签一一对应
np.random.shuffle(x_data)
np.random.seed(116)
np.random.shuffle(t_data)
# tf.random.set_seed(116)

'''数据集分为训练数据+测试数据'''
x_train = x_data[:-30]
t_train = t_data[:-30]
x_test = x_data[-30:]
t_test = t_data[-30:]

'''数据类型转换'''
x_train = tf.cast(x_train, tf.float32)
x_test = tf.cast(x_test, tf.float32)

'''配对[输入特征, 标签], 并分批'''
train_db = tf.data.Dataset.from_tensor_slices((x_train, t_train)).batch(30)
test_db = tf.data.Dataset.from_tensor_slices((x_test, t_test)).batch(30)

'''定义神经网络的可训练参数'''
# 使用seed使每次生成的随机数相同（使大家结果都一致，在现实使用时不写seed）
w1 = tf.Variable(tf.random.truncated_normal([4,3], stddev=0.1, seed=1))
b1 = tf.Variable(tf.random.truncated_normal([3], stddev=0.1, seed=1))


'''初始化超参数'''
lr = 0.1
train_loss_results = []
test_acc = []
epoch = 100
loss_all = 0

'''嵌套循环迭代, with结构更新参数, 显示loss'''
for e in range(epoch):
    # 每次取出一个batch的数据
    for step, (x_train, t_train) in enumerate(train_db):
        with tf.GradientTape() as tape: # whti结构记录梯度信息
            y = tf.matmul(x_train, w1) + b1
            y = tf.nn.softmax(y)
            t = tf.one_hot(t_train, depth=3)
            loss = tf.reduce_mean(tf.square(t-y))  # 均方差
            loss_all += loss.numpy()  # loss累加

        # 计算梯度
        grads = tape.gradient(loss, [w1, b1])

        # 更新参数
        w1.assign_sub(lr * grads[0])
        b1.assign_sub(lr * grads[1])

    '''打印loss'''
    # 每个epoth, 循环四个batch
    print('Epoch: {}, loss: {}', e, loss_all/4)
    train_loss_results.append(loss_all/4)
    loss_all = 0

    '''测试'''
    total_correct, total_num = 0, 0     # 正确次数/总次数
    for x_test, t_test in test_db:
        y = tf.matmul(x_test, w1) + b1
        y = tf.nn.softmax(y)
        pre = np.argmax(y, axis=1)

        # 统一类型
        pre = tf.cast(pre, dtype=t_test.dtype)
        res = tf.cast(tf.equal(pre, t_test), dtype=tf.int32)
        correct = tf.reduce_sum(res)
        total_correct += correct
        total_num += res.shape[0]

    acc = total_correct / total_num
    test_acc.append(acc)
    print('Test_acc: ', acc)
    print('----------------------------------')


plt.title('Loss & Accuracy Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss & Acc')

'''绘制loss曲线'''
plt.plot(train_loss_results, label="$Loss$")

'''绘制Accuracy曲线'''
plt.plot(test_acc, label='$Accuracy$')


plt.legend() # 绘制标签
plt.show() # 绘制图像