#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'Daniel'

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import matplotlib as mpl
mpl.use('TKAgg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 读入数据
df = pd.read_csv('dot.csv')
x_data = np.array(df[['x1', 'x2']])
t_data = np.array(df['y_c'])

x_train = np.vstack(x_data).reshape(-1, 2)
t_train = np.vstack(t_data).reshape(-1, 1)

Y_c = [['red' if y else 'blue'] for y in t_train]

# 转换数据类型
x_train = tf.cast(x_train, dtype=tf.float32)
t_train = tf.cast(t_train, dtype=tf.float32)

train_db = tf.data.Dataset.from_tensor_slices((x_train, t_train)).batch(30)

# 生成神经网络参数, 结构, n * 2, 2 * 11, 11 * 1, 1
w1 = tf.Variable(tf.random.normal([2,11], dtype=tf.float32))
b1 = tf.Variable(tf.constant(0.01, shape=[1,11]))

w2 = tf.Variable(tf.random.normal([11,1], dtype=tf.float32))
b2 = tf.Variable(tf.constant(0.01, shape=[1,1]))

lr = 0.005
epoch = 500

# 训练
for epoch in range(epoch):
    for step, (x_train, t_trian) in enumerate(train_db):
        with tf.GradientTape() as tape:
            h1 = tf.matmul(x_train, w1) + b1
            r1 = tf.nn.relu(h1)
            y = tf.matmul(r1, w2) + b2
            #y = tf.nn.softmax(h2)
            loss = tf.reduce_mean(tf.square(t_trian-y))

        vars = [w1, b1, w2, b2]
        grads = tape.gradient(loss, vars)
        w1.assign_sub(lr * grads[0])
        b1.assign_sub(lr * grads[1])
        w2.assign_sub(lr * grads[2])
        b2.assign_sub(lr * grads[3])

    if epoch%20==0:
        print('epoch:', epoch, 'loss:', float(loss))


# 预测
print('**********predict***********')
# xx 在-3,3之间, 步长为0.01, yy在-3,3之间, 步长为0.01
xx, yy = np.mgrid[-3:3:.1, -3:3:.1]
grid = np.c_[xx.ravel(), yy.ravel()]
grid = tf.cast(grid, tf.float32)

# 将坐标点投入神经网络, 进行预测, probs为输出
probs = []
for x_test in grid:
    h1 = tf.matmul([x_test], w1) + b1
    r1 = tf.nn.relu(h1)
    y = tf.matmul(r1, w2) + b2
    probs.append(y)

# 描数据点
x1 = x_data[:,0]
x2 = x_data[:,1]

plt.scatter(x1, x2, color=np.squeeze(Y_c))

plt.show()


