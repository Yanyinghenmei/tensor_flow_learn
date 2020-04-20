#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'Daniel'

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np

'''交叉熵'''
# loss1 = tf.losses.categorical_crossentropy([1,0], [0.3, 0.7])
# loss2 = tf.losses.categorical_crossentropy([1,0], [0.7, 0.3])
# print(loss1, '\n', loss2)

'''softmax+交叉熵'''
#loss = tf.nn.softmax_cross_entropy_with_logits(y_, y)
y_ = np.array([[1,0,0],[0,1,0]])
y = np.array([[12,3,2],[3,23,5]])
y_ = tf.cast(y_, tf.float32)
y = tf.cast(y, tf.float32)
loss = tf.nn.softmax_cross_entropy_with_logits(y_, y)
print(loss)

exit()


SEED = 23455

'''生成[0-1)的随机数'''
rdm = np.random.RandomState(seed=SEED)
x = rdm.rand(32,2) #  32 * [x1, x2]
y_ = [[x1 + x2 + rdm.rand() / 10.0 - 0.05] for (x1, x2) in x] #  噪声 [0,1)/10 - 0.05

x = tf.cast(x, dtype=tf.float32)
y_ = tf.cast(y_, dtype=tf.float32)


train_db = tf.data.Dataset.from_tensor_slices((x, y_)).batch(2)

w1 = tf.Variable(tf.constant(tf.random.normal([2,1], stddev=1, seed=1)))

epoch = 15000
lr = 0.002

for epoch in range(epoch):
    for step, (x, y_) in enumerate(train_db):
        with tf.GradientTape() as tape:
            y = tf.matmul(x, w1)
            loss_mse = tf.reduce_mean(tf.square(y-y_))
        grad = tape.gradient(loss_mse,w1)
        w1.assign_sub(lr * grad)

    if epoch%500==0:
        print('Epoch: {}, loss: {}'.format(epoch, loss_mse))
        print('w: ', w1.numpy())
        #print('Epoch:', epoch)


#tf.losses.categorical_crossentropy



