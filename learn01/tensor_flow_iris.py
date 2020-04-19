#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'Daniel'

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from sklearn.datasets import load_iris
import tensorflow as tf
import numpy as np

x = load_iris().data
t = load_iris().target

x_tf = tf.constant(x)
t_tf = tf.constant(t)
# print(x_tf)
# print(t_tf)

weight_std = 0.1
lr = 0.1

w = tf.Variable(tf.random.normal([4,3],0.1,0.1))
b = tf.constant([3,], dtype=tf.float32)

for i, ele in enumerate(x):
    y = tf.matmul(tf.constant(ele.reshape(1,-1), dtype=tf.float32), w) + b
    res = tf.nn.softmax(y)
    #print(res)

    ele_t = np.array([0,0,0])
    ele_t[t_tf[i]] = 1
    # print(ele_t)

    with tf.GradientTape() as tape:
        loss = tf.reduce_max(tf.square(tf.subtract(tf.constant(ele_t, dtype=tf.float32), res)))
    grad = tape.gradient(loss, w)
    print(w)
    print(grad)
    # w.assign_sub(grad * lr)


    break
    # y = tf.matmul(ele, w) + b
    # t = tf.constant(t_tf[i])
    # print(y, t)

# print(w)


