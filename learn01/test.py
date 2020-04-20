#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'Daniel'

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np

# a = tf.Variable(tf.random.truncated_normal([2,3], seed=1, stddev=0.1))
# print(a)
#
# b = tf.Variable(tf.random.truncated_normal([3], seed=1, stddev=1))
# print(b)
#
# loss = a + b
# print(loss)
#
# loss = tf.add(a,b)
# print(loss)

'''tf.where(表达式, value1, value2)'''
# a = tf.constant([1,2,3,1,1])
# b = tf.constant([0,1,3,4,5])
# c = tf.where(tf.greater(a,b), a, b)
# print(c)

'''返回一个0-1之间的随机数, 如果维度为空, 则返回标量'''
# rdm = np.random.RandomState(seed=1)
# a = rdm.rand()
# b = rdm.rand(2,3)
# print(a)
# print(b)


'''np.vstack -- 数组纵向堆叠'''
# a = np.array([1,2,3])
# b = np.array([4,5,6])
# c = np.vstack((a,b))
# print(c)

'''np.mgrid[], .ravel(), np.c_[]'''
x, y = np.mgrid[1:4:1, 3:6:1]
# print(x)
# print(y)
# print('----------------------')
# x, y = np.mgrid[3:6:1, 1:4:1]
# print(x)
# print(y)
grid = np.c_[x.ravel(),y.ravel()]
print(x.ravel())
print(y.ravel())
print(grid)

# print(x.ravel())
# print(np.c_[x,y])