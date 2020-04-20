#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'Daniel'

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

a = tf.Variable(tf.random.truncated_normal([2,3], seed=1, stddev=0.1))
print(a)

b = tf.Variable(tf.random.truncated_normal([3], seed=1, stddev=1))
print(b)

loss = a + b
print(loss)

loss = tf.add(a,b)
print(loss)