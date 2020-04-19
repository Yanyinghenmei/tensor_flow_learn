#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'Daniel'

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

print('123')
x = np.arange(0, 5, 0.1)
y = x * 3 + 2

plt.scatter(x, y)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(1, input_shape=(1,)))
model.compile(optimizer='adam', loss='mse')
his = model.fit(x,y,epochs=5000)
pre = model.predict(x)

plt.plot(x, pre)
plt.show()
