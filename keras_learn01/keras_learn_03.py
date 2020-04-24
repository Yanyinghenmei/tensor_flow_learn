#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'Daniel'

import os, sys
sys.path.append(os.pardir)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from dataset import mnist

(x_train, t_train), (x_test,t_test) = mnist.load_mnist(flatten=False,normalize=False)

# 正规化
x_train, x_test = x_train/255.0, x_test/255.0

# 模型
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dense(10,activation='softmax')
])

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['sparse_categorical_accuracy']
)

model.fit(x_train,t_train,batch_size=32,epochs=5,validation_data=(x_test,t_test),validation_freq=1)

model.summary()

