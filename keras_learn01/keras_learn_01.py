#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'Daniel'

'''六步法'''
'''
1. import
2. train, test 准备数据
3. model = tf.keras.models.Sequential()  搭建网络结构, 逐层描述网络, 相当于前向传播
4. model.compile 配置训练方法, 选择优化器, 选择损失函数, 选择损失指标
5. model.fit 训练
6. model.summary 打印网络结构和参数统计
'''

'''import'''
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
from sklearn import datasets

'''train, test'''
x_train = datasets.load_iris().data
t_train = datasets.load_iris().target

np.random.seed(116)
np.random.shuffle(x_train)
np.random.seed(116)
np.random.shuffle(t_train)
tf.random.set_seed(116)

'''sequential'''
model = tf.keras.models.Sequential([
    # 神经元个数, 激活函数, 正则化方法
    tf.keras.layers.Dense(3, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2())
])

'''compile'''
# metrics: 网络评测指标
# accuracy  y_  数值, y  数值
# sparse_accuracy y_ 独热码, y 独热码/概率分布
# sparse_categorical_accuracy y_ 数值 y 独热码/概率分布
# 由于Dense层的激活函数为'softmax' 所以y为概率分布, 同时所给t_train是数值形式, 此处使用 'sparse_categorical_accuracy'
model.compile(optimizer=tf.optimizers.SGD(lr=0.1),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=[tf.keras.metrics.sparse_categorical_accuracy])

'''fit'''
# validation_data 测试数据
# validation_split 测试数据比例
# validation_freq 多少epoch测试一次
model.fit(x_train, t_train, batch_size=32, epochs=500, validation_split=0.2, validation_freq=20)

'''summary'''
model.summary()

