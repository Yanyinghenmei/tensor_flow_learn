#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'Daniel'

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense

import matplotlib as mpl
mpl.use('TKAgg')
import matplotlib.pyplot as plt

from conv_learn01.cifar_10_datasets.cifar import load_data

'''六步法'''
'''
1. 自制数据集, 解决本领域应用
2. 数据增强, 扩充数据集
3. 断点续训, 存取模型
4. 参数存取
5. acc/loss 可视化, 查看训练效果
6. 应用程序, 给图识物
'''

# 数据
(x_train, t_train), (x_test, t_test) = load_data()
x_train = tf.cast(x_train, dtype=tf.float32)
t_train = tf.cast(t_train, dtype=tf.float32)
x_test = tf.cast(x_test, dtype=tf.float32)
t_test = tf.cast(t_test, dtype=tf.float32)

x_train, x_test = x_train/255.0, x_test/255.0

class Baseline(Model):
    def __init__(self):
        super(Baseline, self).__init__()
        self.c1 = Conv2D(filters=6, kernel_size=(5, 5), padding='same')  # 卷积层
        self.b1 = BatchNormalization()
        self.a1 = Activation('relu')
        self.p1 = MaxPool2D(pool_size=(2,2), strides=2, padding='same')  # 池化层
        self.d1 = Dropout(0.2)
        self.flatten = Flatten()
        self.f1 = Dense(128, activation='relu')
        self.d2 = Dropout(0.2)
        self.f2 = Dense(10, activation='softmax')

    def call(self, x):
        x = self.c1(x)
        x = self.b1(x)
        x = self.a1(x)
        x = self.p1(x)
        x = self.d1(x)
        x = self.flatten(x)
        x = self.f1(x)
        x = self.d2(x)
        y = self.f2(x)
        return y

model = Baseline()

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['sparse_categorical_accuracy']
)

'''断点续训'''
checkpoint_save_path = './checkpoint/Baseline.ckpt'
if os.path.exists(checkpoint_save_path + '.index'):
    print('------- load the model -------')
    model.load_weights(checkpoint_save_path)

'''保存参数'''
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_save_path,
    save_weights_only=True,
    save_best_only=True
)

'''训练'''
history = model.fit(
    x_train, t_train,
    batch_size=32,
    epochs=5,
    validation_data=(x_test, t_test),
    validation_freq=1,
    callbacks=[cp_callback]
)

model.summary()

'''保存参数到txt'''
file = open('./weights.txt', 'w')
for v in model.trainable_variables:
    file.write(str(v.name) + '/n')
    file.write(str(v.shape) + '/n')
    file.write(str(v.numpy()) + '/n')
file.close()

'''可视化'''
acc = history.history['sparse_categorical_accuracy']
val_acc = history.history['val_sparse_categorical_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.subplot(1,2,1)
plt.plot(acc, label='train acc')
plt.plot(val_acc, label='val acc')
plt.title('train & val acc')

plt.subplot(1,2,2)
plt.plot(loss, label='trian loss')
plt.plot(val_loss, label='val loss')
plt.title('train & val loss')

plt.legend()
plt.show()


