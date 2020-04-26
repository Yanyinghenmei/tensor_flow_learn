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


class ResnetBlock(Model):
    def __init__(self, filters, strides=1, residual_path=False):
        super(ResnetBlock, self).__init__()
        self.filters = filters
        self.strides = strides
        self.residual_path = residual_path

        self.c1 = Conv2D(filters, (3,3), strides=strides, padding='same', use_bias=False)
        self.b1 = BatchNormalization()
        self.a1 = Activation('relu')

        self.c2 = Conv2D(filters, (3,3), strides=1, padding='same', use_bias=False)
        self.b2 = BatchNormalization()

        # residual_path为True时, 对输入进行下采样, 即使用1x1卷积核做卷积操作改变深度, 顺利相加
        if residual_path==True:
            self.down_c1 = Conv2D(filters, (1,1), strides=strides, padding='same', use_bias=False)
            self.down_b1 = BatchNormalization()

        self.a2 = Activation('relu')

    def call(self, inputs):
        residual = inputs # residual等于输入值本身
        # 即将通过卷积, BN层, 激活层, 计算F(x)
        x = self.c1(inputs)
        x = self.b1(x)
        x = self.a1(x)

        x = self.c2(x)
        y = self.b2(x)

        if self.residual_path==True:
            residual = self.down_c1(inputs)
            residual = self.down_b1(residual)

        out = self.a2(residual + y)
        return out

class ResNet18(Model):
    def __init__(self, block_list, initial_filters=64):
        super(ResNet18, self).__init__()
        self.num_blocks = len(block_list)
        self.block_list = block_list
        self.out_filters = initial_filters

        self.c1 = Conv2D(self.out_filters, (3,3), strides=1, padding='same', use_bias=False)
        self.b1 = BatchNormalization()
        self.a1 = Activation('relu')
        self.blocks = tf.keras.models.Sequential()
        for block_id in range(len(block_list)):
            for layer_id in range(block_list[block_id]):
                if block_id != 0 and layer_id == 0:
                    # 对除第一个block之外的每个block进行下采样
                    block = ResnetBlock(self.out_filters, strides=2, residual_path=True)
                else:
                    block = ResnetBlock(self.out_filters, residual_path=False)
                self.blocks.add(block)
            self.out_filters *= 2 # 下一个block的卷积核数目是上一个的2倍
        self.p1 = tf.keras.layers.GlobalAveragePooling2D()
        self.f1 = tf.keras.layers.Dense(10, activation='softmax',
                                        kernel_regularizer=tf.keras.regularizers.l2())

    def call(self, inputs):
        x = self.c1(inputs)
        x = self.b1(x)
        x = self.a1(x)
        x = self.blocks(x)
        x = self.p1(x)
        y = self.f1(x)
        return y


# 数据
(x_train, t_train), (x_test, t_test) = load_data()
x_train = tf.cast(x_train, dtype=tf.float32)
t_train = tf.cast(t_train, dtype=tf.float32)
x_test = tf.cast(x_test, dtype=tf.float32)
t_test = tf.cast(t_test, dtype=tf.float32)

x_train, x_test = x_train/255.0, x_test/255.0

model = ResNet18([2,2,2,2])

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['sparse_categorical_accuracy']
)

checkpoint_save_path = './checkpoint/ResNet18.ckpt'
if os.path.exists(checkpoint_save_path + '.index'):
    print('--------- load the model ---------')
    model.load_weights(checkpoint_save_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_save_path,
    save_weights_only=True,
    save_best_only=True
)

his = model.fit(
    x_train, t_train,
    batch_size=16,
    epochs=5,
    validation_data=(x_test, t_test),
    validation_freq=1,
    callbacks=[cp_callback]
)

model.summary()

# acc & loss
acc = his.history['sparse_categorical_accuracy']
val_acc = his.history['val_sparse_categorical_accuracy']
loss = his.history['loss']
val_loss = his.history['val_loss']

plt.subplot(1,2,1)
plt.plot(acc, label='acc')
plt.plot(val_acc, label='val_acc')
plt.title('acc')
plt.legend()

plt.subplot(1,2,2)
plt.plot(loss, label='loss')
plt.plot(val_loss, label='val_loss')
plt.title('loss')
plt.legend()

plt.show()

