#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'Daniel'

'''
1. 自制数据集, 解决本领域应用
2. 数据增强, 扩充数据集
3. 断点续训, 存取模型
4. 参数存取
5. acc/loss 可视化, 查看训练效果
6. 应用程序, 给图识物
'''

import os, sys
sys.path.append(os.pardir)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from keras_learn01.generateds_mnist import load_mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator

(x_train, t_train), (x_test, t_test) = load_mnist()
x_train, x_test = x_train/255.0, x_test/255.0

'''数据增强'''
image_gen_train = ImageDataGenerator(
    rescale=1. / 1.,                 # 如果是图像, 分母为255时, 可归至0~1
    rotation_range=45,               # 随机45度旋转
    width_shift_range=.15,           # 宽度偏移
    height_shift_range=.15,          # 高度偏移
    horizontal_flip=True,            # 随机水平翻转
    zoom_range=0.5                   # 随机缩放50%
)

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
image_gen_train.fit(x_train)
image_flow = image_gen_train.flow(x_train, t_train,batch_size=32)

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dense(10,activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['sparse_categorical_accuracy']
)

model.fit(image_flow,epochs=100,validation_data=(x_test,t_test),validation_freq=1)

model.summary()

