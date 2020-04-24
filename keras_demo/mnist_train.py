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

import matplotlib as mpl
mpl.use('TKAgg')
import matplotlib.pyplot as plt

# from keras_demo.generateds_mnist import load_mnist
from dataset.mnist import load_mnist

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
np.set_printoptions(threshold=np.inf)           # np.inf 标识无限大, 不省略打印


(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False,normalize=False)
x_train, x_test = x_train/255.0, x_test/255.0

'''数据增强'''
image_gen_train = ImageDataGenerator(
    rescale=1. / 1.,                 # 如果是图像, 分母为255时, 可归至0~1
    rotation_range=20,               # 随机20度旋转
    width_shift_range=.15,           # 宽度偏移
    height_shift_range=.15,          # 高度偏移
    #horizontal_flip=True,            # 随机水平翻转
    zoom_range=0.3                   # 随机缩放30%
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

'''断点续训'''
checkpoint_save_path = "./checkpoint/mnist.ckpt"
if os.path.exists(checkpoint_save_path + '.index'):
    print('---------load the model----------')
    model.load_weights(checkpoint_save_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_save_path,
    save_weights_only=True,
    save_best_only=True
)

'''开始训练'''
history = model.fit(
    image_flow,epochs=5,
    validation_data=(x_test,t_test),
    validation_freq=1,
    callbacks=[cp_callback]
)

model.summary()

'''参数提取'''
# print(model.trainable_variables)
file = open('./weights.txt', 'w')
for v in model.trainable_variables:
    file.write(str(v.name) + '\n')
    file.write(str(v.shape) + '\n')
    file.write(str(v.numpy()) + '\n')
file.close()

'''acc/loss 可视化'''
acc = history.history['sparse_categorical_accuracy']
val_acc = history.history['val_sparse_categorical_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.subplot(1,2,1)
plt.plot(acc,label='Training Accuracy')
plt.plot(val_acc,label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(loss,label='Training Loss')
plt.plot(val_loss,label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.show()