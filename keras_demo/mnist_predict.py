#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'Daniel'

import tensorflow as tf
import numpy as np
from PIL import Image

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

'''复原模型'''
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dense(10,activation='softmax')
])

'''加载参数'''
checkpoint_save_path = "./checkpoint/mnist.ckpt"
model.load_weights(checkpoint_save_path)

predict_path = './my_data/predict_image/'
while(True):
    img_name = input('Please input image name:' + '\n')
    if img_name == 'exit()':
        exit()

    img_path = predict_path + img_name
    img = Image.open(img_path)
    img = img.resize((28,28), Image.ANTIALIAS)          # 转化为28*28
    img_arr = np.array(img.convert('L'))                # 转化为灰度图

    # 颜色取反
    img_arr = 255 - img_arr

    # 颜色取反 - 纯色图
    # for i in range(28):
    #     for j in range(28):
    #         if img_arr[i][j] < 200:
    #             img_arr[i][j] = 255
    #         else:
    #             img_arr[i][j] = 0

    img_arr = img_arr / 255.0
    x_predict = img_arr[tf.newaxis, ...]            # 由于训练是一个batch一次, 这里给预测数据添加一个维度

    '''预测结果'''
    result = model.predict(x_predict)
    pred = tf.argmax(result, axis=1)

    print('\n')
    tf.print(pred)


