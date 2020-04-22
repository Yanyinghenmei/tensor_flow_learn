#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'Daniel'
import os
from PIL import Image
import numpy as np

os.getcwd()

train_path = '../keras_learn01/my_data/mnist_image_label/train_png_60000/'
train_txt = '../keras_learn01/my_data/mnist_image_label/train_png_60000.txt'
x_train_savepath = '../keras_learn01/my_data/mnist_image_label/mnist_x_train.npy'
y_train_savepath = '../keras_learn01/my_data/mnist_image_label/mnist_y_train.npy'

test_path = '../keras_learn01/my_data/mnist_image_label/test_png_60000/'
test_txt = '../keras_learn01/my_data/mnist_image_label/test_png_60000.txt'
x_test_savepath = '../keras_learn01/my_data/mnist_image_label/mnist_x_test.npy'
y_test_savepath = '../keras_learn01/my_data/mnist_image_label/mnist_y_test.npy'


def generateds(path, txt):
    f = open(txt, 'r')                      # 以只读形式打开 txt文件
    contents = f.readlines()                # 读取文件所有行
    f.close()                               # 关闭TXT文件
    x, y_ = [], []                          # 建立空列表
    for content in contents:                # 逐行读取
        value = content.split()             # 以空格分开, 图片名为value[0], 标签为value[1]
        img_path = path + value[0]          # 拼接图片路径
        img = Image.open(img_path)          # 读入图片
        img = np.array(img.convert('L'))    # 图片变为8位宽度灰度值的np.array格式
        img = img/255.                      # 数据归一化 (实现预处理)
        x.append(img)                       # 归一化后的数据存入数组
        y_.append(value[1])                 # 标签存入数组
        print('loading: ' + content)        # 打印状态提示

    x = np.array(x)                         # 转为np.array 格式
    y_= np.array(y_)                        # 转为np.array 格式
    y_ = y_.astype(np.int64)                # 标签类型转为 np.int64
    return x, y_                            # 返回数据

def load_mnist():
    if os.path.exists(x_train_savepath) and os.path.exists(y_train_savepath) and \
            os.path.exists((x_test_savepath)) and os.path.exists(y_test_savepath):
        print('---------------- Load Datasets --------------')
        x_train_save = np.load(x_train_savepath)
        y_train = np.load(y_train_savepath)
        x_test_save = np.load(x_test_savepath)
        y_test = np.load(y_test_savepath)
        x_train = x_train_save.reshape((len(x_train_save), 28, 28))
        x_test = x_test_save.reshape((len(x_test_save), 28, 28))
        return (x_train, y_train), (x_test, y_test)

    else:
        print('---------------- Generate Datasets --------------')
        x_train, y_train = generateds(train_path, train_txt)
        x_test, y_test = generateds(test_path, test_txt)
        print('---------------- Save Datasets --------------')
        x_train_save = np.reshape(x_train, (len(x_train), -1))
        x_test_save = np.reshape(x_test, (len(x_test), -1))
        np.save(x_train_savepath, x_train_save)
        np.save(y_train_savepath, y_train)
        np.save(x_test_savepath, x_test_save)
        np.save(y_test_savepath, y_test)
        return (x_train, y_train), (x_test, y_test)


'''test'''
# (x_train, y_train), (x_test, y_test) =  load_mnist()
# print(x_train.shape)
# print(y_train.shape)
# print(x_test.shape)
# print(y_test.shape)
