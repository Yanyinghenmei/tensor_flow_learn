#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'Daniel'

import pickle
import numpy as np
import os

train_path_1 = '/data_batch_1'
train_path_2 = '/data_batch_2'
train_path_3 = '/data_batch_3'
train_path_4 = '/data_batch_4'
train_path_5 = '/data_batch_5'
test_path = '/test_batch'

dataset_dir = os.path.dirname(os.path.abspath(__file__))


def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding='bytes')
    return dict

def reshape_image(data):
    list = []
    for d in data:
        d = d.reshape((32, 32, 3))
        list.append(list)
    return list

def load_data():
    x_train = []
    t_train = []
    x_test = []
    t_test = []
    dic1 = unpickle(dataset_dir + train_path_1)
    dic2 = unpickle(dataset_dir + train_path_2)
    dic3 = unpickle(dataset_dir + train_path_3)
    dic4 = unpickle(dataset_dir + train_path_4)
    dic5 = unpickle(dataset_dir + train_path_5)
    test_dic = unpickle(dataset_dir + test_path)

    # 训练数据
    x_train.extend(dic1[b'data'])
    x_train.extend(dic2[b'data'])
    x_train.extend(dic3[b'data'])
    x_train.extend(dic4[b'data'])
    x_train.extend(dic5[b'data'])

    t_train.extend(dic1[b'labels'])
    t_train.extend(dic2[b'labels'])
    t_train.extend(dic3[b'labels'])
    t_train.extend(dic4[b'labels'])
    t_train.extend(dic5[b'labels'])

    x_train = np.array(x_train)
    x_train = x_train.reshape((50000, 3, 32, 32)).transpose(0,2,3,1)

    t_train = np.array(t_train)

    # 测试数据
    x_test.extend(test_dic[b'data'])
    t_test.extend(test_dic[b'labels'])

    x_test = np.array(x_test)
    x_test = x_test.reshape((10000, 3, 32, 32)).transpose(0,2,3,1)

    t_test = np.array(t_test)

    return (x_train, t_train), (x_test, t_test)

# (x_train, t_train), (x_test, t_test) = load_data()
# print(x_train.shape)