#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'Daniel'

import os, sys
sys.path.append(os.pardir)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

import matplotlib as mpl
mpl.use('TKAgg')
import matplotlib.pyplot as plt

from conv_learn01.cifar_10_datasets.cifar import load_data

import numpy as np
np.set_printoptions(threshold=np.inf)


(x_train, t_trian), (x_test, t_test) = load_data()

print(x_train.shape)
print(x_test.shape)

print(t_trian[0])

plt.imshow(x_train[0])
plt.show()