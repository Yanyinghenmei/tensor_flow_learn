
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np


# w = tf.Variable(tf.constant(5, dtype=tf.float32)) # w 初始化为 5
# lr = 0.2                                            # 学习率 0.2
# epochs = 40                                         # 训练次数
#
# for epoch in range(epochs):
#     with tf.GradientTape() as tape:                 # 梯度计算
#         loss = tf.square(w + 1)                     # 损失函数
#     grades = tape.gradient(loss, w)                 # 梯度计算
#     w.assign_sub(lr * grades)                       # 更新 w
#
#     print(w)


# a = tf.Variable(tf.constant([1,5], dtype=tf.float32))
# b = tf.constant([1,5], dtype=tf.float32)
# c = tf.convert_to_tensor(np.array([1,5]), dtype=tf.float32) # 转变np类型的矩阵转变为TensorFlow类型的张量
#
# print(a)
# print(b)
# print(c)

# # 全为 0 的张量
# a = tf.zeros([3,3], dtype=tf.float32)
# print(a)
#
# # 全为 1 的张量
# b = tf.ones([3,3], dtype=tf.float32)
# print(b)
#
# # 全为 指定值 的张量
# c = tf.fill([3,3], 10.0)
# print(c)

# 随机生成参数

# # 正态分布
# a = tf.random.normal([2,2])
# print(a)
#
# # 正态分布 数据更向均值集中
# b = tf.random.truncated_normal([2,2], mean=0.5, stddev=0.5)
# print(b)

# # 均匀分布分布 minval 最小, maxval 最大, 前闭后开
# a = tf.random.uniform([2,2], minval=1, maxval=3)
# print(a)
#
# # 强制类型转换
# b = tf.cast(a, dtype=tf.int32)
# print(b)
#
# # 获取张量中的最小值
# c = tf.reduce_min(a)
#
# # 获取张量中的最大值
# m = tf.reduce_max(a)
# print(c)
# print(m)


# a = tf.random.uniform([2,4], minval=0, maxval=10)
# b = tf.reduce_min(a, axis=1).T
# print(a)
# print(b)

# tf.Variable 将变量标记为'可训练的', 会在反向传播中记录自己的梯度
# w = tf.Variable(tf.random.normal([2,3], mean=0.5, stddev=0.5))
# print(w)


'''
  常见的计算
    tf.add()            加
    tf.subtract()       减
    tf.multiply()       乘
    tf.divide()         除
    tf.square()         平方
    tf.pow()            次方
    tf.sqrt()           开平方
    tf.matmul()         矩阵乘法
'''

'''
  配对 --- 用于数据与标签配对
'''
# x = tf.constant([1,2,3,4], dtype=tf.float32)
# t = tf.constant([4,5,6,7], dtype=tf.float32)
# data = tf.data.Dataset.from_tensor_slices((x,t))
# print(data)
#
# for ele in data:
#     print(ele)

'''
  求导
'''
# w = tf.Variable(tf.constant([2,2], dtype=tf.float32))
# print(w)
#
# with tf.GradientTape() as tape:
#     loss = tf.pow(w,2)                  # loss = w^2
# grad = tape.gradient(loss,w)        # 对w求导
# print(grad)

'''
  遍历
'''
# a = [23,43,54]
# for i, val in enumerate(a):
#     print(i, val)

'''
  index标签 转换为one_hot标签
'''
# t = tf.constant([1,0,2])
# t_one_hot = tf.one_hot(t, depth=3)
# print(t_one_hot)


'''softmax'''
# y = tf.constant([0.3,12,-1])
# p = tf.nn.softmax(y)
# print(p)

'''参数自更新'''
# a = tf.constant([1,2])
# a = tf.Variable(tf.constant(1, dtype=tf.float32))
# a.assign_sub(0.2)
# print(a)

# b = tf.convert_to_tensor(np.array([1,2,3]), dtype=tf.float32)
# vb = tf.Variable(b)
# vb.assign_sub(tf.convert_to_tensor(np.array([0.2,0.2,0.2]), dtype=tf.float32))
# print(vb)

# w = tf.Variable([1,2], dtype=tf.float32)
# w.assign_sub([0.2,0.2])
# print(w)

# print(tf.Variable([1,2]))
# print(tf.Variable(tf.constant([1,2])))
# print(tf.constant([1,2]))
# print(tf.Variable(np.array([1,2]), dtype=tf.float32))






















