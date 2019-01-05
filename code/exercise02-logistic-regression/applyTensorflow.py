import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import tensorflow as tf

df = pd.read_csv('ex2data1.txt',header=None)
train_data = df.values

train_X = train_data[:,:-1]
train_y = train_data[:,-1:]

feature_num = train_X.shape[1]
sample_num = train_X.shape[0]

# 数据集
X = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

# 训练目标
W = tf.Variable(tf.zeros([feature_num,1]))
b = tf.Variable(-.9)

db = tf.matmul(X,tf.reshape(W,[-1,1]))+b
hyp = tf.sigmoid(db)

cost0 = y * tf.log(hyp)
cost1 = (1 - y) * tf.log(1 - hyp)
cost = (cost0 + cost1) / -sample_num
loss = tf.reduce_sum(cost)

optimizer = tf.train.GradientDescentOptimizer(0.001) #learning rate 应当尽可能小
train = optimizer.minimize(loss)

