import pickle as pkl
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.optimize import curve_fit

def fun(x, a, b):
    return a*x*np.exp(-b*x)

def fun1(x, a, b):
    a1 = tf.multiply(a, x)
    b1 = tf.exp(-tf.multiply(b, x))
    y = tf.multiply(a1, b1)
    return y

data = pd.read_excel(r'E:\I-24motiondata/dataset/flowdemo5.xlsx')
density = pd.read_excel(r'E:\I-24motiondata/dataset/densitydemo5.xlsx')

data = np.array(data)
density = np.array(density)
flow = data.flatten()
# flow = list(flow)
density = density.flatten()
# density = list(density)

train_ra = 0.8
train_len = int(len(flow) * train_ra)
train_index = density[: train_len]
test_index = density[train_len: ]
train_label = flow[: train_len]
test_label = flow[train_len: ]

ppot, pcov = curve_fit(fun, train_index, train_label)

a_i, b_i = ppot
print(a_i)
print(b_i)

a_i = tf.Variable(a_i)
print(a_i)
b_i = tf.Variable(b_i)
test_index = tf.constant(test_index)
test_label = tf.constant(test_label)
mfdfit = tf.nn.relu(fun1(test_index, a_i, b_i))

epoch = 300
lr = 0.0000001
lambda_loss = 0.0015
Lreg = lambda_loss * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
loss = tf.reduce_mean(tf.square(mfdfit - test_label))
gradients = tf.gradients(loss, [a_i, b_i])
optimizer = tf.train.GradientDescentOptimizer(lr)
train_op = optimizer.apply_gradients(zip(gradients, [a_i, b_i]))
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print(sess.run(a_i))
    print(sess.run(b_i))
    for i in range(epoch):
        sess.run(train_op)
    a_i = sess.run(a_i)
    b_i = sess.run(b_i)
    print(a_i)
    print(b_i)

density = np.array(density, dtype=np.float64)

plt.scatter(density, flow, c='b', marker='.')
plt.plot(density, fun(density, a_i, b_i), 'r')
plt.grid()
plt.legend()
plt.show()