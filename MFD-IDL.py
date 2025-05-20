import pickle as pkl
import umap.umap_ as umap
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import math
import os
import numpy.linalg as la
from input_data import preprocess_data
from tgcn import tgcnCell
from sklearn.model_selection import train_test_split

from visualization1 import plot_result, plot_error
from sklearn.metrics import mean_squared_error, mean_absolute_error
import time

time_start = time.time()
###### Settings ######
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_integer('training_epoch', 300, 'Number of epochs to train.')
flags.DEFINE_integer('gru_units', 64, 'hidden units of gru.')
flags.DEFINE_integer('seq_len', 6, '  time length of inputs.')
flags.DEFINE_integer('pre_len', 1, 'time length of prediction.')
flags.DEFINE_float('train_rate', 0.8, 'rate of training set.')
flags.DEFINE_integer('batch_size', 33, 'batch size.')
flags.DEFINE_string('dataset', 'los', 'sz or los.')
flags.DEFINE_string('model_name', 'tgcn', 'tgcn')
model_name = FLAGS.model_name
data_name = FLAGS.dataset
train_rate = FLAGS.train_rate
seq_len = FLAGS.seq_len
output_dim = pre_len = FLAGS.pre_len
batch_size = FLAGS.batch_size
lr = FLAGS.learning_rate
training_epoch = FLAGS.training_epoch
gru_units = FLAGS.gru_units

###### load data ######
data = pd.read_excel(r'E:\I-24motiondata/dataset/flowdemo.xlsx')
adjor = pd.read_excel(r'E:\I-24motiondata/dataset/adjdemo1.xlsx')
density = pd.read_excel(r'E:\I-24motiondata/dataset/densitydemo.xlsx')
data = np.array(data)
adjor = np.array(adjor)
density = np.array(density)
# data = data[:, 0:30]
# adjor = adjor[0:30, 0:30]
# density = density[:, 0:30]
print(data.shape)
print(adjor.shape)

time_len = data.shape[0]
num_nodes = data.shape[1]

print(time_len)
print(num_nodes)

data1 = np.mat(data,dtype=np.float32)*36
data2 = np.array(density)
# noise = np.random.normal(0,2,size=data.shape)
# noise = np.random.poisson(16,size=data.shape)
# scaler = MinMaxScaler()
# scaler.fit(noise)
# noise = scaler.transform(noise)
# data1 = data1 + noise

def introduce_random_missing(data, missing_rate=0.1):
    mask = np.random.random(data.shape)
    data_with_missing = data.copy()
    data_with_missing[mask < missing_rate] = np.nan
    return data_with_missing

def fill_missing_with_neighbor_mean(data_with_missing):
    filled_data = data_with_missing.copy()
    rows, cols = data_with_missing.shape

    for i in range(rows):
        for j in range(cols):
            if np.isnan(filled_data[i, j]):
                # 收集周围8个邻居的值
                neighbors = []
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == 0 and dj == 0:
                            continue  # 跳过自身
                        ni, nj = i + di, j + dj
                        if 0 <= ni < rows and 0 <= nj < cols and not np.isnan(filled_data[ni, nj]):
                            neighbors.append(filled_data[ni, nj])

                # 如果有邻居值，用均值填充；否则保持nan
                if neighbors:
                    filled_data[i, j] = np.mean(neighbors)
                # 如果没有邻居值，暂时保持nan（可以在外层再处理）

    return filled_data


# 缺失比例设定
missing_rate = 0

# 引入随机缺失值
data_with_missing = introduce_random_missing(data1, missing_rate=missing_rate)

# 填充缺失值
data1 = fill_missing_with_neighbor_mean(data_with_missing)

testdata = np.zeros((time_len, num_nodes))

def fun(x, a=0.7293506, b=0.01037809):
    return a*x*np.exp(-b*x)

# def fun(x, a=0.853573, b=0.022964):
#     return a*x*np.exp(-b*x)

def fun1(x, a, b):
    a1 = tf.multiply(a, x)
    b1 = tf.exp(-tf.multiply(b, x))
    y = tf.multiply(a1, b1)
    return y

def gated_fusion(spacor, temcor):
    spacor = tf.cast(spacor, tf.float32)
    temcor = tf.cast(temcor, tf.float32)
    z = tf.matmul(spacor, weight1['out']) + tf.matmul(temcor, weight2['out']) + biase['out']
    z = tf.nn.sigmoid(z)
    adj = tf.matmul(z, spacor) + tf.matmul((1 - z), temcor)
    return adj

def fusion(dl, mfd):
    trdata = tf.multiply(alpha, dl) + tf.multiply((1 - alpha), mfd)
    return trdata

def correlation(x1, x2):
    m1 = np.mean(x1)
    m2 = np.mean(x2)
    c1 = np.std(x1)
    c2 = np.std(x2)
    cor = np.mean((x1-m1)*(x2-m2))
    autoco = cor/np.sqrt(c1*c2)
    return autoco

def cor_attention(x, y, weight1_att, weight2_att):
    x = tf.unstack(x, axis=1)
    # y = tf.unstack(y, axis=1)
    x = tf.reshape(x, shape=[num_nodes, -1])
    y = tf.reshape(y, shape=[num_nodes, -1])
    # x = tf.transpose(x)
    # y = tf.transpose(y)
    query = tf.matmul(x, weight1_att['out'])
    key = tf.matmul(x, weight1_att['out'])
    att_score = tf.matmul(query, tf.transpose(key))
    value = tf.matmul(y, weight2_att['out'])
    result = tf.matmul(att_score, value)
    print(result.shape)
    return result

def softmax(x):
   e_x = np.exp(x - np.max(x))
   return e_x / np.sum(e_x, axis=0)

def TGCN(_X, _weights, _biases, mfdest):
    ###
    cell_1 = tgcnCell(gru_units, adj, num_nodes=num_nodes)
    cell = tf.nn.rnn_cell.MultiRNNCell([cell_1], state_is_tuple=True)
    _X = tf.unstack(_X, axis=1)
    outputs, states = tf.nn.static_rnn(cell, _X, dtype=tf.float32)
    m = []
    for i in outputs:
        o = tf.reshape(i, shape=[-1, num_nodes, gru_units])
        o = tf.reshape(o, shape=[-1, gru_units])
        m.append(o)
    last_output = m[-1]
    output = tf.matmul(last_output, _weights['out']) + _biases['out']
    output = tf.reshape(output, shape=[-1, num_nodes, pre_len])
    output = tf.transpose(output, perm=[0, 2, 1])
    output = tf.reshape(output, shape=[-1, num_nodes])
    mfdest = tf.reshape(mfdest, shape=[-1, num_nodes])
    output = fusion(output, mfdest)
    return output, m, states

# Graph weights
weights = {
    'out': tf.Variable(tf.random_normal([gru_units, pre_len], mean=1.0), name='weight_o')}
biases = {
    'out': tf.Variable(tf.random_normal([pre_len]), name='bias_o')}

weight1 = {
    'out': tf.Variable(tf.random_normal([num_nodes, num_nodes], mean=1.0), name='weight1_o')}
weight2 = {
    'out': tf.Variable(tf.random_normal([num_nodes, num_nodes], mean=1.0), name='weight1_o')}
biase = {
    'out': tf.Variable(tf.random_normal([num_nodes]), name='bias_o')}
alpha = 0.8

alpha = tf.Variable(alpha)
weight1_att = {
    'out': tf.Variable(tf.random_normal([seq_len, pre_len], mean=1.0), name='weight1_o')}
weight2_att = {
    'out': tf.Variable(tf.random_normal([seq_len, pre_len], mean=1.0), name='weight1_o')}

datagraph1 = data1[100: 300, :]
sflow = np.zeros((num_nodes, num_nodes))
for i in range(num_nodes):
    for j in range(num_nodes):
        a = datagraph1[:, i]
        b = datagraph1[:, j]
        sflow[i, j] = correlation(a, b.T)
# np.set_printoptions(threshold=np.inf)
sflow = softmax(sflow)

timelen = datagraph1.shape[0]
tflow = np.zeros((timelen, timelen))
for i in range(timelen):
    for j in range(timelen):
        a = datagraph1[i, :]
        b = datagraph1[j, :]
        tflow[i, j] = correlation(a, b.T)
tflow = softmax(tflow)

datagraph2 = data2[100: 500, :]
sdensity = np.zeros((num_nodes, num_nodes))
for i in range(num_nodes):
    for j in range(num_nodes):
        a = datagraph2[:, i]
        b = datagraph2[:, j]
        sdensity[i, j] = correlation(a, b.T)
sdensity = softmax(sdensity)

tdensity = np.zeros((timelen, timelen))
for i in range(timelen):
    for j in range(timelen):
        a = datagraph2[i, :]
        b = datagraph2[j, :]
        tdensity[i, j] = correlation(a, b.T)
tdensity = softmax(tdensity)

spatial_cor = np.dot(sflow, sdensity.T)
temporal_cor = np.dot(tflow, tdensity.T)
for i in range(len(temporal_cor)):
    sample = temporal_cor[i]
    for j in range(len(sample)):
        if np.isnan(sample[j]):
            sample[j] = 0
data_reducer = umap.UMAP(n_neighbors=3, n_components=num_nodes)
temporal_cor = data_reducer.fit_transform(temporal_cor)
temporal_cor = temporal_cor.T
temporal_cor = data_reducer.fit_transform(temporal_cor)

STCor = gated_fusion(spatial_cor, temporal_cor)
variables = tf.global_variables()
saver = tf.train.Saver(tf.global_variables())
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
sess.run(tf.global_variables_initializer())

#### normalization
mfd = fun(data2)
trainX, trainY, testX, testY = preprocess_data(data1, time_len, train_rate, seq_len, pre_len)
mtrainX, mtrainY, mtestX, mtestY = preprocess_data(mfd, time_len, train_rate, seq_len, pre_len)
# print(trainX.shape)

totalbatch = int(trainX.shape[0] / batch_size)
training_data_count = len(trainX)

###### placeholders ######
inputs = tf.placeholder(tf.float32, shape=[None, seq_len, num_nodes])
labels = tf.placeholder(tf.float32, shape=[None, pre_len, num_nodes])
mfdest = tf.placeholder(tf.float32, shape=[None, pre_len, num_nodes])

STCor = sess.run(STCor)
adj = np.zeros((num_nodes, num_nodes))
for i in range(num_nodes):
    for j in range(num_nodes):
        if adjor[i, j] == 0:
            adj[i, j] = 0
        else:
            adj[i, j] = STCor[i, j]
    adj[i, i] = 1
# print(adj)

if model_name == 'tgcn':
    pred, ttts, ttto = TGCN(inputs, weights, biases, mfdest)

y_pred = pred

###### optimizer ######
lambda_loss = 0.0015
Lreg = lambda_loss * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
label = tf.reshape(labels, [-1, num_nodes])
##loss
alpha = sess.run(alpha)
loss = alpha*tf.nn.l2_loss(y_pred - label) + (1-alpha)*tf.nn.l2_loss(y_pred - mfdest) + Lreg
##rmse
error = tf.sqrt(tf.reduce_mean(tf.square(y_pred - label)))
optimizer = tf.train.AdamOptimizer(lr).minimize(loss)

###### Initialize session ######
# sess = tf.Session()
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
sess.run(tf.global_variables_initializer())

out = 'out/%s' % (model_name)
# out = 'out/%s_%s'%(model_name,'perturbation')
path1 = '%s_%s_lr%r_batch%r_unit%r_seq%r_pre%r_epoch%r' % (
model_name, data_name, lr, batch_size, gru_units, seq_len, pre_len, training_epoch)
path = os.path.join(out, path1)
if not os.path.exists(path):
    os.makedirs(path)


###### evaluation ######
def evaluation(a, b):
    rmse = math.sqrt(mean_squared_error(a, b))
    mae = mean_absolute_error(a, b)
    F_norm = la.norm(a - b, 'fro') / la.norm(a, 'fro')
    r2 = 1 - ((a - b) ** 2).sum() / ((a - a.mean()) ** 2).sum()
    var = 1 - (np.var(a - b)) / np.var(a)
    return rmse, mae, 1 - F_norm, r2, var


x_axe, batch_loss, batch_rmse, batch_pred = [], [], [], []
test_loss, test_rmse, test_mae, test_acc, test_r2, test_var, test_pred = [], [], [], [], [], [], []

for epoch in range(training_epoch):
    for m in range(totalbatch):
        mini_batch = trainX[m * batch_size: (m + 1) * batch_size]
        mini_label = trainY[m * batch_size: (m + 1) * batch_size]
        mfdest_batch = mtrainY[m * batch_size: (m + 1) * batch_size]
        _, loss1, rmse1, train_output = sess.run([optimizer, loss, error, y_pred],
                                                 feed_dict={inputs: mini_batch, labels: mini_label, mfdest: mfdest_batch})
        batch_loss.append(loss1)
        batch_rmse.append(rmse1)
        testset_batch = 0.8*mtestY

        # Test completely at every epoch
    loss2, rmse2, test_output = sess.run([loss, error, y_pred],
                                         feed_dict={inputs: testX, labels: testY, mfdest: testset_batch})
    test_label = np.reshape(testY, [-1, num_nodes])
    rmse, mae, acc, r2_score, var_score = evaluation(test_label, test_output)
    test_label1 = test_label
    test_output1 = test_output
    test_loss.append(loss2)
    test_rmse.append(rmse)
    test_mae.append(mae)

    test_acc.append(acc)
    test_r2.append(r2_score)
    test_var.append(var_score)
    test_pred.append(test_output1)

    print('Iter:{}'.format(epoch),
          'train_rmse:{:.4}'.format(batch_rmse[-1]),
          'test_loss:{:.4}'.format(loss2),
          'test_rmse:{:.4}'.format(rmse),
          'test_acc:{:.4}'.format(acc))

    if (epoch % 500 == 0):
        saver.save(sess, path + '/model_100/TGCN_pre_%r' % epoch, global_step=epoch)

time_end = time.time()
print(time_end - time_start, 's')

############## visualization ###############
b = int(len(batch_rmse) / totalbatch)
batch_rmse1 = [i for i in batch_rmse]
train_rmse = [(sum(batch_rmse1[i * totalbatch:(i + 1) * totalbatch]) / totalbatch) for i in range(b)]
batch_loss1 = [i for i in batch_loss]
train_loss = [(sum(batch_loss1[i * totalbatch:(i + 1) * totalbatch]) / totalbatch) for i in range(b)]
index = test_rmse.index(np.min(test_rmse))
test_result = test_pred[index]
print(test_label1.shape)
var = pd.DataFrame(test_result)
# var.to_csv(path_or_buf=r'F:\datashangbo\NHTS_data_csv/msz32.csv', index=False)
# var.to_csv(path + '/test_result.csv', index=False, header=False)

plot_result(test_result, test_label1, path)
plot_error(train_rmse, train_loss, test_rmse, test_acc, test_mae, path)

print(alpha)
print('min_rmse:%r' % (np.min(test_rmse)),
      'min_mae:%r' % (test_mae[index]),
      'max_acc:%r' % (test_acc[index]),
      'r2:%r' % (test_r2[index]),
      'var:%r' % test_var[index])

df1 = pd.DataFrame(test_result)
df2 = pd.DataFrame(test_label1)

# df1.to_excel('MDMGF_Fri.xlsx', index=False)
# # df2.to_excel('realdata.xlsx', index=False)
