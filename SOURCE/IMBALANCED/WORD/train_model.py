#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 19:51:39 2019

@author: ghosh128
"""

import sys
sys.path.append("../")
import os
import numpy as np
import config
import tensorflow as tf
from scipy import optimize
tf.set_random_seed(1)
#%%
print("LOAD DATA")
train_data_strong = np.load(os.path.join(config.NUMPY_DIR, "train_data_strong.npy")).astype(np.float32)
train_data_weak = np.load(os.path.join(config.NUMPY_DIR, "train_data_weak.npy")).astype(np.float32)
num_features = train_data_strong.shape[-1] - 2

print("COMPUTE THETA")
O_s = train_data_strong[:,-1]
Y_s = train_data_strong[:,-2]
N_s = len(O_s)
num_levels = len(np.unique(O_s))

#Inequality 1 : theta_l_i - \xi_l_i <= y_i
A1 = np.zeros((N_s, 2*N_s+(num_levels-1)))
A1[list(range(N_s)), list(range(N_s))] = -1
A1[list(np.where(O_s>1)[0]), list(2*N_s + O_s[np.where(O_s>1)[0]].astype("int") -1 -1)] = 1
b1 = train_data_strong[:,-2]
b1 = np.reshape(b1, (-1,1))

#Inequality 2 : -theta_u_i - \xi_u_i <= -y_i
A2 = np.zeros((N_s, 2*N_s+(num_levels-1)))
A2[list(range(N_s)), list(range(N_s, N_s + N_s))] = -1
A2[list(np.where(O_s<num_levels)[0]), list(2*N_s + O_s[np.where(O_s<num_levels)[0]].astype("int") -1)] = -1
b2 = -Y_s
b2[np.where(O_s==num_levels)] += 1
b2 = np.reshape(b2, (-1,1))

#Inequality 3 : - \xi_l_i, -\xi_u_i <=0
A3 = np.zeros((2*N_s, 2*N_s+(num_levels-1)))
A3[list(range(2*N_s)), list(range(2*N_s))] = -1
b3 = np.zeros((2*N_s,1))
b3 = np.reshape(b3, (-1,1))

#Inequality 4 : -theta_k <=0 and theta_k <=1
A4 = np.zeros((2*(num_levels-1), 2*N_s+(num_levels-1)))
A4[list(range(num_levels-1)), list(range(2*N_s, 2*N_s+num_levels-1))] = -1
A4[list(range(num_levels-1, num_levels-1+num_levels-1)), list(range(2*N_s, 2*N_s+num_levels-1))] = 1
b4 = np.concatenate((np.zeros((num_levels-1,1)), np.ones((num_levels-1,1))))
b4 = np.reshape(b4, (-1,1))

f = np.concatenate((np.ones((2*N_s,1)), np.zeros((num_levels-1,1))))
A = np.concatenate((A1, A2, A3, A4))
b = np.concatenate((b1, b2, b3, b4))
x = optimize.linprog(c=f, A_ub=A, b_ub=b)
x = x["x"]
slack_var_lower = x[:N_s]
slack_var_upper = x[N_s:2*N_s]
theta = x[2*N_s:2*N_s+num_levels-1].astype(np.float32)

print("COMPUTE POSITIVENESS")
positiveness = np.zeros(num_levels).astype(np.float32)
for i in np.unique(O_s):
    positiveness[int(i-1)] = np.mean(Y_s[np.where(O_s == i)[0]])
positiveness = np.reshape(positiveness, (-1,1))

RESULT_DIR = os.path.join(config.RESULT_DIR, "WORD")
if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)
np.save(os.path.join(RESULT_DIR, "theta"), theta)
np.save(os.path.join(RESULT_DIR, "positiveness"), theta)

true_pos = positiveness[train_data_weak[:, -1].astype("int")-1].astype(np.float32)
#%%
print("BUILD MODEL")
tf.reset_default_graph()
with tf.name_scope('data'):
    X = tf.placeholder(tf.float32, [None, num_features], name="inputs")
    Y = tf.placeholder(tf.float32, [None, 1], name="labels")
    X_ord = tf.placeholder(tf.float32, [None, num_features], name="ordinal_inputs")
    Y_Pos = tf.placeholder(tf.float32, [None, 1], name="true_positiveness")

with tf.variable_scope("Variables", reuse=tf.AUTO_REUSE):
    W = tf.get_variable("W", [num_features, 1], initializer=tf.contrib.layers.xavier_initializer())
    b = tf.get_variable("b", [1], initializer=tf.zeros_initializer())

Z = tf.matmul(X, W, name="multiply_weights")
Z = tf.add(Z, b, name="add_bias")
Z = tf.sigmoid(Z)

Z_ord = tf.matmul(X_ord, W, name="multiply_weights")
Z_ord = tf.add(Z_ord, b, name="add_bias")
Z_ord = tf.sigmoid(Z_ord)

probs = tf.nn.sigmoid(tf.multiply(config.WORD_s, tf.subtract(Z_ord, theta)))
pred_ord = tf.reshape(tf.subtract(1.0, probs[:,0]), (-1,1))
for i in range(1, positiveness.shape[0]-1):
    pred_ord = tf.concat([pred_ord, tf.reshape(probs[:, i-1]-probs[:, i], (-1,1))], 1)
pred_ord = tf.concat([pred_ord, tf.reshape(probs[:, -1], (-1,1))], 1)
Z_pos = tf.matmul(pred_ord, positiveness)

with tf.name_scope("loss_function"):
    strong_loss = tf.reduce_mean(tf.square(Z - Y))
    ord_loss = -tf.log(tf.reduce_sum(tf.multiply(tf.nn.l2_normalize(Y_Pos,0),tf.nn.l2_normalize(Z_pos,0))))
    loss = (strong_loss + config.WORD_reg_param*ord_loss)/(1+config.WORD_reg_param)
tf.summary.scalar('loss', loss)
global_step = tf.Variable(0, name='global_step', trainable=False)

with tf.variable_scope("optimizer", reuse=tf.AUTO_REUSE):
    optimizer = tf.train.AdamOptimizer(config.WORD_learning_rate).minimize(loss, global_step)
#%%
print("TRAIN MODEL")
saver = tf.train.Saver()
merged_summary_op = tf.summary.merge_all()
with tf.Session() as sess:
    summary_writer = tf.summary.FileWriter(os.path.join(config.MODEL_DIR, "WORD"), sess.graph)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    for i in range(config.WORD_n_epochs):

        data_strong = train_data_strong[:,:-2]
        labels_strong = np.reshape(train_data_strong[:, -2], [-1, 1])
        data_weak = train_data_weak[:,:-2]

        feed_dict = {X:data_strong, Y:labels_strong, X_ord:data_weak, Y_Pos:true_pos}
        summary_str, _, loss_epoch, strong_loss_epoch, ord_loss_epoch = sess.run([merged_summary_op, optimizer, loss, strong_loss, ord_loss], feed_dict=feed_dict)
        summary_writer.add_summary(summary_str, global_step=global_step.eval())
        if not (i%100):
            print('Epoch:{0} Loss:{1:.4f}, Strong_loss:{2:.4f}, Ord_Loss:{3:.4f}'.format(i, loss_epoch, strong_loss_epoch, ord_loss_epoch))
    summary_writer.close()
    save_path = saver.save(sess, os.path.join(config.MODEL_DIR, "WORD", "model_imbalanced.ckpt"))
