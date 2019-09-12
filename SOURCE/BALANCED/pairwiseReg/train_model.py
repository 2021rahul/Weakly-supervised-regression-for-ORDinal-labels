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
import random
import tensorflow as tf
tf.set_random_seed(1)
#%%
print("LOAD DATA")
train_data_strong = np.load(os.path.join(config.NUMPY_DIR, "train_data_strong.npy"))
train_data_weak = np.load(os.path.join(config.NUMPY_DIR, "train_data_weak.npy"))

num_features = train_data_strong.shape[-1] - 2
num_levels = len(np.unique(train_data_strong[:,-1]))

index_dict = {}
for i in range(num_levels):
    print("Number of entries of level", i+1, len(np.where(train_data_weak[:,-1]==i+1)[0]))
    index_dict[i+1] = np.where(train_data_weak[:,-1]==i+1)[0]
#%%
print("BUILD MODEL")
tf.reset_default_graph()
with tf.name_scope('data'):
    X = tf.placeholder(tf.float32, [None, num_features], name="inputs")
    Y = tf.placeholder(tf.float32, [None, 1], name="labels")
    Xhigh = tf.placeholder(tf.float32, [None, num_features], name="inputs_high")
    Xlow = tf.placeholder(tf.float32, [None, num_features], name="inputs_low")

with tf.variable_scope("Variables", reuse=tf.AUTO_REUSE):
    W = tf.get_variable("W", [num_features, 1], initializer=tf.contrib.layers.xavier_initializer())
    b = tf.get_variable("b", [1], initializer=tf.zeros_initializer())

Z = tf.matmul(X, W, name="multiply_weights")
Z = tf.add(Z, b, name="add_bias")
Z = tf.nn.sigmoid(Z)

Zhigh = tf.matmul(Xhigh, W, name="multiply_weights")
Zhigh = tf.add(Zhigh, b, name="add_bias")
Zhigh = tf.nn.sigmoid(Zhigh)

Zlow = tf.matmul(Xlow, W, name="multiply_weights")
Zlow = tf.add(Zlow, b, name="add_bias")
Zlow = tf.nn.sigmoid(Zlow)

with tf.name_scope("loss_function"):
    strong_loss = tf.reduce_mean(tf.square(Z - Y))
    pair_loss = tf.reduce_mean(tf.divide(1, tf.nn.sigmoid(Zhigh-Zlow)))
    loss = strong_loss + config.pairwiseReg_reg_param1*tf.nn.l2_loss(W) + config.pairwiseReg_reg_param2*pair_loss
tf.summary.scalar('loss', loss)
global_step = tf.Variable(0, name='global_step', trainable=False)

with tf.variable_scope("optimizer", reuse=tf.AUTO_REUSE):
    optimizer = tf.train.AdamOptimizer(config.pairwiseReg_learning_rate).minimize(loss, global_step)
#%%
print("TRAIN MODEL")
saver = tf.train.Saver()
merged_summary_op = tf.summary.merge_all()
with tf.Session() as sess:
    summary_writer = tf.summary.FileWriter(os.path.join(config.MODEL_DIR, "pairwiseReg"), sess.graph)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    for i in range(config.pairwiseReg_n_epochs):
        data = train_data_strong[:,:-2]
        labels = np.reshape(train_data_strong[:, -2], [-1, 1])
        xhigh = np.zeros((1,num_features))
        xlow = np.zeros((1,num_features))
        for level in range(1,num_levels):
            high_index = index_dict[level+1][random.sample(range(len(index_dict[level+1])), len(index_dict[level+1]))]
            xhigh = np.vstack((xhigh, train_data_weak[high_index,:-2]))
            low_index = index_dict[level][random.sample(range(len(index_dict[level])), len(index_dict[level]))]
            xlow = np.vstack((xlow, train_data_weak[low_index,:-2]))
        xhigh = xhigh[1:,:]
        xlow = xlow[1:,:]
        feed_dict = {X:data, Y:labels, Xhigh:xhigh, Xlow:xlow}
        summary_str, _, loss_epoch = sess.run([merged_summary_op, optimizer, loss], feed_dict=feed_dict)
        summary_writer.add_summary(summary_str, global_step=global_step.eval())
        if not (i%100):
            print('Epoch: {0} Loss: {1}'.format(i, loss_epoch))
    summary_writer.close()
    save_path = saver.save(sess, os.path.join(config.MODEL_DIR, "pairwiseReg", "model.ckpt"))
