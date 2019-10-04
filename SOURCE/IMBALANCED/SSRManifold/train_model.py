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
tf.set_random_seed(1)
#%%
print("LOAD DATA")
train_data_strong = np.load(os.path.join(config.NUMPY_DIR, "train_data_strong.npy")).astype(np.float32)
train_data_weak = np.load(os.path.join(config.NUMPY_DIR, "train_data_weak.npy")).astype(np.float32)

num_features = train_data_strong.shape[-1] - 2
num_levels = len(np.unique(train_data_strong[:,-1]))
#%%
print("BUILD MODEL")
config.SSRManifold_batch_size = 1000
tf.reset_default_graph()
with tf.name_scope('data'):
    X = tf.placeholder(tf.float32, [None, num_features], name="inputs")
    X_unlabeled = tf.placeholder(tf.float32, [None, num_features], name="inputs_unlabeled")
    Y = tf.placeholder(tf.float32, [None, 1], name="labels")

with tf.variable_scope("Variables", reuse=tf.AUTO_REUSE):
    W = tf.get_variable("W", [num_features, 1], initializer=tf.contrib.layers.xavier_initializer())
    b = tf.get_variable("b", [1], initializer=tf.zeros_initializer())

Z = tf.matmul(X, W, name="multiply_weights")
Z = tf.add(Z, b, name="add_bias")
Z = tf.nn.sigmoid(Z)

Z_unlabeled = tf.matmul(X_unlabeled, W, name="multiply_weights")
Z_unlabeled = tf.add(Z_unlabeled, b, name="add_bias")
Z_unlabeled = tf.nn.sigmoid(Z_unlabeled)

with tf.name_scope("loss_function"):
    squared_loss = tf.reduce_mean(tf.square(Z - Y))
    fx_diff = tf.square(tf.subtract(Z_unlabeled, tf.transpose(Z_unlabeled)))
    r = tf.reshape(tf.reduce_sum(X_unlabeled*X_unlabeled, 1), [-1, 1])
    x_dist = r - 2*tf.matmul(X_unlabeled, tf.transpose(X_unlabeled)) + tf.transpose(r)
    corr_loss = -tf.reduce_mean(tf.contrib.metrics.streaming_pearson_correlation(tf.reshape(fx_diff, [-1,1]), tf.reshape(x_dist, [-1,1]))[1])
    loss = (squared_loss + config.SSRManifold_reg_param*corr_loss)/(1+config.SSRManifold_reg_param)
tf.summary.scalar('loss', loss)

global_step = tf.Variable(0, name='global_step', trainable=False)

with tf.variable_scope("optimizer", reuse=tf.AUTO_REUSE):
    optimizer = tf.train.AdamOptimizer(config.SSRManifold_learning_rate).minimize(loss, global_step)
#%%
print("TRAIN MODEL")
saver = tf.train.Saver()
merged_summary_op = tf.summary.merge_all()
with tf.Session() as sess:
    summary_writer = tf.summary.FileWriter(os.path.join(config.MODEL_DIR, "SSRManifold"), sess.graph)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    k=0
    for i in range(config.SSRManifold_n_epochs):
        data = train_data_strong[:,:-2]
        labels = np.reshape(train_data_strong[:, -2], [-1, 1])

        if k*config.SSRManifold_batch_size>len(train_data_weak) or (k+1)*config.SSRManifold_batch_size>len(train_data_weak):
            k = 0
        data_batch = train_data_weak[(k*config.SSRManifold_batch_size)%len(train_data_weak):((k+1)*config.SSRManifold_batch_size)%len(train_data_weak), :]
        data_unlabeled = data_batch[:, :-2]

        feed_dict = {X:data, Y:labels, X_unlabeled:data_unlabeled}
        summary_str, _, loss_epoch = sess.run([merged_summary_op, optimizer, loss], feed_dict=feed_dict)
        summary_writer.add_summary(summary_str, global_step=global_step.eval())
        if not (i%100):
            print('Epoch:{0} Loss:{1}'.format(i, loss_epoch))
    summary_writer.close()
    save_path = saver.save(sess, os.path.join(config.MODEL_DIR, "SSRManifold", "model.ckpt"))
