#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 10:16:00 2019

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
num_features = train_data_strong.shape[-1] - 2
#%%
print("BUILD MODEL")
tf.reset_default_graph()
with tf.name_scope('data'):
    X = tf.placeholder(tf.float32, [None, num_features], name="inputs")
    Y = tf.placeholder(tf.float32, [None, 1], name="labels")

with tf.variable_scope("Variables", reuse=tf.AUTO_REUSE):
    W = tf.get_variable("W", [num_features, 1], initializer=tf.contrib.layers.xavier_initializer())
    b = tf.get_variable("b", [1], initializer=tf.zeros_initializer())

Z = tf.matmul(X, W, name="multiply_weights")
Z = tf.add(Z, b, name="add_bias")
Z = tf.sigmoid(Z)

with tf.name_scope("loss_function"):
    loss = tf.reduce_mean(tf.square(Z - Y))
tf.summary.scalar('loss', loss)
global_step = tf.Variable(0, name='global_step', trainable=False)

with tf.variable_scope("optimizer", reuse=tf.AUTO_REUSE):
    optimizer = tf.train.AdamOptimizer(config.OnlyStrong_learning_rate).minimize(loss, global_step)
#%%
print("TRAIN MODEL")
saver = tf.train.Saver()
merged_summary_op = tf.summary.merge_all()
with tf.Session() as sess:
    summary_writer = tf.summary.FileWriter(os.path.join(config.MODEL_DIR, "OnlyStrong"), sess.graph)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    for i in range(config.OnlyStrong_n_epochs):

        data_strong = train_data_strong[:,:-2]
        labels_strong = np.reshape(train_data_strong[:, -2], [-1, 1])

        feed_dict = {X:data_strong, Y:labels_strong}
        summary_str, _, loss_epoch = sess.run([merged_summary_op, optimizer, loss], feed_dict=feed_dict)
        summary_writer.add_summary(summary_str, global_step=global_step.eval())
        if not (i%100):
            print('Epoch {0}: Loss: {1}'.format(i, loss_epoch))
    summary_writer.close()
    save_path = saver.save(sess, os.path.join(config.MODEL_DIR, "OnlyStrong", "model.ckpt"))
