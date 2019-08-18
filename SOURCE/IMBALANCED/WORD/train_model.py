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
train_data_strong = np.load(os.path.join(config.NUMPY_DIR, "data_strong_6.npy")).astype(np.float32)
train_data_weak = np.load(os.path.join(config.NUMPY_DIR, "data_weak_6.npy")).astype(np.float32)
theta = np.load(os.path.join(config.NUMPY_DIR, "theta_6.npy")).astype(np.float32)
positiveness = np.reshape(np.load(os.path.join(config.NUMPY_DIR, "positiveness_6.npy")).astype(np.float32), (-1,1))

true_pos = positiveness[train_data_weak[:, -1].astype("int")-1].astype(np.float32)
num_features = train_data_strong.shape[-1] - 2
#%%
print("BUILD MODEL")

tf.reset_default_graph()
with tf.name_scope('data'):
    X = tf.placeholder(tf.float32, [None, num_features], name="inputs")
    Y = tf.placeholder(tf.float32, [None, 1], name="labels")
    X_ord = tf.placeholder(tf.float32, [None, num_features], name="ordinal_inputs")
    Pos = tf.placeholder(tf.float32, [None, 1], name="true_positiveness")

with tf.variable_scope("Variables", reuse=tf.AUTO_REUSE):
    W = tf.get_variable("W", [num_features, 1], initializer=tf.contrib.layers.xavier_initializer())
    b = tf.get_variable("b", [1], initializer=tf.zeros_initializer())

Z = tf.matmul(X, W, name="multiply_weights")
Z = tf.add(Z, b, name="add_bias")
Z = tf.sigmoid(Z)

Z_ord = tf.matmul(X_ord, W, name="multiply_weights")
Z_ord = tf.add(Z_ord, b, name="add_bias")
Z_ord = tf.sigmoid(Z_ord)

probs = tf.multiply(config.s, tf.subtract(Z_ord, theta))
probs = tf.nn.sigmoid(probs)
pred_ord = tf.reshape(tf.subtract(1.0, probs[:,0]), (-1,1))
for i in range(1, positiveness.shape[0]-1):
    pred_ord = tf.concat([pred_ord, tf.reshape(probs[:, i-1]-probs[:, i], (-1,1))], 1)
pred_ord = tf.concat([pred_ord, tf.reshape(probs[:, -1], (-1,1))], 1)

est_pos = tf.matmul(pred_ord, positiveness)

with tf.name_scope("loss_function"):
    strong_loss = tf.reduce_mean(tf.square(Z - Y))
    ord_loss = -tf.log(tf.reduce_sum(tf.multiply(tf.nn.l2_normalize(true_pos,0),tf.nn.l2_normalize(est_pos,0))))
    loss = strong_loss + config.reg_param*ord_loss
tf.summary.scalar('loss', loss)
global_step = tf.Variable(0, name='global_step', trainable=False)

with tf.variable_scope("optimizer", reuse=tf.AUTO_REUSE):
    optimizer = tf.train.AdamOptimizer(config.learning_rate).minimize(loss, global_step)
#%%
print("TRAIN MODEL")
saver = tf.train.Saver()
merged_summary_op = tf.summary.merge_all()
with tf.Session() as sess:
    summary_writer = tf.summary.FileWriter(os.path.join(config.MODEL_DIR, "IMBALNCED", "WORD"), sess.graph)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    for i in range(config.n_epochs):

        data_strong = train_data_strong[:,:-2]
        labels_strong = np.reshape(train_data_strong[:, -2], [-1, 1])
        data_weak = train_data_weak[:,:-2]

        feed_dict = {X:data_strong, Y:labels_strong, X_ord:data_weak}
        summary_str, _, loss_epoch, strong_loss_epoch, ord_loss_epoch = sess.run([merged_summary_op, optimizer, loss, strong_loss, ord_loss], feed_dict=feed_dict)
        summary_writer.add_summary(summary_str, global_step=global_step.eval())
        if not (i%100):
            print('Epoch {0}: Loss: {1}, Strong_loss: {2}, Ord_Loss: {3}'.format(i, loss_epoch, strong_loss_epoch, ord_loss_epoch))
    summary_writer.close()
    save_path = saver.save(sess, os.path.join(config.MODEL_DIR, "IMBALNCED", "WORD", "model.ckpt"))

