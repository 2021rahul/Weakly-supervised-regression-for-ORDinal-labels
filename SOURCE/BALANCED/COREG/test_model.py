#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 12:32:57 2019

@author: ghosh128
"""

import sys
sys.path.append("../")
import os
import numpy as np
import config
import tensorflow as tf
import matplotlib.pyplot as plt
tf.set_random_seed(1)
#%%
print("LOAD DATA")
test_data = np.load(os.path.join(config.NUMPY_DIR, "test_data.npy"))
#%%
print("BUILD MODEL")
num_features = test_data.shape[-1] - 1

tf.reset_default_graph()
with tf.name_scope('data'):
    X = tf.placeholder(tf.float32, [None, num_features], name="inputs")
    Y = tf.placeholder(tf.float32, [None, 1], name="labels")

with tf.variable_scope("Variables", reuse=tf.AUTO_REUSE):
    W = tf.get_variable("W", [num_features, 1], initializer=tf.contrib.layers.xavier_initializer())
    b = tf.get_variable("b", [1], initializer=tf.zeros_initializer())

Z = tf.matmul(X, W, name="multiply_weights")
Z = tf.add(Z, b, name="add_bias")

with tf.name_scope("loss_function"):
    strong_loss = tf.reduce_mean(tf.square(Z - Y))
    ord_loss = tf.reduce_mean(tf.square(Z - Y))
    loss = strong_loss + config.reg_param*ord_loss
tf.summary.scalar('loss', loss)
global_step = tf.Variable(0, name='global_step', trainable=False)

with tf.variable_scope("optimizer", reuse=tf.AUTO_REUSE):
    optimizer = tf.train.AdamOptimizer(config.learning_rate).minimize(loss, global_step)
#%%
print("TEST MODEL")
saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, os.path.join(config.MODEL_DIR, "BALNCED", "WORD", "model.ckpt"))
    data = test_data[:,:-1]
    labels = np.reshape(test_data[:, -1], [-1, 1])
    feed_dict = {X: data}
    preds = sess.run(Z, feed_dict=feed_dict)
plt.scatter(labels, preds)
plt.title('Actual vs Predicted plot')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.savefig(os.path.join(config.RESULT_DIR, "BALNCED", "WORD", "ActualvsPredicted.png"))
