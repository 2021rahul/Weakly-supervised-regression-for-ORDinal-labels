#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 10:16:25 2019

@author: ghosh128
"""

import sys
sys.path.append("../")
import os
import numpy as np
import config
import tensorflow as tf
from sklearn.metrics import mean_squared_error
from math import sqrt
tf.set_random_seed(1)
#%%
print("LOAD DATA")
test_data = np.load(os.path.join(config.NUMPY_DIR, "data_weak.npy"))
num_features = test_data.shape[-1] - 2
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
#%%
print("TEST MODEL")
saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, os.path.join(config.MODEL_DIR, "IMBALNCED", "OnlyStrong", "model.ckpt"))
    data = test_data[:,:-2]
    feed_dict = {X: data}
    preds = sess.run(Z, feed_dict=feed_dict)

labels = np.reshape(test_data[:, -2], [-1, 1])

k_RMSE = np.zeros((1,3))
for k in range(1,len(labels),10):
    indices = np.argsort(preds[:,0])[::-1]
    pred_top_k_rmse = sqrt(mean_squared_error(labels[indices[:k],0], preds[indices[:k],0]))
    print("Top K Root Mean Squared Error(Pred):", pred_top_k_rmse)
    indices = np.argsort(labels[:,0])[::-1]
    true_top_k_rmse = sqrt(mean_squared_error(labels[indices[:k],0], preds[indices[:k],0]))
    print("Top K Root Mean Squared Error(True):", true_top_k_rmse)
    GM_top_k_rmse = sqrt(pred_top_k_rmse*true_top_k_rmse)
    print("Top K Root Mean Squared Error(GM):", GM_top_k_rmse)
    k_RMSE = np.vstack((k_RMSE, np.reshape(np.array([pred_top_k_rmse, true_top_k_rmse, GM_top_k_rmse]), (1,-1))))

k_RMSE = k_RMSE[1:,:]
RESULT_DIR = os.path.join(config.RESULT_DIR, "IMBALNCED", "OnlyStrong")
if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)
np.save(os.path.join(RESULT_DIR, "k_RMSE"), k_RMSE)
