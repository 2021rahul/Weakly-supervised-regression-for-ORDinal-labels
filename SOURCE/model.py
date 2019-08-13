#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 19:51:39 2019

@author: ghosh128
"""

import os
import numpy as np
import config
import tensorflow as tf
import math
tf.set_random_seed(1)
#%%
print("BUILD MODEL")
num_features = 10
learning_rate = 0.0001

tf.reset_default_graph()
with tf.name_scope('data'):
    X = tf.placeholder(tf.float32, [None, num_features], name="inputs")
    Y = tf.placeholder(tf.float32, [None, 1], name="labels")

with tf.variable_scope("Variables", reuse=tf.AUTO_REUSE):
    W = tf.get_variable("W", [num_features, 1], initializer=tf.contrib.layers.xavier_initializer())
    b = tf.get_variable("b", [1], initializer=tf.zeros_initializer())

Z = tf.matmul(X, W, name="multiply_weights")
Z = tf.add(Z, b, name="add_bias")
Z = tf.nn.sigmoid(Z)

with tf.name_scope("loss_function"):
    sig_z_weak = tf.multiply(s, tf.subtract(Z_weak, gamma_vals))
    y_hat_weak = tf.nn.sigmoid(sig_z_weak)
    weak_acc = tf.reduce_mean(tf.multiply(A, y_hat_weak) + tf.multiply(1-A, 1-y_hat_weak), axis=0)

    sig_z_strong = tf.multiply(s, tf.subtract(Z_strong, gamma_vals))
    y_hat_strong = tf.nn.sigmoid(sig_z_strong)
    strong_acc = tf.reduce_mean(tf.multiply(Y, y_hat_strong) + tf.multiply(1-Y, 1-y_hat_strong), axis=0)

#    gamma_acc = (strong_acc + lam*weak_acc)/(1+lam)
    gamma_acc = weak_acc
#    gamma_acc = strong_acc
    softmax_acc = tf.pow(x, gamma_acc)/tf.reduce_sum(tf.pow(x, gamma_acc))
    loss = -tf.tensordot(softmax_acc, gamma_acc, axes=1)
tf.summary.scalar('loss', loss)
global_step = tf.Variable(0, name='global_step', trainable=False)

with tf.variable_scope("optimizer", reuse=tf.AUTO_REUSE):
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step)
# %%
print("GENERATE MAP")
saver = tf.train.Saver()
pred_map = np.zeros(data_img.shape[:-1])
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    saver.restore(sess, os.path.join(MODEL_DIR, "model.ckpt"))
    for i in range(low, data_img.shape[0]-high):
        if not i%100:
            print(i)
        data_batch = []
        label_batch = []
        for j in range(low, data_img.shape[1]-high):
            data_batch.append(data_img[i-low:i+high, j-low:j+high, :])
        data_batch = np.asarray(data_batch, dtype=np.float32)
        feed_dict = {X: data_batch}
        pred_map[i,low:data_img.shape[1]-high] = np.reshape(sess.run(prediction, feed_dict=feed_dict), (-1))

np.save(os.path.join(RESULT_DIR, "Prob_Map"), pred_map)
# %%
pred_map = np.load(os.path.join(RESULT_DIR, "Prob_Map.npy"))
threshold = 0
if threshold:
    pred_map[pred_map > 0.01*threshold] = 1
    pred_map[pred_map <= 0.01*threshold] = 0
    filename = os.path.join(RESULT_DIR, "Prob_Map_"+str(threshold)+".tif")
else:
    filename = os.path.join(RESULT_DIR, "Prob_Map.tif")

tif_with_meta = gdal.Open(os.path.join(SENTINEL_DIR, 'Image.tif'), gdalconst.GA_ReadOnly)
gt = tif_with_meta.GetGeoTransform()
driver = gdal.GetDriverByName("GTiff")
dest = driver.Create(filename, data_img.shape[1], data_img.shape[0], 1, gdal.GDT_Float64)
dest.GetRasterBand(1).WriteArray(pred_map)
dest.SetGeoTransform(gt)
wkt = tif_with_meta.GetProjection()
srs = osr.SpatialReference()
srs.ImportFromWkt(wkt)
dest.SetProjection(srs.ExportToWkt())
dest = None
