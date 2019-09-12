#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 19:45:56 2019

@author: ghosh128
"""

import os
import shutil
import config
import numpy as np
import pandas as pd
import h5py
import random

import matplotlib.pyplot as plt
file_pos = 3
if not os.path.exists(config.NUMPY_DIR[file_pos:]):
    os.makedirs(config.NUMPY_DIR[file_pos:])
if not os.path.exists(config.RESULT_DIR[file_pos:]):
    os.makedirs(config.RESULT_DIR[file_pos:])
if not os.path.exists(config.MODEL_DIR[file_pos:]):
    os.makedirs(config.MODEL_DIR[file_pos:])
#%%
print("READ MAT DATA")
if config.DATASET == "D6":
    data = pd.read_csv(os.path.join(config.DATA_DIR[3:], 'train.csv'))
    data = data.values

    X = data[:,:-1]
    for i in range(X.shape[1]):
        X[:,i] = (X[:,i]-np.mean(X[:,i]))/np.std(X[:,i])

    Y = np.reshape(data[:,-1], (-1,1))
    min_val = min(Y)
    max_val = max(Y)
    Y = (Y-min_val)/(max_val - min_val)

    O = np.zeros((Y.shape))
    bins = [0.1, 0.3, 0.45, 0.5, 0.6]
    level = 1
    O[np.where(Y<bins[0])[0],0] = level
    level = level+1
    for i in range(len(bins)-1):
        O[np.where(np.logical_and(Y>=bins[i], Y<bins[i+1]))[0], 0] = level
        level = level + 1
    O[np.where(Y>=bins[-1])[0], 0] = level

    num_strong = 10
    data_strong = np.zeros((1, X.shape[1]+2))
    data_weak = np.zeros((1, X.shape[1]+2))
    for i in np.unique(O):
        index = np.where(O==i)[0]
        index = index[random.sample(range(len(index)), len(index))]
        data_strong = np.vstack((data_strong, np.concatenate((X[index[:num_strong],:],Y[index[:num_strong],:],O[index[:num_strong],:]), axis = 1)))
        data_weak = np.vstack((data_weak, np.concatenate((X[index[num_strong:],:],Y[index[num_strong:],:],O[index[num_strong:],:]), axis = 1)))
    data_strong = data_strong[1:,:]
    data_weak = data_weak[1:,:]

    print("DATA WEAK SIZE", data_weak.shape)
    print("WEAK LABELS:", np.unique(data_weak[:,-1]))

    train_data_weak = np.zeros((1,data_weak.shape[1]))
    test_data = np.zeros((1,data_weak.shape[1]))
    for weak_labels in np.unique(data_weak[:,-1]):
        index_labels = np.where(data_weak[:,-1] == weak_labels)[0].astype(int)
        index = random.sample(range(0, len(index_labels)), len(index_labels))
        index_labels = index_labels[index]
        total = len(index_labels)
        train_data_weak = np.concatenate((train_data_weak, data_weak[index_labels[:int(total/2)], :]))
        test_data = np.concatenate((test_data, data_weak[index_labels[:int(total/2)], :]))
    train_data_weak = train_data_weak[1:,:]
    test_data = test_data[1:,:]

    train_data_strong = data_strong
    validate_data = train_data_weak[random.sample(range(0, len(train_data_weak)), config.num_validate),:]
else:
    mat_data = {}
    with h5py.File(os.path.join(config.DATA_DIR[3:], config.DATASET+'.mat'), 'r') as f:
       for k, v in f.items():
           mat_data[k] = np.array(v)

    data_weak = np.concatenate((np.transpose(mat_data['x_w_b']), np.transpose(mat_data['y_w_b']), np.transpose(mat_data['o_w_b'])), axis=1)
    print("DATA WEAK SIZE", data_weak.shape)
    print("WEAK LABELS:", np.unique(data_weak[:,-1]))

    train_data_weak = np.zeros((1,data_weak.shape[1]))
    test_data = np.zeros((1,data_weak.shape[1]))
    for weak_labels in np.unique(data_weak[:,-1]):
        index_labels = np.where(data_weak[:,-1] == weak_labels)[0].astype(int)
        index = random.sample(range(0, len(index_labels)), len(index_labels))
        index_labels = index_labels[index]
        total = len(index_labels)
        train_data_weak = np.concatenate((train_data_weak, data_weak[index_labels[:int(total/2)], :]))
        test_data = np.concatenate((test_data, data_weak[index_labels[:int(total/2)], :]))
    train_data_weak = train_data_weak[1:,:]
    test_data = test_data[1:,:]

    train_data_strong = np.concatenate((np.transpose(mat_data['x_s']), np.transpose(mat_data['y_s']), np.transpose(mat_data['o_s'])), axis=1)
    validate_data = train_data_weak[random.sample(range(0, len(train_data_weak)), config.num_validate),:]

print("TRAIN DATA STRONG SIZE", train_data_strong.shape)
print("TRAIN DATA WEAK SIZE", train_data_weak.shape)
print("VALIDATE DATA SIZE", validate_data.shape)
print("TEST DATA SIZE", test_data.shape)

np.save(os.path.join(config.NUMPY_DIR[file_pos:], "train_data_strong"), train_data_strong)
np.save(os.path.join(config.NUMPY_DIR[file_pos:], "train_data_weak"), train_data_weak)
np.save(os.path.join(config.NUMPY_DIR[file_pos:], "validate_data"), validate_data)
np.save(os.path.join(config.NUMPY_DIR[file_pos:], "test_data"), test_data)
