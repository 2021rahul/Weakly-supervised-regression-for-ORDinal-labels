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
from scipy import io
from sklearn.metrics import mean_squared_error
from math import sqrt
#%%
print("LOAD DATA")
test_data = np.load(os.path.join(config.NUMPY_DIR, "data_weak_6.npy"))
preds = io.loadmat(os.path.join(config.RESULT_DIR, "IMBALANCED", "COREG", "Y.mat"))["y"]

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
RESULT_DIR = os.path.join(config.RESULT_DIR, "IMBALANCED", "COREG")
if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)
np.save(os.path.join(RESULT_DIR, "k_RMSE_6"), k_RMSE)
