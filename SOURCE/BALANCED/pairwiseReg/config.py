#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 19:37:47 2019

@author: ghosh128
"""

import os
#%% FILES INFO
DATASET = "BlogFeedback"
DATASET_DIR = os.path.join("../../../DATA", DATASET)
DATA_DIR = os.path.join(DATASET_DIR, "DATA")
NUMPY_DIR = os.path.join(DATASET_DIR, "NUMPY")
if not os.path.exists(NUMPY_DIR):
    os.makedirs(NUMPY_DIR)
RESULT_DIR = os.path.join(DATASET_DIR, "RESULT")
if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)
MODEL_DIR = os.path.join(DATASET_DIR, "MODEL")
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
#%% DATA INFO
year = "2018"
resolution = 10
tile_size = 1000
pad = 0
n_bands = 10

#%% TRAIN INFO
reg_param = 1
learning_rate = 0.0001
n_epochs = 10000