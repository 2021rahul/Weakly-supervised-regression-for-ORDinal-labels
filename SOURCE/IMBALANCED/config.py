#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 19:37:47 2019

@author: ghosh128
"""

import os
#%% FILES INFO
DATASET = "D4"
k = 3000
DATA_DIR = os.path.join("../../../DATA", DATASET)
NUMPY_DIR = os.path.join(DATA_DIR, "NUMPY")
RESULT_DIR = os.path.join(DATA_DIR, "RESULT")
MODEL_DIR = os.path.join(DATA_DIR, "MODEL")

#%% DATA INFO
num_strong = 20
num_weak = 10000
num_validate = num_strong
num_test = num_weak

#%% TRAIN INFO

# OnlyStrong
OnlyStrong_learning_rate = 0.001
OnlyStrong_n_epochs = 10000

# SSRManifold
SSRManifold_reg_param = 1.0
SSRManifold_learning_rate = 0.001
SSRManifold_n_epochs = 10000

# pairwiseReg
pairwiseReg_reg_param1 = 1.0
pairwiseReg_reg_param2 = 1.0
pairwiseReg_learning_rate = 0.01
pairwiseReg_n_epochs = 10000

# WORD
WORD_reg_param = 1.0
WORD_learning_rate = 0.01
WORD_n_epochs = 10000
WORD_s = 1000.0
