#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 19:37:47 2019

@author: ghosh128
"""

import os
#%% FILES INFO
TILE = "CentralEquatoria"
TILE_DIR = os.path.join("../DATA", TILE)
SENTINEL_DIR = os.path.join(TILE_DIR, "SENTINEL")
if not os.path.exists(SENTINEL_DIR):
    os.makedirs(SENTINEL_DIR)
NUMPY_DIR = os.path.join(TILE_DIR, "NUMPY")
if not os.path.exists(NUMPY_DIR):
    os.makedirs(NUMPY_DIR)
RESULT_DIR = os.path.join(TILE_DIR, "RESULT", "PATCH_CNN")
if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)
MODEL_DIR = os.path.join(TILE_DIR, "MODEL", "PATCH_CNN")
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
#%% DATA INFO

year = "2018"
resolution = 10
tile_size = 1000
pad = 0
n_bands = 10