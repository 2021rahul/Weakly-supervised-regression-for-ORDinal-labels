#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 18:03:50 2019

@author: ghosh128
"""

import os
import shutil
import numpy as np
import config
import matplotlib.pyplot as plt

upto = 1000

WORD_k_RMSE = np.load(os.path.join(config.RESULT_DIR[3:], "IMBALNCED", "WORD", "k_RMSE.npy"))
plt.plot(WORD_k_RMSE[:upto,-1], c="G")

pairwiseReg_k_RMSE = np.load(os.path.join(config.RESULT_DIR[3:], "IMBALNCED", "pairwiseReg", "k_RMSE.npy"))
plt.plot(pairwiseReg_k_RMSE[:upto,-1], c="R")

plt.show()
#%%
shutil.rmtree("/".join(config.DATA_DIR.split("/")[:-1]))
