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

#OnlyStrong_k_RMSE = np.load(os.path.join(config.RESULT_DIR[3:], "IMBALNCED", "OnlyStrong", "k_RMSE.npy"))
#plt.plot(OnlyStrong_k_RMSE[:upto,-1], c="Y")
#
#SSRManifold_k_RMSE = np.load(os.path.join(config.RESULT_DIR[3:], "IMBALNCED", "SSRManifold", "k_RMSE.npy"))
#plt.plot(SSRManifold_k_RMSE[:upto,-1], c="G")
#
#pairwiseReg_k_RMSE = np.load(os.path.join(config.RESULT_DIR[3:], "IMBALNCED", "pairwiseReg", "k_RMSE.npy"))
#plt.plot(pairwiseReg_k_RMSE[:upto,-1], c="B")
#
#WORD_k_RMSE = np.load(os.path.join(config.RESULT_DIR[3:], "IMBALNCED", "WORD", "k_RMSE_6.npy"))
#plt.plot(WORD_k_RMSE[:upto,-1], c="R")
#
#plt.show()
##%%
##OnlyStrong_k_RMSE = np.load(os.path.join(config.RESULT_DIR[3:], "IMBALNCED", "OnlyStrong", "k_RMSE.npy"))
##plt.plot(OnlyStrong_k_RMSE[:upto,-1], c="Black")
##
#WORD_k_RMSE_2 = np.load(os.path.join(config.RESULT_DIR[3:], "IMBALNCED", "WORD", "k_RMSE_2.npy"))
#plt.plot(WORD_k_RMSE_2[:upto,-1], c="Y")
#
#WORD_k_RMSE_3 = np.load(os.path.join(config.RESULT_DIR[3:], "IMBALNCED", "WORD", "k_RMSE_3.npy"))
#plt.plot(WORD_k_RMSE_3[:upto,-1], c="G")
#
#WORD_k_RMSE_4 = np.load(os.path.join(config.RESULT_DIR[3:], "IMBALNCED", "WORD", "k_RMSE_4.npy"))
#plt.plot(WORD_k_RMSE_4[:upto,-1], c="B")
#
#WORD_k_RMSE_6 = np.load(os.path.join(config.RESULT_DIR[3:], "IMBALNCED", "WORD", "k_RMSE_6.npy"))
#plt.plot(WORD_k_RMSE_6[:upto,-1], c="R")
#
#BALANCED_WORD_k_RMSE_6 = np.load(os.path.join(config.RESULT_DIR[3:], "BALNCED", "WORD", "k_RMSE_6.npy"))
#
#plt.ylim(0.1, 0.2)
#plt.show()
##%%
k = 130
#print(OnlyStrong_k_RMSE[k, :])
#print(SSRManifold_k_RMSE[k, :])
#print(pairwiseReg_k_RMSE[k, :])
BALANCED_WORD_k_RMSE_2 = np.load(os.path.join(config.RESULT_DIR[3:], "BALANCED", "WORD", "k_RMSE_2.npy"))
print(BALANCED_WORD_k_RMSE_2[k, :])
#print(WORD_k_RMSE_6[k, :])
#%%
shutil.rmtree("/".join(config.DATA_DIR.split("/")[:-1]))
#%%

OnlyStrong_k_RMSE = np.load(os.path.join(config.RESULT_DIR[3:], "IMBALANCED", "OnlyStrong", "k_RMSE.npy"))
plt.plot(OnlyStrong_k_RMSE[:upto,-1], c="Y")

COREG_k_RMSE_6 = np.load(os.path.join(config.RESULT_DIR[3:], "BALANCED", "COREG", "k_RMSE_6.npy"))
plt.plot(COREG_k_RMSE_6[:upto,-1], c="G")

pairwiseReg_k_RMSE = np.load(os.path.join(config.RESULT_DIR[3:], "IMBALANCED", "pairwiseReg", "k_RMSE.npy"))
plt.plot(pairwiseReg_k_RMSE[:upto,-1], c="Black")

BALANCED_WORD_k_RMSE_6 = np.load(os.path.join(config.RESULT_DIR[3:], "BALANCED", "WORD", "k_RMSE_6.npy"))
plt.plot(BALANCED_WORD_k_RMSE_6[:upto,-1], c="G")

WORD_k_RMSE_6 = np.load(os.path.join(config.RESULT_DIR[3:], "IMBALANCED", "WORD", "k_RMSE_6.npy"))
plt.plot(WORD_k_RMSE_6[:upto,-1], c="R")

plt.show()