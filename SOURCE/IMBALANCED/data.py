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
from scipy import optimize
#%%
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
#bins = [0.1, 0.3, 0.5]
#bins = [0.3, 0.6]
#bins = [0.5]
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
np.save(os.path.join(config.NUMPY_DIR[3:], "data_strong_"+str(len(bins)+1)), data_strong)
np.save(os.path.join(config.NUMPY_DIR[3:], "data_weak_"+str(len(bins)+1)), data_weak)
#%%
#print("READ MAT DATA")
#mat_data = {}
#with h5py.File(os.path.join(config.DATA_DIR[3:], config.DATASET+'.mat'), 'r') as f:
#    for k, v in f.items():
#        mat_data[k] = np.array(v)
#
#data_strong = np.concatenate((np.transpose(mat_data['x_s']), np.transpose(mat_data['y_s']), np.transpose(mat_data['o_s'])), axis=1)
#np.save(os.path.join(config.NUMPY_DIR[3:], "data_strong"), data_strong)
#
#data_weak = np.concatenate((np.transpose(mat_data['x_w_b']), np.transpose(mat_data['y_w_b']), np.transpose(mat_data['o_w_b'])), axis=1)
#np.save(os.path.join(config.NUMPY_DIR[3:], "data_weak"), data_weak)
#%%
print("COMPUTE THETA")
O_s = data_strong[:,-1]
Y_s = data_strong[:,-2]
N_s = len(O_s)
num_levels = len(np.unique(O_s))

#Inequality 1 : theta_l_i - \xi_l_i <= y_i
A1 = np.zeros((N_s, 2*N_s+(num_levels-1)))
A1[list(range(N_s)), list(range(N_s))] = -1
A1[list(np.where(O_s>1)[0]), list(2*N_s + O_s[np.where(O_s>1)[0]].astype("int") -1 -1)] = 1
b1 = data_strong[:,-2]
b1 = np.reshape(b1, (-1,1))

#Inequality 2 : -theta_u_i - \xi_u_i <= -y_i
A2 = np.zeros((N_s, 2*N_s+(num_levels-1)))
A2[list(range(N_s)), list(range(N_s, N_s + N_s))] = -1
A2[list(np.where(O_s<num_levels)[0]), list(2*N_s + O_s[np.where(O_s<num_levels)[0]].astype("int") -1)] = -1
b2 = -Y_s
b2[np.where(O_s==num_levels)] += 1
b2 = np.reshape(b2, (-1,1))

#Inequality 3 : - \xi_l_i, -\xi_u_i <=0
A3 = np.zeros((2*N_s, 2*N_s+(num_levels-1)))
A3[list(range(2*N_s)), list(range(2*N_s))] = -1
b3 = np.zeros((2*N_s,1))
b3 = np.reshape(b3, (-1,1))

#Inequality 4 : -theta_k <=0 and theta_k <=1
A4 = np.zeros((2*(num_levels-1), 2*N_s+(num_levels-1)))
A4[list(range(num_levels-1)), list(range(2*N_s, 2*N_s+num_levels-1))] = -1
A4[list(range(num_levels-1, num_levels-1+num_levels-1)), list(range(2*N_s, 2*N_s+num_levels-1))] = 1
b4 = np.concatenate((np.zeros((num_levels-1,1)), np.ones((num_levels-1,1))))
b4 = np.reshape(b4, (-1,1))

f = np.concatenate((np.ones((2*N_s,1)), np.zeros((num_levels-1,1))))
A = np.concatenate((A1, A2, A3, A4))
b = np.concatenate((b1, b2, b3, b4))
x = optimize.linprog(c=f, A_ub=A, b_ub=b)
x = x["x"]
slack_var_lower = x[:N_s]
slack_var_upper = x[N_s:2*N_s]
theta = x[2*N_s:2*N_s+num_levels-1]

np.save(os.path.join(config.NUMPY_DIR[3:], "theta_"+str(len(bins)+1)), theta)
#%%
print("COMPUTE POSITIVENESS")
positiveness = np.zeros(num_levels)
for i in np.unique(O_s):
    positiveness[int(i-1)] = np.mean(Y_s[np.where(O_s == i)[0]])

np.save(os.path.join(config.NUMPY_DIR[3:], "positiveness_"+str(len(bins)+1)), positiveness)
#%%
shutil.rmtree("/".join(config.DATA_DIR.split("/")[:-1]))
#%%
print(len(np.where(O==1)[0]))
print(len(np.where(O==2)[0]))
print(len(np.where(O==3)[0]))
print(len(np.where(O==4)[0]))
print(len(np.where(O==5)[0]))
print(len(np.where(O==6)[0]))
#%%
from scipy import io

strong_data = np.load(os.path.join(config.NUMPY_DIR[3:], "data_strong_6.npy"))
weak_data = np.load(os.path.join(config.NUMPY_DIR[3:], "data_weak_6.npy"))

scipy.io.savemat(os.path.join(config.NUMPY_DIR[3:],'strong_data.mat'), {'strong_data': strong_data})
scipy.io.savemat(os.path.join(config.NUMPY_DIR[3:],'weak_data.mat'), {'weak_data': weak_data})