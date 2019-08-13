#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 19:45:56 2019

@author: ghosh128
"""

import os
import config
import pandas as pd

see = pd.read_csv(os.path.join(config.DATA_DIR, "blogData_train.csv"), header=None)
vals = see.values
print(vals.shape[-1])
#data = vals[:,:-1]
#labels = vals[:,-1]
