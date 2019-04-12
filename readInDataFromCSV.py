# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 18:46:25 2019

@author: bowye
"""
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

path = "Data/labelledRadarPoints.csv"

pointData = pd.read_csv(path, index_col=False)

