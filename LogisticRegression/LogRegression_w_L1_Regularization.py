# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 16:10:23 2019

@author: bowye
"""

import pandas as pd 
import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import confusion_matrix
from sklearn import model_selection
import pickle
import time

#Load in cleansed radar points data frame
train_path = "../Data/data_train.csv"
test_path = "../Data/data_test.csv"

train_set = pd.read_csv(train_path, index_col=False)
test_set = pd.read_csv(test_path, index_col=False)

train_set['BasicCategory'].value_counts()
test_set['BasicCategory'].value_counts()
# =============================================================================
# 
# LOGISTIC REGRESSION
#
# =============================================================================
X=train_set[['x','y','dyn_prop','rcs','vx_comp','vy_comp','ambig_state','x_rms','y_rms','invalid_state','pdh0','vx_rms','vy_rms']].to_numpy()
y=train_set['BasicCategoryNum']

#Actualy Logistic Regression Model Run
start = time.clock()
generatedLogReg = LogisticRegressionCV(cv=4,penalty='l1', multi_class='ovr',solver='saga',n_jobs=5,class_weight ='balanced').fit(X,y)
print (time.clock() - start)
filename = 'LogRegCV_4Folds_OVR_L1_balanced.sav'
pickle.dump(generatedLogReg, open(filename, 'wb')) #export model so you don't need to rerun it and wait forever

# =============================================================================
# 
# READ IN DATA SINCE IT HAS ALREADY BEEN RAN
# 
# =============================================================================
#read in the model since it was already ran
filename='savedModels/LogRegCV_10Folds.sav'
importLogReg = pickle.load(open(filename, 'rb'))

X_test=test_set[['x','y','dyn_prop','rcs','vx_comp','vy_comp','ambig_state','x_rms','y_rms','invalid_state','pdh0','vx_rms','vy_rms']].to_numpy()
predictedTestVals = importLogReg.predict(X_test)
trueTestVals=test_set['BasicCategoryNum']

confusion_matrix(trueTestVals,predictedTestVals)
 