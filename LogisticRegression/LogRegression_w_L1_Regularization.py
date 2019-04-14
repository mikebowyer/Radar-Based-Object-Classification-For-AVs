# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 16:10:23 2019

@author: bowye
"""

import pandas as pd 
import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn import model_selection
import pickle

#Load in cleansed radar points data frame
path = "Data/CleansedRadarPoints.csv"
pointDF = pd.read_csv(path, index_col=False)

# =============================================================================
# 
# LOGISTIC REGRESSION
#
# =============================================================================
X= pointDF[['x','y','dyn_prop','rcs','vx_comp','vy_comp','ambig_state','x_rms','y_rms','invalid_state','pdh0','vx_rms','vy_rms']].to_numpy()

pointDF["BasicCategory"] = pointDF["BasicCategory"].astype('category')
pointDF.dtypes
pointDF["BasicCategoryNum"] = pointDF["BasicCategory"].cat.codes
pointDF.head()

y=pointDF[['BasicCategoryNum']]

#Actualy Logistic Regression Model Run
#logReg = LogisticRegressionCV(cv=10,penalty='l1', multi_class='multinomial',solver='saga',n_jobs=4).fit(X,y)
filename = 'LogRegCV_10Folds.sav'
#pickle.dump(logReg, open(filename, 'wb')) #export model so you don't need to rerun it and wait forever

# =============================================================================
# 
# READ IN DATA SINCE IT HAS ALREADY BEEN RAN
# 
# =============================================================================
#read in the model since it was already ran
logReg = pickle.load(open(filename, 'rb'))
print("training score : %.3f " % (logReg.score(X, y)))
coef = logReg.coef_


