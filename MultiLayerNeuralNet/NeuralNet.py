# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 13:23:16 2019

@author: bowye
"""

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
import pandas as pd
# =============================================================================
# 
# LOAD IN DATA
# 
# =============================================================================
train_path = "../Data/UnderSampled/UnderSampled_TrainSet.csv"
train_set = pd.read_csv(train_path, index_col=False)
test_path = "../Data/UnderSampled/UnderSampled_TestSet.csv"
test_set = pd.read_csv(test_path, index_col=False)

X_train=train_set[['dyn_prop','rcs','vx_comp','ambig_state','x_rms','y_rms','invalid_state','vx_rms','vy_rms']].to_numpy()
y_train=train_set['BasicCategoryNum']

X_test=test_set[['dyn_prop','rcs','vx_comp','ambig_state','x_rms','y_rms','invalid_state','vx_rms','vy_rms']].to_numpy()
y_test=test_set['BasicCategoryNum']
# =============================================================================
# 
# RUN MULTILAYER NEURAL NETWORK
# 
# =============================================================================
clf = MLPClassifier(solver='adam',
                   hidden_layer_sizes=(5, 2), random_state=1, alpha=.001, activation='logistic')
clf.fit(X_train, y_train)

predictedlTrainVals = clf.predict(X_train)
#Create Confusion Matrix for training set
NNTraincm=confusion_matrix(y_train,predictedlTrainVals)
plot_confusion_matrix(NNTraincm, normalize=True,
                      title='Confusion Matrix for Logistic Regression with L2 Penalty on Training Set',
                      target_names=['Bicycle','Bus','Construction','Motorcycle','Pass. Veh.','Pedestrian','Tractor','Trailer'])

predictedlTestVals = clf.predict(X_test)
NNTestcm=confusion_matrix(y_test,predictedlTestVals)
plot_confusion_matrix(NNTestcm, normalize=True,
                      title='Confusion Matrix for Logistic Regression with L1 Penalty on Test Set',
                      target_names=['Bicycle','Bus','Construction','Motorcycle','Pass. Veh.','Pedestrian','Tractor','Trailer'])
    
    