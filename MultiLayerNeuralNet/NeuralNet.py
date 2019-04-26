# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 13:23:16 2019

@author: bowye
"""

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
import pandas as pd
import time
import pickle

# =============================================================================
# 
# LOAD IN DATA
# 
# =============================================================================
#Load in training and test set
train_path = "../Data/UnderSampledHotEncoded/UnderSampledHotEncoded_TrainSet.csv"
train_set = pd.read_csv(train_path, index_col=False)
test_path = "../Data/UnderSampledHotEncoded/UnderSampledHotEncoded_TestSet.csv"
test_set = pd.read_csv(test_path, index_col=False)
columns=['x','y',
                   'dyn_prop_moving','dyn_prop_stationary','dyn_prop_oncoming','dyn_prop_stationary_candidate',
                   'dyn_prop_unknown','dyn_prop_crossing stationary','dyn_prop_crossing moving',
                   'rcs','vx_comp','vy_comp',
                   'ambig_state_invalid','ambig_state_ambiguous','ambig_state_staggered ramp','ambig_state_unambiguous',
                   'x_rms','y_rms',
                   'invalid_state_valid',
                   'invalid_state_valid_low_RCS',
                   'invalid_state_valid_azimuth_correction',
                   'invalid_state_valid_high_child_prob',
                   'invalid_state_valid_high_prob_50_artefact',
                   'invalid_state_valid_no_local_max',
                   'invalid_state_valid_high_artefact_prob',
                   'invalid_state_valid_above_95m',
                   'invalid_state_valid_high_multitarget_prob',
                   'False alarm <25%','False alarm 75%','False alarm 99.9%',
                   'vx_rms','vy_rms']

columnsSub=[
                   'rcs','vx_comp',
                   'ambig_state_invalid','ambig_state_ambiguous','ambig_state_staggered ramp','ambig_state_unambiguous',
                   'x_rms','y_rms',
                   'invalid_state_valid',
                   'invalid_state_valid_low_RCS',
                   'invalid_state_valid_azimuth_correction',
                   'invalid_state_valid_high_child_prob',
                   'invalid_state_valid_high_prob_50_artefact',
                   'invalid_state_valid_no_local_max',
                   'invalid_state_valid_high_artefact_prob',
                   'invalid_state_valid_above_95m',
                   'invalid_state_valid_high_multitarget_prob',
                   'False alarm <25%','False alarm 75%','False alarm 99.9%',
                   'vx_rms','vy_rms']

X_train=train_set[columnsSub].to_numpy()
y_train=train_set['BasicCategoryNum']

X_test=test_set[columnsSub].to_numpy()
y_test=test_set['BasicCategoryNum']
# =============================================================================
# 
# RUN MULTILAYER NEURAL NETWORK
# 
# =============================================================================
start = time.clock()
clf = MLPClassifier(solver='adam',
                   hidden_layer_sizes=(15,10), random_state=1, alpha=.001, activation='logistic')
clf.fit(X_train, y_train)
print (time.clock() - start)

filename = 'savedModels/MLP_10x5_logistic.sav'
pickle.dump(clf, open(filename, 'wb'))

predictedlTrainVals = clf.predict(X_train)
#Create Confusion Matrix for training set
NNTraincm=confusion_matrix(y_train,predictedlTrainVals)
plot_confusion_matrix(NNTraincm, normalize=True,
                      title='Confusion Matrix for 2 Layer Neural Netowrk with 15x10 Neurons on Training Set',
                      target_names=['Bicycle','Bus','Construction','Motorcycle','Pass. Veh.','Pedestrian','Tractor','Trailer'])

predictedlTestVals = clf.predict(X_test)
NNTestcm=confusion_matrix(y_test,predictedlTestVals)
plot_confusion_matrix(NNTestcm, normalize=True,
                      title='Confusion Matrix for 2 Layer Neural Netowrk with 15x10 Neurons on Test Set',
                      target_names=['Bicycle','Bus','Construction','Motorcycle','Pass. Veh.','Pedestrian','Tractor','Trailer'])
    
    