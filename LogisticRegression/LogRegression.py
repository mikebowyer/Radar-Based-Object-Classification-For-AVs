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
import matplotlib.pyplot as plt
from imblearn.combine import SMOTETomek
import pickle
import time

#Load in training and test set
train_path = "../Data/UnderSampledHotEncoded/UnderSampledHotEncoded_TrainSet.csv"
train_set = pd.read_csv(train_path, index_col=False)
test_path = "../Data/UnderSampledHotEncoded/UnderSampledHotEncoded_TestSet.csv"
test_set = pd.read_csv(test_path, index_col=False)

print(train_set['BasicCategory'].value_counts())
print(test_set['BasicCategory'].value_counts())
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

X_train=train_set[columns].to_numpy()
y_train=train_set['BasicCategoryNum']
X_test=test_set[columns].to_numpy()
y_test=test_set['BasicCategoryNum']
# =============================================================================
# 
# LOGISTIC REGRESSION
#
# =============================================================================
#Actualy Logistic Regression Model Run
start = time.clock()
generatedLogReg = LogisticRegressionCV(cv=5,penalty='l1', multi_class='ovr',solver='saga',n_jobs=6,tol=.005).fit(X_train,y_train)
print (time.clock() - start)
filename = 'savedModels/hotEncode_Train_LogRegCV_3Folds_OVR_L1.sav'
pickle.dump(generatedLogReg, open(filename, 'wb')) #export model so you don't need to rerun it and wait forever

# =============================================================================
# 
# READ IN DATA SINCE IT HAS ALREADY BEEN RAN
# 
# =============================================================================
#read in the model since it was already ran
Train_L1_filepath='savedModels/Train_LogRegCV_5Folds_OVR_L1.sav'
Train_L1 = pickle.load(open(Train_L1_filepath, 'rb'))
Train_L2_filepath='savedModels/Train_LogRegCV_5Folds_OVR_L2.sav'
Train_L2 = pickle.load(open(Train_L2_filepath, 'rb'))

X_test=test_set[['x','y','dyn_prop','rcs','vx_comp','vy_comp','ambig_state','x_rms','y_rms','invalid_state','pdh0','vx_rms','vy_rms']].to_numpy()
y_test=test_set['BasicCategoryNum']
# =============================================================================
# 
# EVALUATING L1 LOG REG For Trianing & Test Sets
# 
# =============================================================================
#Obtain Coeffecients
Train_L1_coef=Train_L1.coef_
predictedl1TrainVals = Train_L1.predict(X_train)
#Create &Confusion Matrix for training set
L1Traincm=confusion_matrix(y_train,predictedl1TrainVals)
plot_confusion_matrix(L1Traincm, normalize=True,
                      title='Confusion Matrix for Logistic Regression with L1 Penalty on Training Set',
                      target_names=['Bicycle','Bus','Construction','Motorcycle','Pass. Veh.','Pedestrian','Tractor','Trailer'])

predictedl1TestVals = Train_L1.predict(X_test)
L1Testcm=confusion_matrix(y_test,predictedl1TestVals)
plot_confusion_matrix(L1Testcm, normalize=True,
                      title='Confusion Matrix for Logistic Regression with L1 Penalty on Test Set',
                      target_names=['Bicycle','Bus','Construction','Motorcycle','Pass. Veh.','Pedestrian','Tractor','Trailer'])

# =============================================================================
# 
# EVALUATING L2 LOG REG
# 
# =============================================================================
#Obtain Coeffecients
Train_L2_coef=Train_L2.coef_
predictedl1TrainVals = Train_L2.predict(X_train)

#Obtain Coeffecients
Train_L2_coef=Train_L2.coef_
predictedl2TrainVals = Train_L2.predict(X_train)
#Create Confusion Matrix for training set
L2Traincm=confusion_matrix(y_train,predictedl2TrainVals)
plot_confusion_matrix(L2Traincm, normalize=True,
                      title='Confusion Matrix for Logistic Regression with L2 Penalty on Training Set',
                      target_names=['Bicycle','Bus','Construction','Motorcycle','Pass. Veh.','Pedestrian','Tractor','Trailer'])

#Create Confusion Matric for Test Set
predictedl2TestVals = Train_L2.predict(X_test)
L2Testcm=confusion_matrix(y_test,predictedl2TestVals)
plot_confusion_matrix(L2Testcm, normalize=True,
                      title='Confusion Matrix for Logistic Regression with L2 Penalty on Test Set',
                      target_names=['Bicycle','Bus','Construction','Motorcycle','Pass. Veh.','Pedestrian','Tractor','Trailer'])













def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()