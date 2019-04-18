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
train_path = "../Data/OverSampled/OverSampled_TrainSet.csv"
train_set = pd.read_csv(train_path, index_col=False)
test_path = "../Data/OverSampled/OverSampled_TestSet.csv"
test_set = pd.read_csv(test_path, index_col=False)

train_set['BasicCategory'].value_counts()
test_set['BasicCategory'].value_counts()

X=train_set[['x','y','dyn_prop','rcs','vx_comp','vy_comp','ambig_state','x_rms','y_rms','invalid_state','pdh0','vx_rms','vy_rms']].to_numpy()
y=train_set['BasicCategoryNum']
# =============================================================================
# 
# LOGISTIC REGRESSION
#
# =============================================================================
train_set['BasicCategory'].value_counts()
train_set['BasicCategoryNum'].value_counts()
#Actualy Logistic Regression Model Run
start = time.clock()
generatedLogReg = LogisticRegressionCV(cv=3,penalty='l1', multi_class='ovr',solver='saga',n_jobs=6,tol=.005).fit(X,y)
print (time.clock() - start)
filename = 'overSampled_LogRegCV_3Folds_OVR_L1_balanced.sav'
pickle.dump(generatedLogReg, open(filename, 'wb')) #export model so you don't need to rerun it and wait forever

# =============================================================================
# 
# READ IN DATA SINCE IT HAS ALREADY BEEN RAN
# 
# =============================================================================
#read in the model since it was already ran
CV4Folds_L1_filepath='savedModels/LogRegCV_4Folds_OVR_L1_balanced.sav'
CV4Folds_L1 = pickle.load(open(CV4Folds_L1_filepath, 'rb'))
CV4Folds_L2_filepath='savedModels/LogRegCV_4Folds_OVR_L2_balanced.sav'
CV4Folds_L2 = pickle.load(open(CV4Folds_L2_filepath, 'rb'))

X_test=test_set[['x','y','dyn_prop','rcs','vx_comp','vy_comp','ambig_state','x_rms','y_rms','invalid_state','pdh0','vx_rms','vy_rms']].to_numpy()
# =============================================================================
# 
# EVALUATING L1 LOG REG
# 
# =============================================================================
CV4Folds_L1_coef=CV4Folds_L1.coef_
CV4Folds_L1_scores=CV4Folds_L1.scores_ 

predictedl1TestVals = CV4Folds_L1.predict(X_test)
trueTestVals=test_set['BasicCategoryNum']

#Create &Confusion Matrix
L1cm=confusion_matrix(trueTestVals,predictedl1TestVals)

plot_confusion_matrix(L1cm, normalize=True,target_names=['0','1','2','3','4','5','6','7'])
# =============================================================================
# 
# EVALUATING L2 LOG REG
# 
# =============================================================================
CV4Folds_L2_coef=CV4Folds_L2.coef_
CV4Folds_L2_scores=CV4Folds_L2.scores_ 

predictedl2TestVals = CV4Folds_L2.predict(X_test)
trueTestVals=test_set['BasicCategoryNum']

#Create &Confusion Matrix
L2cm=confusion_matrix(trueTestVals,predictedl2TestVals)

plot_confusion_matrix(L2cm, normalize=True,target_names=['0','1','2','3','4','5','6','7'])














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