# -*- coding: utf-8 -*-
"""
@author Brandon Fedoruk
"""
import pandas as pd
from sklearn import svm
import time
import pickle
from sklearn.metrics import confusion_matrix  
import numpy as np

def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True,saveFileName='foo'):
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
    #import numpy as np
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
    plt.savefig(saveFileName, bbox_inches='tight')
    plt.show()
    

# Initial read of sanitized data MUST be in working directory
filePath1   = "UnderSampled_TrainSet.csv"
filePath2   = "UnderSampled_TestSet.csv"
train_data = pd.read_csv(filePath1, index_col=False)
test_data  = pd.read_csv(filePath2, index_col=False)
#clf = svm.SVC(C=1.0, cache_size=2000, class_weight='balanced', coef0=0.0,
#              decision_function_shape='ovo', degree=3, gamma='scale', kernel='rbf',
#              max_iter=-1, probability=False, random_state=None, shrinking=True,
#              tol=0.1, verbose=False)

# Pre process dataset.
X_train = train_data[['rcs','vx_comp','ambig_state','x_rms','y_rms','vx_rms','vy_rms']]
y_train = train_data['BasicCategoryNum']

X_test  = test_data[['rcs','vx_comp','ambig_state','x_rms','y_rms','vx_rms','vy_rms']]
y_test  = test_data['BasicCategoryNum']

#generate model coefficents
clf = svm.SVC(C=1.0,gamma='scale', decision_function_shape='ovo',cache_size=5000,
              tol=0.1,kernel='linear',class_weight='balanced')
#train model
start = time.clock()
clf.fit(X_train,y_train) 
print (time.clock() - start)

#export model so you don't need to rerun it and wait forever
filename= 'McSVM_no_xy_vycomp_pdh0_2.sav' 
pickle.dump(clf, open(filename, 'wb')) 

# Import model for post information processing
#svm_import = pickle.load(open(filename, 'rb'))

#Prediction on the test set
#pred_test = svm_import.predict(X_test)
pred_test = clf.predict(X_test)

#Prediction on the training set
#pred_train = svm_import.predict(X_train)
pred_train = clf.predict(X_train)  

#Evaluate on the training dataset to determine severity of overfitting
conf_ytrain=confusion_matrix(y_train,pred_train)
plot_confusion_matrix(conf_ytrain, normalize=True,
                      title='Confusion Matrix for SVM with RBF, C = 1.0 on Training Set',
                      target_names=['Bicycle','Bus','Construction','Motorcycle','Pass. Veh.','Pedestrian','Tractor','Trailer'],
                      saveFileName='conf_ytrain')

#Evaluate on the test dataset
conf_ytest=confusion_matrix(y_test,pred_test)

plot_confusion_matrix(conf_ytest, normalize=True,
                      title='Confusion Matrix for SVM with RBF, C = 1.0 on Test Set',
                      target_names=['Bicycle','Bus','Construction','Motorcycle','Pass. Veh.','Pedestrian','Tractor','Trailer'],
                      saveFileName='conf_ytest')

#Save confusion matrix data
df=pd.DataFrame(conf_ytrain)                            
df.to_csv('confusion_ytrain_Usamp_rbf_C1.csv')
#df.to_html('confusion_ytrain_Usamp_rbf_C1.html')

df=pd.DataFrame(conf_ytest)
df.to_csv('confusion_ytest_Usamp_rbf_C1.csv')
#df.to_html('confusion_ytest_Usamp_rbf_C1.html')