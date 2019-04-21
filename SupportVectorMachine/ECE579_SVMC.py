# -*- coding: utf-8 -*-
"""
@author Brandon Fedoruk
"""
import pandas as pd
from sklearn import svm
import time
import pickle
from sklearn.metrics import classification_report, confusion_matrix  

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
X_train = train_data[['dyn_prop','rcs','vx_comp','vy_comp','ambig_state','x_rms','y_rms','invalid_state','pdh0','vx_rms','vy_rms']]
y_train = train_data['BasicCategoryNum']

X_test  = test_data[['dyn_prop','rcs','vx_comp','vy_comp','ambig_state','x_rms','y_rms','invalid_state','pdh0','vx_rms','vy_rms']]
y_test  = test_data['BasicCategoryNum']

#generate model coefficents
clf = svm.SVC(gamma='scale', decision_function_shape='ovo',cache_size=5000,tol=0.1)
#train model
start = time.clock()
clf.fit(X_train,y_train) 
print (time.clock() - start)


#export model so you don't need to rerun it and wait forever
filename= 'McSVM_noxy.sav' 
pickle.dump(clf, open(filename, 'wb')) 

# Import model for post information processing
svm_import = pickle.load(open(filename, 'rb'))

#Prediction on the test set
pred_test = svm_import.predict(X_test)
#Prediction on the training set
pred_train = svm_import.predict(X_train)  

#Evaluate on the training dataset to determine severity of overfitting
print(confusion_matrix(y_train,
                       pred_train))
                       #title='SVM With RBF Confusion Matrix On Training Set',
                       #target_names=['Bicycle','Bus','Construction','Motorcycle','Pass. Veh.','Pedestrian','Tractor','Trailer']))
#print(confusion_matrix_trainingset)
print(classification_report(y_train,
                            pred_train))
                            #title='SVM With RBF Classification Report On Training Set',
                            #target_names=['Bicycle','Bus','Construction','Motorcycle','Pass. Veh.','Pedestrian','Tractor','Trailer']))

#Evaluate on the test dataset
print(confusion_matrix(y_test,
                       pred_test))
                       #title='SVM With RBF Confusion Matrix On Test Set',
                       #target_names=['Bicycle','Bus','Construction','Motorcycle','Pass. Veh.','Pedestrian','Tractor','Trailer']))

print(classification_report(y_test,
                            pred_test))
                            #title='SVM With RBF Classification Report On Test Set',
                            #target_names=['Bicycle','Bus','Construction','Motorcycle','Pass. Veh.','Pedestrian','Tractor','Trailer']))