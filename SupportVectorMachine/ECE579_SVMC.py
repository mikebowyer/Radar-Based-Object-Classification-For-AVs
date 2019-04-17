# -*- coding: utf-8 -*-
"""
@author Brandon Fedoruk
"""
import pandas as pd
from sklearn.svm import SVC
import time
import pickle
from sklearn.metrics import classification_report, confusion_matrix  
#filePath = "CleansedRadarPoints.csv"
#data = pd.read_csv(filePath, index_col=False)

#X = data[['x','y','dyn_prop','rcs','vx_comp','vy_comp','ambig_state','x_rms','y_rms','invalid_state','pdh0','vx_rms','vy_rms']]
#y = data[['BasicCategoryNum']]

#from sklearn.model_selection import KFold
#k-fold CV (5 folds)
#kf = KFold(n_splits=5)
#KFold(n_splits=5, random_state=None, shuffle=False)
#Generate the k-fold sets
#for train_index, test_index in kf.split(X):
  #  X_train, X_test = X[train_index], X[test_index]
  #  y_train, y_test = y[train_index], y[test_index]


#from sklearn.model_selection import train_test_split  
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

#train,test = train_test_split(data, test_size = 0.20)  
#train.to_csv('data_train.csv')
#test.to_csv('data_test.csv')


filePath1   = "data_train.csv"
filePath2   = "data_test.csv"
train_data = pd.read_csv(filePath1, index_col=False)
test_data  = pd.read_csv(filePath2, index_col=False)
#clf = svm.SVC(C=1.0, cache_size=2000, class_weight='balanced', coef0=0.0,
#              decision_function_shape='ovo', degree=3, gamma='scale', kernel='rbf',
#              max_iter=-1, probability=False, random_state=None, shrinking=True,
#              tol=0.1, verbose=False)


X_train = train_data[['x','y','dyn_prop','rcs','vx_comp','vy_comp','ambig_state','x_rms','y_rms','invalid_state','pdh0','vx_rms','vy_rms']]
y_train = train_data['BasicCategoryNum']

X_train = X_train.head(10)
y_train = y_train.head(10)

X_test  = test_data[['x','y','dyn_prop','rcs','vx_comp','vy_comp','ambig_state','x_rms','y_rms','invalid_state','pdh0','vx_rms','vy_rms']]
y_test  = test_data['BasicCategoryNum']

clf = SVC(gamma='scale', decision_function_shape='ovo',cache_size=5000,tol=0.1)
#train
start = time.clock()
svm_output = clf.fit(X_train,y_train) 
print (time.clock() - start)


#export model so you don't need to rerun it and wait forever
filename = 'McSVM.sav' 
pickle.dump(svm_output, open(filename, 'wb')) 


#Predict
pred = clf.predict(X_test,y_test)  

#Evaluate
print(confusion_matrix(y_test,pred))  
print(classification_report(y_test,pred))  