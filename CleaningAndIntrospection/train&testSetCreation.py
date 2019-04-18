# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 21:41:39 2019

@author: bowye
"""
import pandas as pd 
import numpy as np
from imblearn.combine import SMOTETomek
from sklearn.model_selection import train_test_split  

#read in clean data set
path = "../Data/CleansedRadarPoints.csv"
pointDF = pd.read_csv(path, index_col=False)
columns=['x','y','dyn_prop','rcs','vx_comp','vy_comp','ambig_state','x_rms','y_rms','invalid_state','pdh0','vx_rms','vy_rms','BasicCategoryNum']
#divide data from labels
X=pointDF[['x','y','dyn_prop','rcs','vx_comp','vy_comp','ambig_state','x_rms','y_rms','invalid_state','pdh0','vx_rms','vy_rms']].to_numpy()
y=pointDF['BasicCategoryNum']
# =============================================================================
# 
# CREATE OVERSAMPLED DATASET
# 
# =============================================================================
smt = SMOTETomek(ratio='auto')
X_smt, y_smt = smt.fit_sample(X, y)

#create test and training sets
X_train, X_test, y_train, y_test = train_test_split(X_smt, y_smt, test_size = 0.20)
# =============================================================================
# 
# Reconstruct full data frame to include BasicCategory
# 
# =============================================================================
y_smt_reshape=y_smt.reshape((-1,1))
fullOverSampled=np.append(X_smt,y_smt_reshape,1)
fullOverSampledDf=pd.DataFrame(fullOverSampled,columns=columns)

fullOverSampledDf['BasicCategoryNum'].value_counts()
#Create Pedestrian Basic Category
peds=fullOverSampledDf[fullOverSampledDf['BasicCategoryNum']==5]
peds = peds.assign(BasicCategory='Pedestrian')
#Create Passenger Vehicle Basic Category
cars=fullOverSampledDf[fullOverSampledDf['BasicCategoryNum']==4]
cars = cars.assign(BasicCategory='Passenger Vehicle')
#Create Bicycle Basic Category
bikes=fullOverSampledDf[fullOverSampledDf['BasicCategoryNum']==0]
bikes = bikes.assign(BasicCategory='Bicycle')
#Create Motorcycle Basic Category
motorcycles=fullOverSampledDf[fullOverSampledDf['BasicCategoryNum']==3]
motorcycles = motorcycles.assign(BasicCategory='Motorcycle')
#Create tractor Basic Category
tractors=fullOverSampledDf[fullOverSampledDf['BasicCategoryNum']==6]
tractors = tractors.assign(BasicCategory='Tractor')
#Create Tractor Basic Category
trailers=fullOverSampledDf[fullOverSampledDf['BasicCategoryNum']==7]
trailers = trailers.assign(BasicCategory='Trailer')
#Create Bus Basic Category
Buses=fullOverSampledDf[fullOverSampledDf['BasicCategoryNum']==1]
Buses = Buses.assign(BasicCategory='Bus')
#Create Construction Vehicle Basic Category
constructionVehicles=fullOverSampledDf[fullOverSampledDf['BasicCategoryNum']==2]
constructionVehicles = constructionVehicles.assign(BasicCategory='Construction Vehicle')

reconstructuredOverSampledDF=peds.append([cars, bikes,motorcycles,tractors,trailers,Buses,constructionVehicles])

reconstructuredOverSampledDF['BasicCategory'].value_counts()
reconstructuredOverSampledDF.to_csv('OverSampled_FullDataSet.csv')
# =============================================================================
# 
# SPLIT DATA INTO TEST AND TRAINING SETS AND SAVE IT
# 
# =============================================================================
train,test = train_test_split(reconstructuredOverSampledDF, test_size = 0.20)
train.to_csv('OverSampled_TrainSet.csv')
test.to_csv('OverSampled_TestSet.csv')