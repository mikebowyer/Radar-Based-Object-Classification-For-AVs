# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 21:41:39 2019

@author: bowye
"""
import pandas as pd 
import numpy as np
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTENC
from sklearn.model_selection import train_test_split  
import time
#read in clean data set
path = "../Data/CleansedRadarPoints.csv"
pointDF = pd.read_csv(path, index_col=False)
columns=['x','y','dyn_prop','rcs','vx_comp','vy_comp','ambig_state','x_rms','y_rms','invalid_state','pdh0','vx_rms','vy_rms','BasicCategoryNum']
#divide data from labels
pointDF.x_rms.value_counts()
# =============================================================================
# 
# CREATE OVERSAMPLED DATASET
# 
# =============================================================================

pointDF["dyn_prop"] = pointDF["dyn_prop"].astype('category')
pointDF["ambig_state"] = pointDF["ambig_state"].astype('category')
pointDF["x_rms"] = pointDF["x_rms"].astype('category')
pointDF["y_rms"] = pointDF["y_rms"].astype('category')
pointDF["y_rms"] = pointDF["y_rms"].astype('category')
pointDF["invalid_state"] = pointDF["invalid_state"].astype('category')
pointDF["pdh0"] = pointDF["pdh0"].astype('category')
pointDF["vx_rms"] = pointDF["vx_rms"].astype('category')
pointDF["vy_rms"] = pointDF["vy_rms"].astype('category')

X=pointDF[['x','y','dyn_prop','rcs','vx_comp','vy_comp','ambig_state','x_rms','y_rms','invalid_state','pdh0','vx_rms','vy_rms']]
y=pointDF['BasicCategoryNum']
sm = SMOTENC(categorical_features=[2,6,7,8,9,10,11,12],n_jobs=6)

start = time.clock()
smt = SMOTETomek(smote=SMOTENC,ratio='auto')
X_smt, y_smt = sm.fit_sample(X, y)
print (time.clock() - start)
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

reconstructedOverSampledDF=peds.append([cars, bikes,motorcycles,tractors,trailers,Buses,constructionVehicles])

reconstructedOverSampledDF['BasicCategory'].value_counts()
reconstructedOverSampledDF.to_csv('OverSampled_FullDataSet.csv')
# =============================================================================
# 
# SPLIT DATA INTO TEST AND TRAINING SETS AND SAVE IT
# 
# =============================================================================
train,test = train_test_split(reconstructedOverSampledDF, test_size = 0.20)
train.to_csv('OverSampled_TrainSet.csv')
test.to_csv('OverSampled_TestSet.csv')