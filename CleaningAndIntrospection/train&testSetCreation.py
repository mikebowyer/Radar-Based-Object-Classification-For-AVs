# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 21:41:39 2019

@author: bowye
"""
import pandas as pd 
import numpy as np
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import TomekLinks
from imblearn.over_sampling import SMOTENC
from sklearn.model_selection import train_test_split  
from imblearn.under_sampling import ClusterCentroids
from imblearn.under_sampling import RandomUnderSampler
import time
#read in clean data set
path = "../Data/CleansedRadarPoints.csv"
pointDF = pd.read_csv(path, index_col=False)
columns=['x','y','dyn_prop','rcs','vx_comp','vy_comp','ambig_state','x_rms','y_rms','invalid_state','pdh0','vx_rms','vy_rms','BasicCategoryNum']
#divide data from labels
print(pointDF.BasicCategory.value_counts())
# =============================================================================
# 
# CREATE UNDERSAMPLED DATASET
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

start = time.clock()

#Remove passenger car samples randomly
desiredSampleCounts= {
		4 : 75000}
rus = RandomUnderSampler(sampling_strategy=desiredSampleCounts)
X_undersampled, y_undersampled = rus.fit_resample(X, y)

#Remove tractor samples randomly
desiredSampleCounts= {
		6 : 75000}
rus = RandomUnderSampler(sampling_strategy=desiredSampleCounts)
X_undersampled, y_undersampled = rus.fit_resample(X_undersampled, y_undersampled)
print(np.bincount(y_undersampled))

#Remove Tomek Pairs
underSampleObj = TomekLinks(sampling_strategy='all',n_jobs=5)
X_undersampledTomek, y_undersampledTomek = underSampleObj.fit_resample(X_undersampled, y_undersampled)
print(np.bincount(y_undersampledTomek))

#Over sample minority classes to match majority classes
overSampleObj = SMOTENC(categorical_features=[2,6,7,8,9,10,11,12],n_jobs=6)
X_final, y_final = overSampleObj.fit_resample(X_undersampledTomek, y_undersampledTomek)
print(np.bincount(y_final))

print (time.clock() - start)
# =============================================================================
# 
# Reconstruct full data frame to include BasicCategory
# 
# =============================================================================
y_final_reshape=y_final.reshape((-1,1))
underSampledData=np.append(X_final,y_final_reshape,1)
underSampledDf=pd.DataFrame(underSampledData,columns=columns)

underSampledDf['BasicCategoryNum'].value_counts()
#Create Pedestrian Basic Category
peds=underSampledDf[underSampledDf['BasicCategoryNum']==5]
peds = peds.assign(BasicCategory='Pedestrian')
#Create Passenger Vehicle Basic Categoryz
cars=underSampledDf[underSampledDf['BasicCategoryNum']==4]
cars = cars.assign(BasicCategory='Passenger Vehicle')
#Create Bicycle Basic Category
bikes=underSampledDf[underSampledDf['BasicCategoryNum']==0]
bikes = bikes.assign(BasicCategory='Bicycle')
#Create Motorcycle Basic Category
motorcycles=underSampledDf[underSampledDf['BasicCategoryNum']==3]
motorcycles = motorcycles.assign(BasicCategory='Motorcycle')
#Create tractor Basic Category
tractors=underSampledDf[underSampledDf['BasicCategoryNum']==6]
tractors = tractors.assign(BasicCategory='Tractor')
#Create Tractor Basic Category
trailers=underSampledDf[underSampledDf['BasicCategoryNum']==7]
trailers = trailers.assign(BasicCategory='Trailer')
#Create Bus Basic Category
Buses=underSampledDf[underSampledDf['BasicCategoryNum']==1]
Buses = Buses.assign(BasicCategory='Bus')
#Create Construction Vehicle Basic Category
constructionVehicles=underSampledDf[underSampledDf['BasicCategoryNum']==2]
constructionVehicles = constructionVehicles.assign(BasicCategory='Construction Vehicle')

reconstructedUnderSampledDF=peds.append([cars, bikes,motorcycles,tractors,trailers,Buses,constructionVehicles])

reconstructedUnderSampledDF['BasicCategory'].value_counts()
reconstructedUnderSampledDF.to_csv('UnderSampled_FullDataSet.csv')
# =============================================================================
# 
# SPLIT DATA INTO TEST AND TRAINING SETS AND SAVE IT
# 
# =============================================================================
train,test = train_test_split(reconstructedUnderSampledDF, test_size = 0.20)
train.to_csv('UnderSampled_TrainSet.csv')
test.to_csv('UnderSampled_TestSet.csv')