# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 18:46:25 2019

@author: bowye
"""
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

path = "../Data/labelledRadarPoints.csv"

#Load in radar points data frame
radarPointDF = pd.read_csv(path, index_col=False)

# =============================================================================
# 
# REMOVING INVALID DATA FROM DATASET
# 
# =============================================================================
valid_radarPointDF=radarPointDF[~(radarPointDF['invalid_state'] == 1)]
valid_radarPointDF=valid_radarPointDF[~(valid_radarPointDF['invalid_state'] == 2)]
valid_radarPointDF=valid_radarPointDF[~(valid_radarPointDF['invalid_state'] == 3)]
valid_radarPointDF=valid_radarPointDF[~(valid_radarPointDF['invalid_state'] == 5)]
valid_radarPointDF=valid_radarPointDF[~(valid_radarPointDF['invalid_state'] == 6)]
valid_radarPointDF=valid_radarPointDF[~(valid_radarPointDF['invalid_state'] == 7)]
valid_radarPointDF=valid_radarPointDF[~(valid_radarPointDF['invalid_state'] == 13)]
valid_radarPointDF=valid_radarPointDF[~(valid_radarPointDF['invalid_state'] == 14)]
print(sum(radarPointDF['invalid_state'].value_counts()))
print(sum(valid_radarPointDF['invalid_state'].value_counts()))
# =============================================================================
# 
# PLOT ALL CATEGORIES
# 
# =============================================================================
#Plot Occurances of different categories
fig, ax = plt.subplots()
ax.set_title('Frequency of Label Categories in Data Set')
ax.set_xlabel('Category Types')
ax.set_ylabel('Occurances of each Category')
radarPointDF['category'].value_counts().plot(ax=ax,kind='bar')
plt.show()
# =============================================================================
# 
# CREATING BASIC CATEGORY DATAFRAME
# 
# =============================================================================
#Create Pedestrian Basic Category
peds=valid_radarPointDF[valid_radarPointDF['category'].str.contains('pedestrian')]
peds = peds.assign(BasicCategory='Pedestrian')

#Create Passenger Vehicle Basic Category
cars=valid_radarPointDF[valid_radarPointDF['category'].str.contains('vehicle.car')]
cars = cars.assign(BasicCategory='Passenger Vehicle')

#Create Bicycle Basic Category
bikes=valid_radarPointDF[valid_radarPointDF['category'].str.contains('bicycle')]
bikes = bikes.assign(BasicCategory='Bicycle')

#Create Motorcycle Basic Category
motorcycles=valid_radarPointDF[valid_radarPointDF['category'].str.contains('motorcycle')]
motorcycles = motorcycles.assign(BasicCategory='Motorcycle')

#Create tractor Basic Category
tractors=valid_radarPointDF[valid_radarPointDF['category'].str.contains('truck')]
tractors = tractors.assign(BasicCategory='Tractor')

#Create Tractor Basic Category
trailers=valid_radarPointDF[valid_radarPointDF['category'].str.contains('trailer')]
trailers = trailers.assign(BasicCategory='Trailer')

#Create Bus Basic Category
Buses=valid_radarPointDF[valid_radarPointDF['category'].str.contains('bus')]
Buses = Buses.assign(BasicCategory='Bus')

#Create Construction Vehicle Basic Category
constructionVehicles=valid_radarPointDF[valid_radarPointDF['category'].str.contains('vehicle.construction')]
constructionVehicles = constructionVehicles.assign(BasicCategory='Construction Vehicle')

#Append all basic categories into final dataframe
radarPointwBasicDF=peds.append([cars, bikes,motorcycles,tractors,trailers,Buses,constructionVehicles])
# =============================================================================
# 
# PLOTTING BASIC CATEGORY FREQUENCIES
# 
# =============================================================================
#Plot Occurances of Basic categories
fig, ax = plt.subplots()
ax.set_title('Frequency of Basic Categories in Data Set')
ax.set_xlabel('Basic Category Types')
ax.set_ylabel('Occurances of each Category')
radarPointwBasicDF['BasicCategory'].value_counts().plot(ax=ax,kind='bar')
plt.show()
# =============================================================================
# 
# REMOVING UNEEDED COLUMNS (z, id, vx, vy, is_quality_valid)
# 
# =============================================================================
cleanDF= radarPointwBasicDF.drop(columns=['z','id','vx','vy','is_quality_valid'])
cleanDF["BasicCategory"] = cleanDF["BasicCategory"].astype('category')
cleanDF.dtypes
cleanDF["BasicCategoryNum"] = cleanDF["BasicCategory"].cat.codes
cleanDF.head()
# =============================================================================
# 
# SAVING CLEANSED DATA INTO CSV
# 
# =============================================================================
cleanDF.to_csv('../Data/CleansedRadarPoints.csv',index=False)
