# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 19:40:16 2019

@author: bowye
"""
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split  
import pandas as pd 

# =============================================================================
# 
# READ IN UNDERSAMPLED DATA SET
# 
# =============================================================================
path = "../Data/UnderSampled/UnderSampled_FullDataSet.csv"
pointDF = pd.read_csv(path, index_col=False)
columns=['x','y','dyn_prop','rcs','vx_comp','vy_comp','ambig_state','x_rms','y_rms','invalid_state','pdh0','vx_rms','vy_rms','BasicCategoryNum']
#divide data from labels
print(pointDF.BasicCategory.value_counts())
# =============================================================================
# 
# CREATE HOT ENCODING
# 
# =============================================================================
#Create Hot Encoding for Dynamic Properties
dyn_prop_enc = OneHotEncoder(handle_unknown='error',categories='auto')
X=pointDF[['dyn_prop']]
print(X['dyn_prop'].value_counts())
dyn_prop_enc.fit(X)
print(dyn_prop_enc.categories_)
dyn_prop_array=dyn_prop_enc.transform(X).toarray()
pointDF.insert(value=dyn_prop_array[:,0],column='dyn_prop_moving',loc=0)
pointDF.insert(value=dyn_prop_array[:,1],column='dyn_prop_stationary',loc=0)
pointDF.insert(value=dyn_prop_array[:,2],column='dyn_prop_oncoming',loc=0)
pointDF.insert(value=dyn_prop_array[:,3],column='dyn_prop_stationary_candidate',loc=0)
pointDF.insert(value=dyn_prop_array[:,4],column='dyn_prop_unknown',loc=0)
pointDF.insert(value=dyn_prop_array[:,5],column='dyn_prop_crossing stationary',loc=0)
pointDF.insert(value=dyn_prop_array[:,6],column='dyn_prop_crossing moving',loc=0)
pointDF=pointDF.drop(columns=['dyn_prop'])

#Create Hot Encoding for Ambiguous states
ambig_state_enc = OneHotEncoder(handle_unknown='error',categories='auto')
X=pointDF[['ambig_state']]
print(X['ambig_state'].value_counts())
ambig_state_enc.fit(X)
print(ambig_state_enc.categories_)
ambig_state_array =ambig_state_enc.transform(X).toarray()
pointDF.insert(value=ambig_state_array[:,0],column='ambig_state_invalid',loc=0)
pointDF.insert(value=ambig_state_array[:,1],column='ambig_state_ambiguous',loc=0)
pointDF.insert(value=ambig_state_array[:,2],column='ambig_state_staggered ramp',loc=0)
pointDF.insert(value=ambig_state_array[:,3],column='ambig_state_unambiguous',loc=0)
pointDF=pointDF.drop(columns=['ambig_state'])

#Create Hot Encoding for invalid states
invalid_state_enc = OneHotEncoder(handle_unknown='error',categories='auto')
X=pointDF[['invalid_state']]
print(X['invalid_state'].value_counts())
invalid_state_enc.fit(X)
print(invalid_state_enc.categories_)
invalid_state_array=invalid_state_enc.transform(X).toarray()
pointDF.insert(value=invalid_state_array[:,0],column='invalid_state_valid',loc=0)
pointDF.insert(value=invalid_state_array[:,1],column='invalid_state_valid_low_RCS',loc=0)
pointDF.insert(value=invalid_state_array[:,2],column='invalid_state_valid_azimuth_correction',loc=0)
pointDF.insert(value=invalid_state_array[:,3],column='invalid_state_valid_high_child_prob',loc=0)
pointDF.insert(value=invalid_state_array[:,4],column='invalid_state_valid_high_prob_50_artefact',loc=0)
pointDF.insert(value=invalid_state_array[:,5],column='invalid_state_valid_no_local_max',loc=0)
pointDF.insert(value=invalid_state_array[:,6],column='invalid_state_valid_high_artefact_prob',loc=0)
pointDF.insert(value=invalid_state_array[:,7],column='invalid_state_valid_above_95m',loc=0)
pointDF.insert(value=invalid_state_array[:,8],column='invalid_state_valid_high_multitarget_prob',loc=0)
pointDF=pointDF.drop(columns=['invalid_state'])

#Create Hot Encoding for pdh0
pdh0_enc = OneHotEncoder(handle_unknown='error',categories='auto')
X=pointDF[['pdh0']]
print(X['pdh0'].value_counts())
pdh0_enc.fit(X)
print(pdh0_enc.categories_)
pdh0_array=pdh0_enc.transform(X).toarray()
pointDF.insert(value=pdh0_array[:,0],column='False alarm <25%',loc=0)
pointDF.insert(value=pdh0_array[:,1],column='False alarm 75%',loc=0)
pointDF.insert(value=pdh0_array[:,2],column='False alarm 99.9%',loc=0)
pointDF=pointDF.drop(columns=['pdh0'])

#Save hot encoded data set to csv
pointDF.to_csv('../Data/UnderSampledHotEncoded/UnderSampledHotEncoded_FullSet.csv')
# =============================================================================
# 
# SPLIT DATA INTO TEST AND TRAINING SETS AND SAVE IT
# 
# =============================================================================
train,test = train_test_split(pointDF, test_size = 0.20)
train.to_csv('../Data/UnderSampledHotEncoded/UnderSampledHotEncoded_TrainSet.csv')
test.to_csv('../Data/UnderSampledHotEncoded/UnderSampledHotEncoded_TestSet.csv')