# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 18:46:25 2019

@author: bowye
"""
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

path = "Data/labelledRadarPoints.csv"

#Load in radar points data frame
radarPointDF = pd.read_csv(path, index_col=False)

print(radarPointDF.head(3))
radarPointDF.iloc[0]

#Plot Occurances of different categories
fig, ax = plt.subplots()
ax.set_title('Frequency of Label Categories in Data Set')
ax.set_xlabel('Category Types')
ax.set_ylabel('Occurances of each Category')
radarPointDF['category'].value_counts().plot(ax=ax,kind='bar')
plt.show()


humanPeds=radarPointDF['category']=='human.pedestrian.construction_worker'
humanPedsDF=radarPointDF[humanPeds]


x=radarPointDF.rcs
y=radarPointDF.category

plt.scatter(x, y)