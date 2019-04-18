# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 22:46:56 2019

@author: bowye
"""

import pandas as pd 
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
from mlxtend.plotting import category_scatter

path = "../Data/OverSampled/OverSampled_FullDataSet.csv"

#Load in cleansed radar points data frame
pointDF = pd.read_csv(path, index_col=False)
# =============================================================================
# 
# PLOTTING FREQUENCIES OF EACH CLASS
#     
# =============================================================================
fig, ax = plt.subplots()
ax.set_title('Frequency of Basic Categories in Over Sampled Data Set')
ax.set_xlabel('Basic Category Types')
ax.set_ylabel('Occurances of each Category')
pointDF['BasicCategory'].value_counts().plot(ax=ax,kind='bar')
plt.show()
# =============================================================================
# 
# PLOTTING HISTOGRAMS OF RCS for Objects
#
# =============================================================================
#Pedestrians
pedsDF=pointDF[pointDF['BasicCategory'].str.contains('Pedestrian')]
pedRCSHist=pedsDF.hist(column='rcs')
pl.suptitle("Histogram of Pedestrian Radar Cross Section(RCS) values")
pl.xlabel("RCS Bins")
pl.ylabel("Number of RCS Bin Occurances for Pedestrians")
#Passenger Vehicles
carsDF=pointDF[pointDF['BasicCategory'].str.contains('Passenger Vehicle')]
carsRCSHist=carsDF.hist(column='rcs')
pl.suptitle("Histogram of Car Radar Cross Section(RCS) values")
pl.xlabel("RCS Bins")
pl.ylabel("Number of RCS Bin Occurances for Cars")
#Bicycles
carsDF=pointDF[pointDF['BasicCategory'].str.contains('Bicycle')]
carsRCSHist=carsDF.hist(column='rcs')
pl.suptitle("Histogram of Bicycle Radar Cross Section(RCS) values")
pl.xlabel("RCS Bins")
pl.ylabel("Number of RCS Bin Occurances for Bicycles")
#Trailer
carsDF=pointDF[pointDF['BasicCategory'].str.contains('Trailer')]
carsRCSHist=carsDF.hist(column='rcs')
pl.suptitle("Histogram of Trailer Radar Cross Section(RCS) values")
pl.xlabel("RCS Bins")
pl.ylabel("Number of RCS Bin Occurances for Trailers")

# =============================================================================
# 
# PLOTTING HISTOGRAMS OF pdh0 for Objects
#
# =============================================================================
pointDF['pdh0'].value_counts()
#Pedestrians
pedPDH0ist=pedsDF.hist(column='pdh0')
pl.suptitle("Histogram of Pedestrian False alarm probability (pdh0) values")
pl.xlabel("pdh0 Bins")
pl.ylabel("Number of pdh0 Bin Occurances for Pedestrians")
#Passenger Vehicles
carsPDH0Hist=carsDF.hist(column='pdh0')
pl.suptitle("Histogram of Car False alarm probability (pdh0) values")
pl.xlabel("pdh0 Bins")
pl.ylabel("Number of pdh0 Bin Occurances for Cars")
#Bicycles
carsPDH0Hist=carsDF.hist(column='pdh0')
pl.suptitle("Histogram of Bicycles False alarm probability (pdh0) values")
pl.xlabel("pdh0 Bins")
pl.ylabel("Number of pdh0 Bin Occurances for Bicycles")
#Trailer
carsPDH0Hist=carsDF.hist(column='pdh0')
pl.suptitle("Histogram of Trailers False alarm probability (pdh0) values")
pl.xlabel("pdh0 Bins")
pl.ylabel("Number of pdh0 Bin Occurances for Trailers")

# =============================================================================
# 
# PLOTTING HISTOGRAMS OF vx_comp for Objects
#
# =============================================================================
pointDF['vx_comp'].value_counts()
#Pedestrians
pedvx_compHist=pedsDF.hist(column='vx_comp')
pl.suptitle("Histogram of Pedestrian Relative Longitudinal Velocity (vx_comp) values")
pl.xlabel("vx_comp Bins")
pl.ylabel("Number of vx_comp Bin Occurances for Pedestrians")
#Passenger Vehicles
carsvx_compHist=carsDF.hist(column='vx_comp')
pl.suptitle("Histogram of Car Relative Longitudinal Velocity (vx_comp) values")
pl.xlabel("vx_comp Bins")
pl.ylabel("Number of vx_comp Bin Occurances for Cars")
#Bicycles
carsvx_compHist=carsDF.hist(column='vx_comp')
pl.suptitle("Histogram of Bicycles Relative Longitudinal Velocity (vx_comp) values")
pl.xlabel("vx_comp Bins")
pl.ylabel("Number of vx_comp Bin Occurances for Bicycles")
#Trailer
carsvx_compHist=carsDF.hist(column='vx_comp')
pl.suptitle("Histogram of Trailers Relative Longitudinal Velocity (vx_comp) values")
pl.xlabel("vx_comp Bins")
pl.ylabel("Number of vx_comp Bin Occurances for Trailers")

# =============================================================================
# 
# PLOTTING HISTOGRAMS OF vy_comp for Objects
#
# =============================================================================
pointDF['vy_comp'].value_counts()
#Pedestrians
pedvx_compHist=pedsDF.hist(column='vy_comp')
pl.suptitle("Histogram of Pedestrian Relative Lateral Velocity (vy_comp) values")
pl.xlabel("vy_comp Bins")
pl.ylabel("Number of vy_comp Bin Occurances for Pedestrians")
#Passenger Vehicles
carsvx_compHist=carsDF.hist(column='vy_comp')
pl.suptitle("Histogram of Car Relative Lateral Velocity (vy_comp) values")
pl.xlabel("vy_comp Bins")
pl.ylabel("Number of vy_comp Bin Occurances for Cars")
#Bicycles
carsvx_compHist=carsDF.hist(column='vy_comp')
pl.suptitle("Histogram of Bicycles Relative Lateral Velocity (vy_comp) values")
pl.xlabel("vy_comp Bins")
pl.ylabel("Number of vy_comp Bin Occurances for Bicycles")
#Trailer
carsvx_compHist=carsDF.hist(column='vy_comp')
pl.suptitle("Histogram of Trailers Relative Lateral Velocity (vy_comp) values")
pl.xlabel("vy_comp Bins")
pl.ylabel("Number of vy_comp Bin Occurances for Trailers")

# =============================================================================
# 
# PLOTTING HISTOGRAMS OF vx_rms for Objects
#
# =============================================================================
pointDF['vx_rms'].value_counts()
#Pedestrians
pedvx_compHist=pedsDF.hist(column='vx_rms')
pl.suptitle("Histogram of Pedestrian Longitudinal Velocity Root Mean Squared Error (vx_rms) values")
pl.xlabel("vx_rms Bins")
pl.ylabel("Number of vx_rms Bin Occurances for Pedestrians")
#Passenger Vehicles
carsvx_compHist=carsDF.hist(column='vx_rms')
pl.suptitle("Histogram of Car Longitudinal Velocity Root Mean Squared Error (vx_rms) values")
pl.xlabel("vx_rms Bins")
pl.ylabel("Number of vx_rms Bin Occurances for Cars")
#Bicycles
carsvx_compHist=carsDF.hist(column='vx_rms')
pl.suptitle("Histogram of Bicycles Longitudinal Velocity Root Mean Squared Error (vx_rms) values")
pl.xlabel("vx_rms Bins")
pl.ylabel("Number of vx_rms Bin Occurances for Bicycles")
#Trailer
carsvx_compHist=carsDF.hist(column='vx_rms')
pl.suptitle("Histogram of Trailers Longitudinal Velocity Root Mean Squared Error (vx_rms) values")
pl.xlabel("vx_rms Bins")
pl.ylabel("Number of vx_rms Bin Occurances for Trailers")
# =============================================================================
# 
# PLOTTING HISTOGRAMS OF vy_rms for Objects
#
# =============================================================================
pointDF['vy_rms'].value_counts()
#Pedestrians
pedvx_compHist=pedsDF.hist(column='vy_rms')
pl.suptitle("Histogram of Pedestrian Lateral Velocity Root Mean Squared Error (vy_rms) values")
pl.xlabel("vy_rms Bins")
pl.ylabel("Number of vy_rms Bin Occurances for Pedestrians")
#Passenger Vehicles
carsvx_compHist=carsDF.hist(column='vy_rms')
pl.suptitle("Histogram of Car Lateral Velocity Root Mean Squared Error (vy_rms) values")
pl.xlabel("vy_rms Bins")
pl.ylabel("Number of vy_rms Bin Occurances for Cars")
#Bicycles
carsvx_compHist=carsDF.hist(column='vy_rms')
pl.suptitle("Histogram of Bicycles Lateral Velocity Root Mean Squared Error (vy_rms) values")
pl.xlabel("vy_rms Bins")
pl.ylabel("Number of vy_rms Bin Occurances for Bicycles")
#Trailer
carsvx_compHist=carsDF.hist(column='vy_rms')
pl.suptitle("Histogram of Trailers Lateral Velocity Root Mean Squared Error (vy_rms) values")
pl.xlabel("vy_rms Bins")
pl.ylabel("Number of vy_rms Bin Occurances for Trailers")

# =============================================================================
# 
# PLOTTING HISTOGRAMS OF ambig_state for Objects
#
# =============================================================================
pointDF['x_rms'].value_counts()
#Pedestrians
pedvx_compHist=pedsDF.hist(column='ambig_state')
pl.suptitle("Histogram of Pedestrian Lateral Velocity Root Mean Squared Error (ambig_state) values")
pl.xlabel("ambig_state Bins")
pl.ylabel("Number of ambig_state Bin Occurances for Pedestrians")
#Passenger Vehicles
carsvx_compHist=carsDF.hist(column='ambig_state')
pl.suptitle("Histogram of Car Lateral Velocity Root Mean Squared Error (ambig_state) values")
pl.xlabel("ambig_state Bins")
pl.ylabel("Number of vy_rms Bin Occurances for Cars")
#Bicycles
carsvx_compHist=carsDF.hist(column='ambig_state')
pl.suptitle("Histogram of Bicycles Lateral Velocity Root Mean Squared Error (ambig_state) values")
pl.xlabel("ambig_state Bins")
pl.ylabel("Number of vy_rms Bin Occurances for Bicycles")
#Trailer
carsvx_compHist=carsDF.hist(column='ambig_state')
pl.suptitle("Histogram of Trailers Lateral Velocity Root Mean Squared Error (ambig_state) values")
pl.xlabel("ambig_state Bins")
pl.ylabel("Number of ambig_state Bin Occurances for Trailers")

# =============================================================================
# 
# PLOTTING scatter plots for Objects
#
# =============================================================================
#
fig  = category_scatter(x='vx_comp', y='vy_comp', label_col='BasicCategory', 
                       data=pointDF, legend_loc='upper left',alpha =.3, colors=('blue', 'green', 'red', 'purple', 'black','yellow', 'cyan','orange'))
plt.xlabel('Relative Longitudinal Velocity (vx_comp)')
plt.ylabel('Relative Lateral Velocity (xy_comp)')
plt.title('Relative Longitudinal Velocity (vx_comp) versus Relative Lateral Velocity (xy_comp')
plt.xlim(-35,35)
plt.ylim(-35,35)
fig.set_figheight(20)
fig.set_figwidth(20)


fig2 = category_scatter(x='x', y='rcs', label_col='BasicCategory', 
                       data=pointDF, legend_loc='upper left',alpha =.3, colors=('blue', 'green', 'red', 'purple', 'black','yellow', 'cyan','orange'))
plt.xlabel('Longitudinal Distance (x)')
plt.ylabel('Radar Cross Section(rcs)')
plt.title('Longitudinal Distance (x) versus Radar Cross Section(rcs)')
#plt.xlim(-50,50)
#plt.ylim(-50,50)
fig2.set_figheight(20)
fig2.set_figwidth(20)

fig3 = category_scatter(x='y', y='rcs', label_col='BasicCategory', 
                       data=pointDF, legend_loc='upper left',alpha =.3, colors=('blue', 'green', 'red', 'purple', 'black','yellow', 'cyan','orange'))
plt.xlabel('Lateral Distance (y)')
plt.ylabel('Radar Cross Section(rcs)')
plt.title('Lateral Distance (y) versus Radar Cross Section(rcs)')
plt.xlim(-75,75)
plt.ylim(-10,50)
fig3.set_figheight(20)
fig3.set_figwidth(20)


fig4  = category_scatter(x='vx_comp', y='rcs', label_col='BasicCategory', 
                       data=pointDF, legend_loc='upper left',alpha =.3, colors=('blue', 'green', 'red', 'purple', 'black','yellow', 'cyan','orange'))
plt.xlabel('Relative Longitudinal Velocity (vx_comp)')
plt.ylabel('Radar Cross Section (RCS)')
plt.title('Relative Longitudinal Velocity (vx_comp) versus Radar Cross Section (RCS)')
plt.xlim(-60,60)
plt.ylim(-10,50)
fig4.set_figheight(20)
fig4.set_figwidth(20)

fig5  = category_scatter(x='vy_comp', y='rcs', label_col='BasicCategory', 
                       data=pointDF, legend_loc='upper left',alpha =.3, colors=('blue', 'green', 'red', 'purple', 'black','yellow', 'cyan','orange'))
plt.xlabel('Relative Lateral Velocity (vy_comp)')
plt.ylabel('Radar Cross Section (RCS)')
plt.title('Relative Lateral Velocity (vy_comp) versus Radar Cross Section (RCS)')
plt.xlim(-60,60)
plt.ylim(-10,50)
fig5.set_figheight(20)
fig5.set_figwidth(20)

fig6  = category_scatter(x='y', y='x', label_col='BasicCategory',
                       data=pointDF, legend_loc='upper left',alpha =.3, colors=('blue', 'green', 'red', 'purple', 'black','yellow', 'cyan','orange'))
plt.ylabel('Longitudinal Position(x)')
plt.xlabel('Lateral Position(y)')
plt.title('Longitudinal Position(x) versus Lateral Position(y)')
#plt.xlim(-60,60)
#plt.ylim(-10,50)
fig6.set_figheight(20)
fig6.set_figwidth(20)



fig7  = category_scatter(x='x', y='y', label_col='BasicCategory', 
                       data=pointDF, legend_loc='upper left',alpha =.3, colors=('blue', 'green', 'red', 'purple', 'black','yellow', 'cyan','orange'))
plt.ylabel('Longitudinal Position(x)')
plt.xlabel('Lateral Position(y)')
plt.title('Longitudinal Position(x) versus Lateral Position(y)')
#plt.xlim(-60,60)
#plt.ylim(-10,50)
fig7.set_figheight(20)
fig7.set_figwidth(20)
# =============================================================================
# 
# notBusDF=pointDF[~(pointDF['BasicCategory'].str.contains('Bus'))]
# notBusDF=notBusDF[~(notBusDF['BasicCategory'].str.contains('Tractor'))]
# notBusDF=notBusDF[~(notBusDF['BasicCategory'].str.contains('Trailer'))]
# notBusDF=notBusDF[~(notBusDF['BasicCategory'].str.contains('Passenger Vehicle'))]
# #notBusDF=pointDF[(pointDF['BasicCategory'].str.contains('Bicycle'))]
# fig6  = category_scatter(x='vy_comp', y='rcs', label_col='BasicCategory', 
#                        data=notBusDF, legend_loc='upper left')
# plt.ylabel('Longitudinal Position(x)')
# plt.xlabel('Lateral Position(y)')
# plt.title('Longitudinal Position(x) versus Lateral Position(y)')
# #plt.xlim(-60,60)
# #plt.ylim(-10,50)
# fig6.set_figheight(20)
# fig6.set_figwidth(20)
# =============================================================================
