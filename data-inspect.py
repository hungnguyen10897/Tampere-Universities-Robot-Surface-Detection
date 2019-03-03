#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 20:39:03 2019

@author: hung
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns

def process_x(X):
#    X = X[:,4:,:]
    X_test_mean = np.mean(X, axis=2)
    X_test_std = np.std(X, axis=2)
    X_test_median = np.median(X,axis = 2)
    
    speed = np.sqrt(np.sum(X[:,4:7,:]**2,axis=1))
    
    avg_speed = np.mean(speed,axis=1)
    max_speed = np.max(speed,axis=1)
    min_speed = np.min(speed,axis=1)
    
    avg_speed = avg_speed.reshape(avg_speed.shape[0],1)
    max_speed = max_speed.reshape(max_speed.shape[0],1)
    min_speed = min_speed.reshape(min_speed.shape[0],1)
    
    acceleration = np.sqrt(np.sum(X[:,7:,:]**2,axis=1))
    
    avg_acceleration = np.mean(acceleration,axis=1)
    max_acceleration = np.max(acceleration,axis=1)
    min_acceleration = np.min(acceleration,axis=1)
    
    avg_acceleration = avg_acceleration.reshape(avg_acceleration.shape[0],1)
    max_acceleration = max_acceleration.reshape(max_acceleration.shape[0],1)
    min_acceleration = min_acceleration.reshape(min_speed.shape[0],1)
    
    


    
    return np.hstack((X_test_mean, X_test_std,X_test_median ,avg_speed,max_speed,min_speed))

groups = pd.read_csv('dataset/groups.csv')

X_test = np.load(r'dataset/X_test_kaggle.npy')

X_train = np.load(r'dataset/X_train_kaggle.npy')

y_train_raw = pd.read_csv(r'dataset/y_train_final_kaggle.csv')

encoder = LabelEncoder()
encoder.fit(np.unique(y_train_raw['Surface']))
y_train = encoder.transform(y_train_raw['Surface'])

X = X_train

X_4 = pd.DataFrame(data=X[:,4,:])
X_4 = X_4.rolling(window=2,axis=1).apply(lambda x: abs(x.max()-x.min()),raw=True)
X_4_max = X_4.max(axis=1).values
X_4_min = X_4.min(axis=1).values


z_soft_tiles = y_train_raw['Surface'] == 'soft_tiles'
z_hard_tiles = y_train_raw['Surface'] == 'hard_tiles'
z_carpet = y_train_raw['Surface'] == 'carpet'
z_tiled = y_train_raw['Surface'] == 'tiled'
z_fine_concrete = y_train_raw['Surface'] == 'fine_concrete'
z_soft_pvc = y_train_raw['Surface'] == 'soft_pvc'
z_concrete = y_train_raw['Surface'] == 'concrete'
z_wood = y_train_raw['Surface'] == 'wood'
z_hard_tiles_large_space = y_train_raw['Surface'] == 'hard_tiles_large_space'

x_soft_tiles = X_train[z_soft_tiles.values,:,:]
x_hard_tiles = X_train[z_hard_tiles.values,:,:]
x_carpet = X_train[z_carpet.values,:,:]
x_tiled = X_train[z_tiled.values,:,:]
x_fine_concrete = X_train[z_fine_concrete.values,:,:]
x_soft_pvc = X_train[z_soft_pvc.values,:,:]
x_concrete = X_train[z_concrete.values,:,:]
x_wood = X_train[z_wood.values,:,:]
x_hard_tiles_large_space = X_train[z_hard_tiles_large_space.values,:,:]

#fig,axes = plt.subplots(nrows=5,ncols=2,figsize=(15,10))
#
#y1 = x_soft_tiles[0,4,:]
#y1_2 = x_soft_tiles[1,4,:]
#y2 = x_hard_tiles[0,4,:]
#y2_2 = x_hard_tiles[1,4,:]
#y3 = x_carpet[0,4,:]
#y3_2 = x_carpet[1,4,:]
#axes[0][0].plot(y1,'r',y1_2,'r',y2,'g',y2_2,'g',y3,'b',y3_2,'b')
#axes[0][0].set_title('4 - Speed X')
#
#y1 = x_soft_tiles[0,5,:]
#y1_2 = x_soft_tiles[1,5,:]
#y2 = x_hard_tiles[0,5,:]
#y2_2 = x_hard_tiles[1,5,:]
#y3 = x_carpet[0,5,:]
#y3_2 = x_carpet[1,5,:]
#axes[0][1].plot(y1,'r',y1_2,'r',y2,'g',y2_2,'g',y3,'b',y3_2,'b')
#axes[0][1].set_title('5 - Speed Y')
#
#y1 = x_soft_tiles[0,6,:]
#y1_2 = x_soft_tiles[1,6,:]
#y2 = x_hard_tiles[0,6,:]
#y2_2 = x_hard_tiles[1,6,:]
#y3 = x_carpet[0,6,:]
#y3_2 = x_carpet[1,6,:]
#axes[1][0].plot(y1,'r',y1_2,'r',y2,'g',y2_2,'g',y3,'b',y3_2,'b')
#axes[1][0].set_title('6 - Speed Z')
#
#y1 = x_soft_tiles[0,7,:]
#y1_2 = x_soft_tiles[1,7,:]
#y2 = x_hard_tiles[0,7,:]
#y2_2 = x_hard_tiles[1,7,:]
#y3 = x_carpet[0,7,:]
#y3_2 = x_carpet[1,7,:]
#axes[1][1].plot(y1,'r',y1_2,'r',y2,'g',y2_2,'g',y3,'b',y3_2,'b')
#axes[1][1].set_title('7 - Acceleration X')
#
#y1 = x_soft_tiles[0,8,:]
#y1_2 = x_soft_tiles[1,8,:]
#y2 = x_hard_tiles[0,8,:]
#y2_2 = x_hard_tiles[1,8,:]
#y3 = x_carpet[0,8,:]
#y3_2 = x_carpet[1,8,:]
#axes[2][0].plot(y1,'r',y1_2,'r',y2,'g',y2_2,'g',y3,'b',y3_2,'b')
#axes[2][0].set_title('8 - Acceleration Y')
#
#y1 = x_soft_tiles[0,9,:]
#y1_2 = x_soft_tiles[1,9,:]
#y2 = x_hard_tiles[0,9,:]
#y2_2 = x_hard_tiles[1,9,:]
#y3 = x_carpet[0,9,:]
#y3_2 = x_carpet[1,9,:]
#axes[2][1].plot(y1,'r',y1_2,'r',y2,'g',y2_2,'g',y3,'b',y3_2,'b')
#axes[2][1].set_title('9 - Acceleration Z')
#
#y1 = x_soft_tiles[0,0,:]
#y1_2 = x_soft_tiles[1,0,:]
#y2 = x_hard_tiles[0,0,:]
#y2_2 = x_hard_tiles[1,0,:]
#y3 = x_carpet[0,0,:]
#y3_2 = x_carpet[1,0,:]
#axes[3][0].plot(y1,'r',y1_2,'r',y2,'g',y2_2,'g',y3,'b',y3_2,'b')
#axes[3][0].set_title('0 - Orientation X')
#
#y1 = x_soft_tiles[0,1,:]
#y1_2 = x_soft_tiles[1,1,:]
#y2 = x_hard_tiles[0,1,:]
#y2_2 = x_hard_tiles[1,1,:]
#y3 = x_carpet[0,1,:]
#y3_2 = x_carpet[1,1,:]
#axes[3][1].plot(y1,'r',y1_2,'r',y2,'g',y2_2,'g',y3,'b',y3_2,'b')
#axes[3][1].set_title('1 - Orientation Y')
#
#y1 = x_soft_tiles[0,2,:]
#y1_2 = x_soft_tiles[1,2,:]
#y2 = x_hard_tiles[0,2,:]
#y2_2 = x_hard_tiles[1,2,:]
#y3 = x_carpet[0,2,:]
#y3_2 = x_carpet[1,2,:]
#axes[4][0].plot(y1,'r',y1_2,'r',y2,'g',y2_2,'g',y3,'b',y3_2,'b')
#axes[4][0].set_title('2 - Orientation Z')
#
#y1 = x_soft_tiles[0,3,:]
#y1_2 = x_soft_tiles[1,3,:]
#y2 = x_hard_tiles[0,3,:]
#y2_2 = x_hard_tiles[1,3,:]
#y3 = x_carpet[0,3,:]
#y3_2 = x_carpet[1,3,:]
#axes[4][1].plot(y1,'r',y1_2,'r',y2,'g',y2_2,'g',y3,'b',y3_2,'b')
#axes[4][1].set_title('3 - Orientation W')
 

fig,axes = plt.subplots(nrows=3,ncols=3,figsize=(15,15))

x1 = pd.DataFrame(np.transpose(x_soft_tiles[0,:,:])).corr()
sns.heatmap(x1,ax = axes[0][0])

x1 = pd.DataFrame(np.transpose(x_soft_tiles[1,:,:])).corr()
sns.heatmap(x1,ax = axes[0][1])

x1 = pd.DataFrame(np.transpose(x_soft_tiles[2,:,:])).corr()
sns.heatmap(x1,ax = axes[0][2])

x2 = pd.DataFrame(np.transpose(x_hard_tiles[0,:,:])).corr()
sns.heatmap(x2,ax = axes[1][0])

x2 = pd.DataFrame(np.transpose(x_hard_tiles[1,:,:])).corr()
sns.heatmap(x2,ax = axes[1][1])

x2 = pd.DataFrame(np.transpose(x_hard_tiles[2,:,:])).corr()
sns.heatmap(x2,ax = axes[1][2])

x3 = pd.DataFrame(np.transpose(x_carpet[0,:,:])).corr()
sns.heatmap(x3,ax = axes[2][0])

x3 = pd.DataFrame(np.transpose(x_carpet[1,:,:])).corr()
sns.heatmap(x3,ax = axes[2][1])

x3 = pd.DataFrame(np.transpose(x_carpet[2,:,:])).corr()
sns.heatmap(x3,ax = axes[2][2])







plt.tight_layout()



