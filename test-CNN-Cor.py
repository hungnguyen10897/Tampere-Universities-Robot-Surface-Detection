#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 18:49:09 2019

@author: hung
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from keras.models import Sequential
from keras.layers.core import Dense,Activation,Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import to_categorical

def process_x(X):
#    X = X[:,4:,:]
    X_test_mean = np.mean(X[:,:,:], axis=2)
    X_test_std = np.std(X[:,:,:], axis=2)
    X_test_median = np.median(X[:,:,:],axis = 2)
    X_test_max = np.max(X[:,:,:],axis = 2)
    X_test_min = np.min(X[:,:,:],axis = 2)
    
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

    winsize = 2;

    X_4_win = pd.DataFrame(data=X[:,4,:])
    X_4_win = X_4_win.rolling(window=winsize,axis=1).apply(lambda x: abs(x.max()-x.min()),raw=True)
    X_4_win_max = X_4_win.max(axis=1).values
    X_4_win_min = X_4_win.min(axis=1).values
    X_4_win_max = X_4_win_max.reshape(X_4_win_max.shape[0],1)
    X_4_win_min = X_4_win_min.reshape(X_4_win_min.shape[0],1)
    X_4_win_mean = X_4_win.mean(axis=1).values
    X_4_win_mean = X_4_win_mean.reshape(X_4_win_mean.shape[0],1)
    X_4_win_median = X_4_win.median(axis=1).values
    X_4_win_median = X_4_win_median.reshape(X_4_win_median.shape[0],1)
    
    X_5_win = pd.DataFrame(data=X[:,5,:])
    X_5_win = X_5_win.rolling(window=winsize,axis=1).apply(lambda x: abs(x.max()-x.min()),raw=True)
    X_5_win_max = X_5_win.max(axis=1).values
    X_5_win_min = X_5_win.min(axis=1).values
    X_5_win_max = X_5_win_max.reshape(X_5_win_max.shape[0],1)
    X_5_win_min = X_5_win_min.reshape(X_5_win_min.shape[0],1)
    X_5_win_mean = X_5_win.mean(axis=1).values
    X_5_win_mean = X_5_win_mean.reshape(X_5_win_mean.shape[0],1)
    X_5_win_median = X_5_win.median(axis=1).values
    X_5_win_median = X_5_win_median.reshape(X_5_win_median.shape[0],1)
    
    X_6_win = pd.DataFrame(data=X[:,6,:])
    X_6_win = X_6_win.rolling(window=winsize,axis=1).apply(lambda x: abs(x.max()-x.min()),raw=True)
    X_6_win_max = X_6_win.max(axis=1).values
    X_6_win_min = X_6_win.min(axis=1).values
    X_6_win_max = X_6_win_max.reshape(X_6_win_max.shape[0],1)
    X_6_win_min = X_6_win_min.reshape(X_6_win_min.shape[0],1)
    X_6_win_mean = X_6_win.mean(axis=1).values
    X_6_win_mean = X_6_win_mean.reshape(X_6_win_mean.shape[0],1)
    X_6_win_median = X_6_win.median(axis=1).values
    X_6_win_median = X_6_win_median.reshape(X_6_win_median.shape[0],1)
    
    X_7_win = pd.DataFrame(data=X[:,4,:])
    X_7_win = X_7_win.rolling(window=winsize,axis=1).apply(lambda x: abs(x.max()-x.min()),raw=True)
    X_7_win_max = X_7_win.max(axis=1).values
    X_7_win_min = X_7_win.min(axis=1).values
    X_7_win_max = X_7_win_max.reshape(X_7_win_max.shape[0],1)
    X_7_win_min = X_7_win_min.reshape(X_7_win_min.shape[0],1)
    X_7_win_mean = X_7_win.mean(axis=1).values
    X_7_win_mean = X_7_win_mean.reshape(X_7_win_mean.shape[0],1)
    X_7_win_median = X_7_win.median(axis=1).values
    X_7_win_median = X_7_win_median.reshape(X_7_win_median.shape[0],1)
    
    X_8_win = pd.DataFrame(data=X[:,5,:])
    X_8_win = X_8_win.rolling(window=winsize,axis=1).apply(lambda x: abs(x.max()-x.min()),raw=True)
    X_8_win_max = X_8_win.max(axis=1).values
    X_8_win_min = X_8_win.min(axis=1).values
    X_8_win_max = X_8_win_max.reshape(X_8_win_max.shape[0],1)
    X_8_win_min = X_8_win_min.reshape(X_8_win_min.shape[0],1)
    X_8_win_mean = X_8_win.mean(axis=1).values
    X_8_win_mean = X_8_win_mean.reshape(X_8_win_mean.shape[0],1)
    X_8_win_median = X_8_win.median(axis=1).values
    X_8_win_median = X_8_win_median.reshape(X_8_win_median.shape[0],1)
    
    X_9_win = pd.DataFrame(data=X[:,6,:])
    X_9_win = X_9_win.rolling(window=winsize,axis=1).apply(lambda x: abs(x.max()-x.min()),raw=True)
    X_9_win_max = X_9_win.max(axis=1).values
    X_9_win_min = X_9_win.min(axis=1).values
    X_9_win_max = X_9_win_max.reshape(X_9_win_max.shape[0],1)
    X_9_win_min = X_9_win_min.reshape(X_9_win_min.shape[0],1)
    X_9_win_mean = X_9_win.mean(axis=1).values
    X_9_win_mean = X_9_win_mean.reshape(X_9_win_mean.shape[0],1)
    X_9_win_median = X_9_win.median(axis=1).values
    X_9_win_median = X_9_win_median.reshape(X_9_win_median.shape[0],1)
    
    X_0 = X[:,0,:].reshape(X.shape[0],8,16)
    X_0_mean = np.mean(X_0,axis=2)
    X_0_std = np.std(X_0,axis=2)
    X_0_median = np.median(X_0,axis=2)
    X_0 = np.hstack([X_0_mean,X_0_std,X_0_median])
    
    X_1 = X[:,1,:].reshape(X.shape[0],8,16)
    X_1_mean = np.mean(X_1,axis=2)
    X_1_std = np.std(X_1,axis=2)
    X_1_median = np.median(X_1,axis=2)
    X_1 = np.hstack([X_1_mean,X_1_std,X_1_median])
    
    X_2 = X[:,2,:].reshape(X.shape[0],8,16)
    X_2_mean = np.mean(X_2,axis=2)
    X_2_std = np.std(X_2,axis=2)
    X_2_median = np.median(X_2,axis=2)
    X_2 = np.hstack([X_2_mean,X_2_std,X_2_median])
    
    X_3 = X[:,3,:].reshape(X.shape[0],8,16)
    X_3_mean = np.mean(X_3,axis=2)
    X_3_std = np.std(X_3,axis=2)
    X_3_median = np.median(X_3,axis=2)
    X_3 = np.hstack([X_3_mean,X_3_std,X_3_median])
    
    X_4 = X[:,4,:].reshape(X.shape[0],8,16)
    X_4_mean = np.mean(X_4,axis=2)
    X_4_std = np.std(X_4,axis=2)
    X_4_median = np.median(X_4,axis=2)
    X_4 = np.hstack([X_4_mean,X_4_std,X_4_median])
    
    X_5 = X[:,5,:].reshape(X.shape[0],8,16)
    X_5_mean = np.mean(X_5,axis=2)
    X_5_std = np.std(X_5,axis=2)
    X_5_median = np.median(X_5,axis=2)
    X_5 = np.hstack([X_5_mean,X_5_std,X_5_median])
    
    X_6 = X[:,6,:].reshape(X.shape[0],8,16)
    X_6_mean = np.mean(X_6,axis=2)
    X_6_std = np.std(X_6,axis=2)
    X_6_median = np.median(X_6,axis=2)
    X_6 = np.hstack([X_6_mean,X_6_std,X_6_median])
    
    X_7 = X[:,7,:].reshape(X.shape[0],8,16)
    X_7_mean = np.mean(X_7,axis=2)
    X_7_std = np.std(X_7,axis=2)
    X_7_median = np.median(X_7,axis=2)
    X_7 = np.hstack([X_7_mean,X_7_std,X_7_median])
    
    X_8 = X[:,8,:].reshape(X.shape[0],8,16)
    X_8_mean = np.mean(X_8,axis=2)
    X_8_std = np.std(X_8,axis=2)
    X_8_median = np.median(X_8,axis=2)
    X_8 = np.hstack([X_8_mean,X_8_std,X_8_median])
    
    X_9 = X[:,9,:].reshape(X.shape[0],8,16)
    X_9_mean = np.mean(X_9,axis=2)
    X_9_std = np.std(X_9,axis=2)
    X_9_median = np.median(X_9,axis=2)
    X_9 = np.hstack([X_9_mean,X_9_std,X_9_median])
    
    a = []
    for i in range(X.shape[0]):
        c = pd.DataFrame(np.transpose(X[i,:,:])).corr().values
        c = np.ravel(c)
        a.append(c)
    a = np.asarray(a)    
    
    return np.hstack((X_test_mean, X_test_std,X_test_median,avg_speed,max_speed,min_speed,
                       X_test_max,X_test_min,
                      X_4_win_max,X_4_win_min,X_4_win_mean,X_4_win_median,
                      X_5_win_max,X_5_win_min,X_5_win_mean,X_5_win_median,
                      X_6_win_max,X_6_win_min,X_6_win_mean,X_6_win_median,
                      X_7_win_max,X_7_win_min,X_7_win_mean,X_7_win_median,
                      X_8_win_max,X_8_win_min,X_8_win_mean,X_8_win_median,
                      X_9_win_max,X_9_win_min,X_9_win_mean,X_9_win_median,
#                      X_0,X_1,X_2,X_3,
#                      X_4,X_5,X_6,     
#                      X_7,X_8,X_9
                      ))
groups = pd.read_csv('dataset/groups.csv')


X_test = np.load(r'dataset/X_test_kaggle.npy')
X_test_processed = process_x(X_test)

X_train = np.load(r'dataset/X_train_kaggle.npy')

#X_train_processed = process_x(X_train).reshape(X_train_processed.shape[0],10,10,1)
X_train_processed = process_x(X_train)


y_train_raw = pd.read_csv(r'dataset/y_train_final_kaggle.csv')

encoder = LabelEncoder()
encoder.fit(np.unique(y_train_raw['Surface']))
y_train = encoder.transform(y_train_raw['Surface'])


X_to_train, X_to_val, y_to_train, y_to_val = train_test_split(X_train_processed, y_train, test_size=0.2,random_state=4)


cv = GroupShuffleSplit(n_splits=1,test_size=0.2,random_state=1)

spl = cv.split(X_train_processed,y_train,groups = groups['Group Id'])

for train,test in spl:

    X_to_train_K = X_train_processed[train]
    X_to_val_K = X_train_processed[test]
    y_to_train_K = y_train[train]
    y_to_val_K = y_train[test]

y_nn = to_categorical(y_to_train_K)

model = Sequential()

model.add(Dense(100,input_dim = X_to_train_K.shape[1] ,activation='sigmoid'))
model.add(Dense(100,activation='relu'))
model.add(Dense(100,activation='sigmoid'))
model.add(Dense(100,activation='sigmoid'))
model.add(Dense(9,activation='softmax'))

model.summary()
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
#
#model.compile(optimizer = 'sgd',loss='categorical_crossentropy',
#              metrics = ['accuracy'],)

model.fit(X_to_train_K,y_nn,batch_size=32,epochs=200,validation_data=[X_to_val_K,to_categorical(y_to_val_K,9)])


#model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


#%%

#model.fit(X_to_train_K,y_nn,batch_size=32,epochs=200,validation_data=[X_to_val_K,to_categorical(y_to_val_K,9)])
#
#
##%%
#
#y_train_nn = to_categorical(y_train)
#          
#model.fit(X_train_processed,y_nn,batch_size=32,epochs=200)
#
#y_pred_nn = model.predict(X_test_processed,batch_size=32)
#y_pred = np.argmax(y_pred_nn,axis=1)
#
#proba_nn = model.predict_proba(X_test_processed,batch_size=32)

