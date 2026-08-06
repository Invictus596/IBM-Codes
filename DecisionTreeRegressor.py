#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 18:44:26 2026

@author: aziz
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import normalize
from sklearn.metrics import mean_squared_error

url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/pu9kbeSaAtRZ7RxdJKX9_A/yellow-tripdata.csv'
raw_data = pd.read_csv(url)
raw_data

correlation_values = raw_data.corr()['tip_amount'].drop('tip_amount')
correlation_values.plot(kind='barh', figsize=(10, 6))
print(correlation_values)

#Data Preprocessing
# extract the labels from the dataframe
y = raw_data['tip_amount'].values.astype('float32')
# drop the target variable from the feature matrix
proc_data = raw_data.drop(['tip_amount'], axis=1)
# get the feature matrix used for training
X = proc_data.values
# normalize the feature matrix
X = normalize(X, axis=1, norm='l1', copy=False)
#Here we use l1(lasso) normalization because we have too many irrelevant features
#and we want to keep the model simple
#We use l2(Ridge) normalization when we have too many correlated features and wants to 
#prevent overfitting

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
dt_reg = DecisionTreeRegressor(criterion = 'squared_error',max_depth=12,random_state=35)
dt_reg.fit(X_train, y_train)
y_pred = dt_reg.predict(X_test)

mse_score = mean_squared_error(y_test, y_pred)
print('MSE score : {0:.3f}'.format(mse_score))

r2_score = dt_reg.score(X_test,y_test)
print('R^2 score : {0:.3f}'.format(r2_score))