#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 14:17:12 2026

@author: aziz
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error as mse, r2_score

url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/UZPRFNucrENAFm25csq6eQ/California-housing.csv'
df = pd.read_csv(url)

# Separate features and target
X = df.drop(columns=["Target"])
y = df["Target"]

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#Initialize model
n = 100
rf = RandomForestRegressor(n_estimators = n,random_state = 42)
#fit model
rf.fit(X_train,y_train)
#Prediction
y_pred = rf.predict(X_test)

#Evaluation metrics
mse_rf = mse(y_pred,y_test)
print(mse_rf)
r2 = r2_score(y_pred,y_test)
print(r2)
#visualizing results
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
plt.axline((0, 0), slope=1, color='black', linestyle='--')
plt.title("Random Forest Predictions vs Actual")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.legend()
plt.show()


