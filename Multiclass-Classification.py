#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 18:23:46 2026

@author: aziz
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsOneClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.multiclass import OneVsRestClassifier

url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/GkDzb7bWrtvGXdPOfk6CIg/Obesity-level-prediction-dataset.csv"
data = pd.read_csv(url)
data.head()

sns.countplot(y='NObeyesdad', data=data)
plt.title('Distribution of Obesity Levels')
plt.show()

print(data.info())
print(data.describe())

#Standardizing continuous numerical features
continous_columns = data.select_dtypes(include=['float64']).columns.tolist()
#Scaling
scaler = StandardScaler()
scaled_feat = scaler.fit_transform(data[continous_columns])
#Converting to a dataframe
scaled_df = pd.DataFrame(scaled_feat , columns = scaler.get_feature_names_out(continous_columns))
#Combining with original dataset
scaled_data = pd.concat([data.drop(columns=continous_columns), scaled_df], axis =1)

#One hot encoding:
#Identifying categorical columns
categorical_columns = scaled_data.select_dtypes(include=['object']).columns.tolist()
categorical_columns.remove('NObeyesdad')  # Exclude target column
#Applying one-hot encoding
encoder = OneHotEncoder(sparse_output=False, drop='first')
encoded_features = encoder.fit_transform(scaled_data[categorical_columns])
#Converting to a DataFrame
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_columns))
#Combining with the original dataset
prepped_data = pd.concat([scaled_data.drop(columns=categorical_columns), encoded_df], axis=1)
#Encoding the target variable
prepped_data['NObeyesdad'] = prepped_data['NObeyesdad'].astype('category').cat.codes

#Defining variables
X = prepped_data.drop('NObeyesdad', axis=1)
y = prepped_data['NObeyesdad']
#Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

#Training logistic regression model using One-vs-All (default)
model_ova = OneVsRestClassifier(make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000, random_state=42)
))
model_ova.fit(X_train, y_train)

#Predictions
y_pred_ova = model_ova.predict(X_test)

# valuation metrics for OvA
acc = model_ova.score(X_test, y_test)
print(f"One-vs-All (OvA) Accuracy: {acc:.2%}")

model_ovo = OneVsOneClassifier(
    make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
)
model_ovo.fit(X_train, y_train)
#Predict & evaluate using the simpler .score() method
y_pred_ovo = model_ovo.predict(X_test)
acc = model_ovo.score(X_test, y_test)
print(f"One-vs-One (OvO) Accuracy: {acc:.2%}")

matching_predictions = np.sum(y_pred_ova == y_pred_ovo)
total_samples = len(y_test)

print(f"Both models agreed on {matching_predictions} out of {total_samples} test samples.")







