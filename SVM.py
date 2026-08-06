#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 17:00:42 2026

@author: aziz
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize,StandardScaler
from sklearn.utils.class_weight import compute_sample_weight as csw
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.svm import LinearSVC

data = pd.read_csv("/home/aziz/Downloads/creditcard.csv")
print(data.head())

#Getting the set of distinct classes
labels = data.Class.unique()
#Getting class sizes
size = data.Class.value_counts().values

data['Class'].value_counts().plot(
    kind='pie', autopct='%1.3f%%', title='Target Variable Value Counts'
)

corr_val = data.corr()['Class'].drop('Class')
corr_val.plot(kind = 'barh')

#Standardizing features
data.iloc[:,1:30] = StandardScaler().fit_transform(data.iloc[:,1:30])
data_matrix = data.values

#X:feature matrix, We excude the time variable here
X = data_matrix[:,1:30]
#y:labels vector
y = data_matrix[:,30]
#Data normaization
X = normalize(X, norm="l1")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#Balancing the target variable 
weight_train = csw('balanced', y_train)

#Decision tree classifier
dt = DecisionTreeClassifier(max_depth=4, random_state=35)
dt.fit(X_train, y_train, sample_weight=weight_train)
y_pred_dt = dt.predict_proba(X_test)[:,1]
roc_auc_dt = roc_auc_score(y_test, y_pred_dt)
print('Decision Tree ROC-AUC score : {0:.3f}'.format(roc_auc_dt))

#Support vector machine
svm = LinearSVC(class_weight='balanced', random_state=31, loss="hinge", fit_intercept=False)
svm.fit(X_train, y_train)
y_pred_svm = svm.decision_function(X_test)
roc_auc_svm = roc_auc_score(y_test, y_pred_svm)
print("SVM ROC-AUC score: {0:.3f}".format(roc_auc_svm))


































