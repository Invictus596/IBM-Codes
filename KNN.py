#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 11:35:30 2026

@author: aziz
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/teleCust1000t.csv'
df = pd.read_csv(url)
df.head()

df['custcat'].value_counts()

correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, fmt='.2f')

correlation_values = abs(df.corr()['custcat'].drop('custcat')).sort_values(ascending=False)

X = df.drop('custcat',axis=1)
y = df['custcat']

#it is important to normalize in a KNN Alg because it uses distance as a calculation for selection
X_norm = StandardScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.2, random_state=4)

#Training the model
k = 6
knn = KNeighborsClassifier(n_neighbors=k)
knn_model = knn.fit(X_train,y_train)

#predicting
yhat = knn_model.predict(X_test)

#Finding accuracy
print("Accuracy:",accuracy_score(y_test,yhat))

Ks = 10
acc = np.zeros((Ks))
std_acc = np.zeros((Ks))
for n in range(1,Ks+1):
    #Train Model and Predict  
    knn_model_n = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)
    yhat = knn_model_n.predict(X_test)
    acc[n-1] = accuracy_score(y_test, yhat)
    std_acc[n-1] = np.std(yhat==y_test)/np.sqrt(yhat.shape[0])
    print(acc,std_acc)

plt.plot(range(1,Ks+1),acc,'g')
plt.fill_between(range(1,Ks+1),acc - 1 * std_acc,acc + 1 * std_acc, alpha=0.10)
plt.legend(('Accuracy value', 'Standard Deviation'))
plt.ylabel('Model Accuracy')
plt.xlabel('Number of Neighbors (K)')
plt.tight_layout()
plt.show()

