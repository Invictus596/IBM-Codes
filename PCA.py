#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  8 16:07:45 2026

@author: aziz
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA 
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
mean = [0, 0]
cov = [[3, 2], [2, 2]]
X = np.random.multivariate_normal(mean=mean, cov=cov, size=200)

plt.figure()
plt.scatter(X[:, 0], X[:, 1],  edgecolor='k', alpha=0.7)
plt.title("Scatter Plot of Bivariate Normal Distribution")
plt.xlabel("X1")
plt.ylabel("X2")
plt.axis('equal')
plt.grid(True)
plt.show()

#initializing 2 component pca model 
pca = pca(n_components=2)
X_pca = pca.fit_transform(X)

"""The principal components are the principal axes, represented in feature space coordinates,
which align with the directions of maximum variance in your data.
"""
comp = pca.components_
print(comp)

"""
The principal components are sorted in decreasing order by their explained variance,
which can be expressed as a ratio:
"""
rat = pca.explained_variance_ratio_
print(rat)

projection_pc1 = np.dot(X, comp[0])
projection_pc2 = np.dot(X, comp[1])

x_pc1 = projection_pc1 * comp[0][0]
y_pc1 = projection_pc1 * comp[0][1]
x_pc2 = projection_pc2 * comp[1][0]
y_pc2 = projection_pc2 * comp[1][1]

plt.figure()
plt.scatter(X[:, 0], X[:, 1], label='Original Data', ec='k', s=50, alpha=0.6)

# Plot the projections along PC1 and PC2
plt.scatter(x_pc1, y_pc1, c='r', ec='k', marker='X', s=70, alpha=0.5, label='Projection onto PC 1')
plt.scatter(x_pc2, y_pc2, c='b', ec='k', marker='X', s=70, alpha=0.5, label='Projection onto PC 2')
plt.title('Linearly Correlated Data Projected onto Principal Components', )
plt.xlabel('Feature 1',)
plt.ylabel('Feature 2',)
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()


#PART 2

#Here we try to reduce a 4D iris dataset into 2D
# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

# Standardize the data
scaler = StandardScaler()
X_scaled2 = scaler.fit_transform(X)

#init pca model 
pca2 = PCA(n_components=2)
X_pca2 = pca.fit_transform(X_scaled2)

# Plot the PCA-transformed data in 2D
plt.figure(figsize=(8,6))

colors = ['navy', 'turquoise', 'darkorange']
lw = 1

for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(X_pca2[y == i, 0], X_pca2[y == i, 1], color=color, s=50, ec='k',alpha=0.7, lw=lw,
                label=target_name)

plt.title('PCA 2-dimensional reduction of IRIS dataset',)
plt.xlabel("PC1",)
plt.ylabel("PC2",)
plt.legend(loc='best', shadow=False, scatterpoints=1,)
# plt.grid(True)
plt.show()

# Explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_

# Plot explained variance ratio for each component
plt.figure(figsize=(10,6))
plt.bar(x=range(1, len(explained_variance_ratio)+1), height=explained_variance_ratio, alpha=1, align='center', label='PC explained variance ratio' )
plt.ylabel('Explained Variance Ratio')
plt.xlabel('Principal Components')
plt.title('Explained Variance by Principal Components')














