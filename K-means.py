#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 15:55:50 2026

@author: aziz
"""

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans 
from sklearn.datasets import make_blobs 
from sklearn.preprocessing import StandardScaler
import seaborn as sns

np.random.seed(0)

"""np.random.seed(0) sets the starting point (the seed) for NumPy's pseudorandom number generator.
When you run code that generates random numbers, computers don't create true randomness—
they use algorithms that generate deterministic sequences based on an initial number.
"""

X, y = make_blobs(n_samples=5000, centers=[[4,4], [-2, -1], [2, -3], [1, 1]], cluster_std=0.9)

"""
Next, we will be making random clusters of points by using the make_blobs class. 
The make_blobs class can take in many inputs,but we will be using these specific ones.

Input

n_samples: The total number of points equally divided among clusters.
Value will be: 5000
centres : The number of centres to generate, or the fixed centre locations.
Value will be: [[4, 4], [-2, -1], [2, -3],[1,1]]
cluster_std: The standard deviation of the clusters.
Value will be: 0.9

Output

X: Array of shape [n_samples, n_features]. (Feature Matrix)
The generated samples.
y: Array of shape [n_samples]. (Response Vector)
The integer labels for cluster membership of each sample.
"""
#Display the scatter plot of randomly generated data
plt.scatter(X[:, 0], X[:, 1], marker='.',alpha=0.3,ec='k',s=80)

#Setting up k-means
k_means = KMeans(init = "k-means++", n_clusters = 4, n_init = 12)

"""
The KMeans class has many parameters that can be used, but we will be using these three:

init: Initialization method of the centroids.
Value will be: k-means++
               k-means++: Selects initial cluster centres for k-means clustering in a smart way to speed up convergence.
n_clusters: The number of clusters to form as well as the number of centroids to generate.
Value will be: 4 (since we have 4 centres)
n_init: Number of times the k-means algorithm will be run with different centroid seeds. 
The final results will be the best output of n_init consecutive runs in terms of inertia.
Value will be: 12

"""
k_means.fit(X)

#Getting labels for each point in the model
k_label = k_means.labels_

#Coordinates of the cluster centers 
k_Clustercentes = k_means.cluster_centers_

#Visualizing clusters
# Initialize the plot with the specified dimensions.
fig = plt.figure(figsize=(6, 4))

# Colors uses a color map, which will produce an array of colors based on
# the number of labels there are. We use set(k_means_labels) to get the
# unique labels.
colors = plt.cm.tab10(np.linspace(0, 1, len(set(k_label))))

# Create a plot
ax = fig.add_subplot(1, 1, 1)

# For loop that plots the data points and centroids.
# k will range from 0-3, which will match the possible clusters that each
# data point is in.
for k, col in zip(range(len([[4, 4], [-2, -1], [2, -3], [1, 1]])), colors):

    # Create a list of all data points, where the data points that are 
    # in the cluster (ex. cluster 0) are labeled as true, else they are
    # labeled as false.
    my_members = (k_label == k)

    # Define the centroid, or cluster center.
    cluster_center = k_Clustercentes[k]

    # Plots the datapoints with color col.
    ax.plot(X[my_members, 0], X[my_members, 1], 'w', markerfacecolor=col, marker='.',ms=10)

    # Plots the centroids with specified color, but with a darker outline
    ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,  markeredgecolor='k', markersize=6)

# Title of the plot
ax.set_title('KMeans')

# Remove x-axis ticks
ax.set_xticks(())

# Remove y-axis ticks
ax.set_yticks(())

# Show the plot
plt.show()

#Customer segmentation with k-means

#Load data
cust_df = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%204/data/Cust_Segmentation.csv")

#Drop the categorical column "Address" because k-means uses euclidean distnace which doesnt work on categorical variables
cust_df = cust_df.drop('Address', axis=1)

cust_df.info()

"""
RangeIndex: 850 entries, 0 to 849
Data columns (total 9 columns):
 #   Column           Non-Null Count  Dtype  
---  ------           --------------  -----  
 0   Customer Id      850 non-null    int64  
 1   Age              850 non-null    int64  
 2   Edu              850 non-null    int64  
 3   Years Employed   850 non-null    int64  
 4   Income           850 non-null    int64  
 5   Card Debt        850 non-null    float64
 6   Other Debt       850 non-null    float64
 7   Defaulted        700 non-null    float64
 8   DebtIncomeRatio  850 non-null    float64
 
 As we have null values here we just drop it and work with a smaller dataset
"""

cust_df = cust_df.dropna()

#Nomalizing the dataset except the customerID column because of obvious reasons
X2 = cust_df.values[:,1:]
scaler = StandardScaler()
clus_df = scaler.fit_transform(X2)

k_means2 = KMeans(init = "k-means++", n_clusters = 3, n_init=10)
k_means2.fit(X2)
labels = k_means2.labels_

cust_df["Clus_km"] = labels

#Checking the centorid values of each cluster 
k_centroids = k_means2.cluster_centers_
#This method returns normalized cluster centres 
#2nd method
k_centroids2 = cust_df.groupby('Clus_km').mean()
#Returns original (Arithmetic mean) cluster centers

#Visualization
area = np.pi * ( X[:, 1])**2  
plt.scatter(X2[:, 0], X2[:, 3], s=area, c=labels.astype(float), cmap='tab10', ec='k',alpha=0.5)
plt.xlabel('Age', fontsize=18)
plt.ylabel('Income', fontsize=16)
plt.show()











