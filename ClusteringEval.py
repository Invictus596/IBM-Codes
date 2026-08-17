#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 11:54:03 2026

@author: aziz
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score, silhouette_samples, davies_bouldin_score

#Generate synthetic data
X,_ = make_blobs(n_samples = 300,centers=4,cluster_std=1.5,random_state=42)

#Fit KMeans
n_clusters = 4
kmeans = KMeans(n_clusters = n_clusters, random_state = 42,n_init=10)
labels = kmeans.fit_predict(X)

#Computing metrics
sil_avg = silhouette_score(X,labels)           # higher is better (range -1 to 1)
sil_samples = silhouette_samples(X, labels)    # per-point silhouette values
db_score = davies_bouldin_score(X, labels)     # lower is better (0 = best)
inertia = kmeans.inertia_                      # lower is better, but always drops as k rises

print(f"Silhouette Score (avg): {sil_avg:.3f}   (higher is better, range -1 to 1)")
print(f"Davies-Bouldin Score:   {db_score:.3f}   (lower is better, 0 = best)")
print(f"Inertia:                {inertia:.2f}   (lower is better)")

#Visualization
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
 
# -- Left: the clusters themselves --
axes[0].scatter(X[:, 0], X[:, 1], c=labels, cmap='tab10', s=50, alpha=0.7, edgecolor='k')
axes[0].scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
                c='red', marker='X', s=200, label='Centroids')
axes[0].set_title(f'KMeans Clusters (k={n_clusters})')
axes[0].set_xlabel('Feature 1')
axes[0].set_ylabel('Feature 2')
axes[0].legend()
 
# -- Right: silhouette plot (shows how well each point fits its cluster) --
y_lower = 10
for i in range(n_clusters):
    cluster_vals = np.sort(sil_samples[labels == i])
    y_upper = y_lower + len(cluster_vals)
    axes[1].fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_vals, alpha=0.7)
    axes[1].text(-0.05, y_lower + 0.5 * len(cluster_vals), str(i))
    y_lower = y_upper + 10
 
axes[1].axvline(x=sil_avg, color='red', linestyle='--', label=f'Avg = {sil_avg:.2f}')
axes[1].set_title('Silhouette Plot')
axes[1].set_xlabel('Silhouette Coefficient')
axes[1].set_ylabel('Cluster')
axes[1].set_yticks([])
axes[1].legend()
 
plt.tight_layout()








