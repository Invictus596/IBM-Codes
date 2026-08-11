#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  8 17:17:05 2026

@author: aziz
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
#import umap.umap_ as umap
from sklearn.manifold import TSNE as ts
from sklearn.decomposition import PCA as pca

# CLuster centers:
centers = [ [ 2, -6, -6],
            [-1,  9,  4],
            [-8,  7,  2],
            [ 4,  7,  9] ]

# Cluster standard deviations:
cluster_std=[1,1,2,3.5]

X, labels_ = make_blobs(n_samples = 500,centers = centers, n_features = 3, cluster_std = cluster_std, random_state = 42)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

tsne = ts(n_components = 2, random_state=42, perplexity = 30, max_iter = 1000)
X_tsne = tsne.fit_transform(X_scaled)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111)
ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels_, cmap='viridis', s=50, alpha=0.7, edgecolor='k')
ax.set_title("2D t-SNE Projection of 3D Data")
ax.set_xlabel("t-SNE Component 1")
ax.set_ylabel("t-SNE Component 2")
ax.set_xticks([])
ax.set_yticks([])
plt.show()

"""
t-SNE projected the data into four distinct clusters, although the original data had some overlap between a few clusters.
You can see that some of the points ended up in the "wrong" cluster, although to be fair, t-SNE has no knowledge of which clusters the points actually belong to.
All the clusters have similar densities.
Two of the blobs are distinct from each other but "gave up" some of their points to the blob they originally had overlapped with.
A "perfect" result would not completely separate the overlaps between blobs.
Notice that the distance between the blobs is consistent with the degree to which they were originally separated.
"""

