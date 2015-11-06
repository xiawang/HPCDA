import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans, KMeans
import seaborn as sns; sns.set(color_codes=True)
import pandas as pd
from processdata import *

"""
Class for training and predicting data with 
kmeans algorithm.
"""

def useKMeans(X, y, init='k-means++', n_clusters=4, n_init=10):
	"""
    Clustering data with k-menas algorithm and proper parameters.
    
    Parameters
    --------------------
        X              -- 2D array, features
        y              -- array, labels (expected to be an empty array)
        init           -- string, indicating method for initialization
                          one of {'k-means++', 'random' or an ndarray}
        n_clusters     -- integer, number of clusters to form
        n_init         -- integer, number of time the k-means algorithm
                          will be run with different centroid seeds
    Returns
    --------------------
        y              -- array, labels corresponding to data points
        centers        -- 2D array, coordinates of cluster centers
        labels_unique  -- array, all different labels
    """
	k_means = KMeans(init='k-means++', n_clusters=n_clusters, n_init=n_init)
	k_means.fit(X)

	y = k_means.labels_ # setting an input empty array
	centers = k_means.cluster_centers_
	labels_unique = np.unique(y)
	return y, centers, labels_unique


def plotKMeans(X, y, centers, n_clusters, x_l, x_h, y_l, y_h):
    """
    Plotting classification with kmeans algorithm.

    Parameters
    --------------------
        X              -- 2D array, features
        y              -- array, labels (not empty)
        centers        -- 2D array, coordinates of cluster centers
        n_clusters     -- integer, number of clusters to form
    """
    for i in range(n_clusters):
        ds = []
        for j in range(len(y)):
            if y[j] == i:
                ds.append(X[j])
        ds = np.array(ds)
        plt.plot(ds[:,0],ds[:,1],'o')
        lines = plt.plot(centers[i,0],centers[i,1],'kx')
        plt.setp(lines,ms=5.0)
        plt.setp(lines,mew=1)
    axes = plt.gca()
    axes.set_xlim([x_l,x_h])
    axes.set_ylim([y_l,y_h])
    plt.show()

