import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics import *
import seaborn as sns; sns.set(color_codes=True)
import pandas as pd

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


def setLabels(k, latency):
    """
    Set labels for different latencies.
    
    Attributes
    --------------------
        k          -- iinteger, indicating number of clusters
        latency    -- list of integers, list of latencies

    Returns
    --------------------
        labels             -- list of integers, list of labels
    """
    labels = []
    latency_range = int(max(latency) - min(latency))
    latency_range_k = int(latency_range*1.0/k)+1
    latency_base = min(latency)
    length = len(latency)
    for i in xrange(length):
        label = (latency[i] - latency_base)//latency_range_k
        labels.append(label)

    return labels


def cmatricstats(y_true, y_pred):
    """
    Calculate confusion matrix statistics (binary).
    
    Attributes
    --------------------
        y_true      -- list of integers, labels for different catagories
        y_pred      -- list of integers, labels for different catagories

    Returns
    --------------------
        sensitivity     -- list of floatting number, confusion matrix sensitivity
        specificity     -- list of floatting number, confusion matrix specificity
        precision       -- list of floatting number, confusion matrix precision
    """
    # generating confusion matrix
    cmatrix = confusion_matrix(y_true, y_pred)
    n,d = cmatrix.shape

    # getting TP, TN, FP, FN
    tp = [0.0] * n
    tn = [0.0] * n
    fp = [0.0] * n
    fn = [0.0] * n
    for i in xrange(n):
        for j in xrange(d):
            if i == j:
                tp[i] = cmatrix[i][j]
            if i != j:
                fn[i] += cmatrix[i][j]
                fp[j] += cmatrix[i][j]

    for i in xrange(n):
        for j in xrange(d):
            for k in xrange(n):
                if k != i and k != j:
                    tn[k] += cmatrix[i][j]
                
    # specificity and sensitivity
    tpfn = [x + y for x, y in zip(tp, fn)]
    sensitivity = [x / y for x, y in zip(tp, tpfn)]
    tnfp = [x + y for x, y in zip(tn, fp)]
    specificity = [x / y for x, y in zip(tn, tnfp)]
    tpfp = [x + y for x, y in zip(tp, fp)]
    precision = [x / y for x, y in zip(tp, tpfp)]

    return sensitivity, specificity, precision


def bicmatricstats(y_true, y_pred):
    # generating confusion matrix
    cmatrix = confusion_matrix(y_true, y_pred)

    # getting TP, TN, FP, FN
    tp = cmatrix[1][1]
    tn = cmatrix[0][0]
    fp = cmatrix[0][1]
    fn = cmatrix[1][0]

    # specificity and sensitivity
    sensitivity = tp/(tp+fn)
    specificity = tn/(tn+fp)
    precision = tp/(tp+fp)

    return sensitivity, specificity, precision