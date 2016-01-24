import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics import *
import seaborn as sns; sns.set(color_codes=True)
import pandas as pd
from processdata import *
from cmatrix import *

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


################################################################################
# training and clustering
################################################################################

def trainingkm():
    for q in xrange(2,5):
        data_lat = Data()
        data_src = Data()
        data_shar = Data()
        data_cputid = Data()
        cpu = Data()

        data_lat.load('test_latency.csv',ify=False)
        data_src.load('test_data_src.csv',ify=False)
        data_shar.load('test_sharmetric.csv',ify=False)
        data_cputid.load('test_tidupumetric.csv',ify=False)
        cpu.load('test_cpu.csv',ify=False)

        X_1,y_1 = data_lat.getXy()
        X_2,y_2 = data_src.getXy()
        X_3,y_3 = cpu.getXy()
        X_4,y_4 = data_shar.getXy()
        X_5,y_5 = data_cputid.getXy()

        latency = []
        data_src = []
        CPU = []
        shar = []
        cputid = []

        for i in xrange(235446):
            if float(X_1[i][0]) < 400:
                latency.append(float(X_1[i][0]))
                data_src.append(float(X_2[i][0]))
                CPU.append(float(X_3[i][0]))
                shar.append(float(X_4[i][0]))
                cputid.append(float(X_5[i][0]))

        my_list = zip(data_src,CPU,shar,cputid)

        y = []
        y, centers, labels_unique = useKMeans(my_list, y, n_clusters=q)
        # plotKMeans(my_list,y,centers,4,0,6,-5,40)

        labels = setLabels(q, latency)
        sensitivity, specificity, precision = cmatricstats(labels, y)

        # print "labels: ", labels
        label_set = set(labels)

        f1 = f1_score(labels, y)
        accuracy = accuracy_score(labels, y)

        print "granularity: ", q

        print "sensitivity: ", sensitivity
        print "specificity: ", specificity
        print "precision: ", precision
        print "f1 score: ", f1
        print "accuracy: ", accuracy


# trainingkm()