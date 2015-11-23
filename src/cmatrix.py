import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns; sns.set(color_codes=True)
import pandas as pd

"""
Class for processing confusion matrix.
"""

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
	latency_range = max(latency) - min(latency)
	latency_range_k = latency_range//k
	latency_base = min(latency)
	length = len(latency)
	for i in xrange(length):
		label = (latency[i] - latency_base)//k
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
        sensitivity     -- floatting number, confusion matrix sensitivity
        specificity     -- floatting number, confusion matrix specificity
    """
	# generating confusion matrix
	cmatrix = metrics.confusion_matrix(y_true, y_pred)
	# getting TP, TN, FP, FN
	tp = cmatrix[1][1]
	tn = cmatrix[0][0]
	fp = cmatrix[0][1]
	fn = cmatrix[1][0]
	# specificity and sensitivity
	sensitivity = tp/(tp+fn)
	specificity = tn/(tn+fp)

	return sensitivity, specificity