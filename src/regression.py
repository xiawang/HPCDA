import os
import csv
import numpy as np
from sklearn import linear_model
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
import seaborn as sns; sns.set(color_codes=True)
import pandas as pd

from processdata import *
from optimize import *

"""
Class for training and predicting data with 
regression algorithm.
"""

def useLinearRegression(X, y, fit_intercept=True, normalize=False):
	lnregr = linear_model.LinearRegression(fit_intercept=fit_intercept, normalize=normalize)
	lnregr.fit(X, y)
	# y_pred = lnregr.predict(X)
	# accuracy = accuracy_score(y, y_pred)
	# print "accuracy: ", accuracy
	print "linear decision function: ", lnregr.coef_, " X + ", lnregr.intercept_


def useRidgeRegression(X, y, alpha=1.0, max_iter=None):
	riregr = linear_model.Ridge(alpha=alpha, max_iter=max_iter)
	riregr.fit(X, y)
	# y_pred = riregr.predict(X)
	# accuracy = accuracy_score(y, y_pred)
	# print "accuracy: ", accuracy
	print "ridge decision function: ", riregr.coef_, " X + ", riregr.intercept_

data_lat = Data()
data_src = Data()
cpu = Data()

data_lat.load('test_latency.csv',ify=False)
data_src.load('test_data_src.csv',ify=False)
cpu.load('test_cpu.csv',ify=False)
X_1,y_1 = data_lat.getXy()
X_2,y_2 = data_src.getXy()
X_3,y_3 = cpu.getXy()
latency = []
data_src = []
CPU = []

for i in xrange(235446):
	latency.append(float(X_1[i][0]))
	data_src.append(float(X_2[i][0]))
	CPU.append(float(X_3[i][0]))

my_list = zip(data_src,CPU)

useLinearRegression(my_list, latency)
useRidgeRegression(my_list, latency)