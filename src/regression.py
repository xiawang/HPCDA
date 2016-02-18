import os
import csv
import numpy as np
from sklearn import linear_model
from sklearn.metrics import *
from sklearn.cross_validation import *
from sklearn.decomposition import PCA

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
	print "linear decision function: "
	print lnregr.coef_, " X + ", lnregr.intercept_
	# y_pred = lnregr.predict(X)

	# plotting data
	# fig = plt.figure()
	# ax = fig.add_subplot(111)
	# fig.suptitle('linear regression', fontsize=12, fontweight='bold')
	# length = len(y_pred)
	# xa = list(range(length))
	# for i in xrange(length):
	# 	ax.plot(y[i], xa[i], 'bo')
	# 	ax.plot(y_pred[i], xa[i], 'ro')

	# plt.show();


	# mse = mean_squared_error(y, y_pred)
	# print "Mean squared error regression loss: ", mse
	# mae = mean_absolute_error(y, y_pred)
	# print "Mean absolute error regression loss: ", mae
	# cv_scores = cross_val_score(lnregr, X, y, cv=11, scoring=scorer)
	# print "Cross validation scores: ", cv_scores


def useRidgeRegression(X, y, alpha=0.1, max_iter=None):
	riregr = linear_model.Ridge(alpha=alpha, max_iter=max_iter)
	riregr.fit(X, y)
	# y_pred = riregr.predict(X)
	# accuracy = accuracy_score(y, y_pred)
	# print "accuracy: ", accuracy
	print "ridge decision function: "
	print riregr.coef_, " X + ", riregr.intercept_
	# y_pred = riregr.predict(X)
	# mse = mean_squared_error(y, y_pred)
	# print "Mean squared error regression loss: ", mse
	# mae = mean_absolute_error(y, y_pred)
	# print "Mean absolute error regression loss: ", mae
	# cv_scores = cross_val_score(riregr, X, y, cv=11, scoring=scorer)
	# print "Cross validation scores: ", cv_scores


def useLasso(X, y, alpha=0.1):
	lsregr = linear_model.Lasso(alpha=alpha)
	lsregr.fit(X, y)
	# y_pred = riregr.predict(X)
	# accuracy = accuracy_score(y, y_pred)
	# print "accuracy: ", accuracy
	print "ridge decision function: "
	print lsregr.coef_, " X + ", lsregr.intercept_
	# y_pred = lsregr.predict(X)
	# mse = mean_squared_error(y, y_pred)
	# print "Mean squared error regression loss: ", mse
	# mae = mean_absolute_error(y, y_pred)
	# print "Mean absolute error regression loss: ", mae
	# cv_scores = cross_val_score(lsregr, X, y, cv=11, scoring=scorer)
	# print "Cross validation scores: ", cv_scores


def my_custom_loss_func(ground_truth, predictions):
	"""
	Customized scoring function for regression model.
	"""
	total = len(predictions)
	diff = np.abs(ground_truth - predictions)
	truth_list = map(lambda x: x<40, diff)
	truth_val = sum(truth_list)
	return truth_val*1.0/total

scorer  = make_scorer(my_custom_loss_func, greater_is_better=True)

################################################################################
# training
################################################################################

data_lat = Data()
data_src = Data()
data_shar = Data()
data_cputid = Data()
cpu = Data()
data_x = Data()
data_y = Data()
data_z = Data()
data_ffshar = Data()

data_lat.load('test_latency.csv',ify=False)
data_src.load('test_data_src.csv',ify=False)
data_shar.load('test_sharmetric.csv',ify=False) # pipeline
data_cputid.load('test_tidupumetric.csv',ify=False)
cpu.load('test_cpu.csv',ify=False)
data_x.load('test_x.csv',ify=False)
data_y.load('test_y.csv',ify=False)
data_z.load('test_z.csv',ify=False)
data_ffshar.load('test_ffsharing_5.csv',ify=False) # fuzzy false sharing

X_1,y_1 = data_lat.getXy()
X_2,y_2 = data_src.getXy()
X_3,y_3 = cpu.getXy()
X_4,y_4 = data_shar.getXy()
X_5,y_5 = data_cputid.getXy()
X_6,y_6 = data_x.getXy()
X_7,y_7 = data_y.getXy()
X_8,y_8 = data_z.getXy()
X_9,y_9 = data_ffshar.getXy()

# latency = []
# data_src = []
# CPU = []
# shar = []
# cputid = []

# for i in xrange(235446):
# 	if float(X_1[i][0]) < 120:
# 		latency.append(float(X_1[i][0]))
# 		data_src.append(float(X_2[i][0]))
# 		CPU.append(float(X_3[i][0]))
# 		shar.append(float(X_4[i][0]))
# 		cputid.append(float(X_5[i][0]))

# my_list = zip(data_src,CPU,shar,cputid)

# useLinearRegression(my_list, latency)
# useRidgeRegression(my_list, latency)

# examine most useful part
latency_s = []
data_src_s = []
CPU_s = []
shar_s = []
cputid_s = []
ds_x = []
ds_y = []
ds_z = []
ffshar_s = []

for i in xrange(235446):
	if float(X_1[i][0]) > 0 and float(X_1[i][0]) < 500: # only consider if latency is small
		latency_s.append(float(X_1[i][0]))
		data_src_s.append(float(X_2[i][0]))
		CPU_s.append(float(X_3[i][0]))
		shar_s.append(float(X_4[i][0]))
		cputid_s.append(float(X_5[i][0]))
		ds_x.append(float(X_6[i][0]))
		ds_y.append(float(X_7[i][0]))
		ds_z.append(float(X_8[i][0]))
		ffshar_s.append(float(X_9[i][0]))

my_list_s = zip(data_src_s,CPU_s,shar_s,cputid_s,ds_x,ds_y,ds_z,ffshar_s)

print "Linear: "
useLinearRegression(my_list_s, latency_s)
print ""

# print "Linear - normalized: "
# useLinearRegression(my_list_s, latency_s, fit_intercept=True, normalize=True)
# print ""

print "Ridge: "
useRidgeRegression(my_list_s, latency_s, alpha=0.05)
print ""

print "Lasso: "
useLasso(my_list_s, latency_s, alpha=0.001)
print ""