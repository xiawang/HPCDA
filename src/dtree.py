import os
import csv
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import cross_val_score

import matplotlib.pyplot as plt
import seaborn as sns; sns.set(color_codes=True)
import pandas as pd

from processdata import *
from optimize import *

"""
Class for training and predicting data with 
decision tree algorithm.
"""

def useDecisionTree(X, y, criterion='entropy'):
	dtclf = DecisionTreeClassifier(criterion=criterion)
	print cross_val_score(dtclf, X, y, cv=10)

def useCustomizedDecisionTree(X, y):
	pass

################################################################################
# training
################################################################################

data_lat = Data()
data_src = Data()
data_shar = Data()
data_cputid = Data()
data_ffs = Data()

data_lat.load('test_latency.csv',ify=False)
data_src.load('test_data_src.csv',ify=False)
data_shar.load('test_sharmetric.csv',ify=False)
data_cputid.load('test_tidupumetric.csv',ify=False)
data_ffs.load('test_ffsharing.csv',ify=False)

preview('test_latency.csv')

X_1,y_1 = data_lat.getXy()
X_2,y_2 = data_src.getXy()
X_3,y_3 = data_ffs.getXy()
X_4,y_4 = data_shar.getXy()
X_5,y_5 = data_cputid.getXy()

latency = []
data_src = []
ffshar = []
shar = []
cputid = []

for i in xrange(235446):
    if float(X_1[i][0]) < 500:
        latency.append(float(X_1[i][0]))
        data_src.append(float(X_2[i][0]))
        ffshar.append(float(X_3[i][0]))
        shar.append(float(X_4[i][0]))
        cputid.append(float(X_5[i][0]))

my_list = zip(data_src,ffshar,shar,cputid)

# catagorize latency
max_lat = 0
for i in xrange(len(latency)):
	temp = latency[i]
	if latency[i] >= max_lat:
		max_lat = latency[i]
	latency[i] = temp//60

print max_lat

useDecisionTree(my_list, latency)