import os
import csv
import numpy as np
np.random.seed(sum(map(ord, "regression")))
import seaborn as sns; sns.set(color_codes=True)
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats

from mpl_toolkits.mplot3d import Axes3D

from processdata import *
from kmeans import *
from optimize import *
from kmeans import *

'''
    This file contains the method for checking relationship between the
    latency and cache distribution.
'''


def checkCache():
	"""
	Customized function for checking the latency and ffs metric.
	"""
	# first read in the latency
	data1 = Data()
	data1.load('test_latency.csv',ify=False)
	data2 = Data()
	data2.load('test_data_src.csv',ify=False)

	X_1,y_1 = data1.getXy()
	X_2,y_2 = data2.getXy()
	feature1 = []
	feature2 = []
	print "Latency & cache data loaded..."

	for i in range(235446):
	    feature1.append(float(X_1[i][0]))
	    feature2.append(float(X_2[i][0]))
	print "Latency & cache data converted to numpy array..."

	# plot using pandas and seaborn
	# g = sns.distplot(feature1);
	fig = plt.figure()
	fig.suptitle('Caching level', fontsize=12, fontweight='bold')
	ax = fig.add_subplot(111)
	ax.plot(feature2, feature1, 'b.')
	ax.set_xlabel('caching level')
	ax.set_ylabel('latency (cycles)')

	axes = plt.gca()
	axes.set_xlim([-4,6])

	# sns.plt.show();
	plt.show();
	print "checkLatency passed..." + '\n'

# x = {1,2,3,4,5}
# y = {2,4,6,8,10}
# sns.regplot(x=x, y=y);
# sns.plt.show();

checkCache()