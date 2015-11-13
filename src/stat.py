import os
import csv
import numpy as np; np.random.seed(0)
import seaborn as sns; sns.set(color_codes=True)
import matplotlib.pyplot as plt
import pandas as pd

from mpl_toolkits.mplot3d import Axes3D

from processdata import *
from kmeans import *
from optimize import *

def checkTime():
	"""
	Customized function for checking the time feature.
	"""
	# first read in some features from sample
	data = Data()
	ft1 = extract('samples.csv', 13, start=1)
	print "Data loaded..."

	# do some optimization
	ft1 = bsc_min_standardization(ft1)
	ft1 = toHour(ft1)
	print "Data optimized..."

	# then write out tesing csv
	my_list = zip(ft1)
	writeCSV('test_time.csv', my_list)
	print "Data written..."

	# load testing csv and plot
	data.load('test_time.csv',ify=False)
	X,y = data.getXy()
	feature1 = []
	print "Time data loaded..."

	for i in range(235446):
	    feature1.append(float(X[i][0]))
	print "Time data converted to numpy array..."

	# plot using pandas and seaborn
	g = sns.distplot(feature1);

	sns.plt.show();
	print "checkTime passed..." + '\n'


def checkLatency():
	"""
	Customized function for checking the time feature.
	"""
	# first read in some features from sample
	data = Data()
	ft1 = extract('samples.csv', 16, start=1)
	print "Data loaded..."

	# do some optimization
	ft1 = toFloat(ft1)
	unq_latency = np.unique(ft1)
	print unq_latency.size
	print "Data optimized..."

	# then write out tesing csv
	my_list = zip(ft1)
	writeCSV('test_latency.csv', my_list)
	print "Data written..."

	# load testing csv and plot
	data.load('test_latency.csv',ify=False)
	X,y = data.getXy()
	feature1 = []
	print "Latency data loaded..."

	for i in range(235446):
	    feature1.append(float(X[i][0]))
	print "Latency data converted to numpy array..."

	# plot using pandas and seaborn
	g = sns.distplot(feature1);

	sns.plt.show();
	print "checkLatency passed..." + '\n'

def checkDataSrc():
	"""
	Customized function for checking the data_src feature.
	"""
	# first read in some features from sample
	data = Data()
	ft1 = extract('samples.csv', 17, start=1)
	print "Data loaded..."

	# do some optimization
	ft1 = toInteger(ft1)
	ft1 = map(lambda x: map_data_src(x), ft1)
	print "Data optimized..."

	# then write out tesing csv
	my_list = zip(ft1)
	writeCSV('test_data_src.csv', my_list)
	print "Data written..."

	# load testing csv and plot
	data.load('test_data_src.csv',ify=False)
	X,y = data.getXy()
	feature1 = []
	print "Data_src loaded..."

	for i in range(235446):
	    feature1.append(float(X[i][0]))
	print "Data_src converted to numpy array..."

	# plot using pandas and seaborn
	g = sns.distplot(feature1);

	sns.plt.show();
	print "checkDataSrc passed..." + '\n'

def checkAddr():
	"""
	Customized function for checking the address feature.
	"""
	# first read in some features from sample
	data = Data()
	ft1 = extract('samples.csv', 14, start=1)
	print "Data loaded..."

	# do some optimization
	ft1 = toFloat(ft1)
	unq_addr = np.unique(ft1)
	print unq_addr.size
	print "Data optimized..."

	# then write out tesing csv
	my_list = zip(ft1)
	writeCSV('test_addr.csv', my_list)
	print "Data written..."

	# load testing csv and plot
	data.load('test_addr.csv',ify=False)
	X,y = data.getXy()
	feature1 = []
	print "Address data loaded..."

	for i in range(235446):
	    feature1.append(float(X[i][0]))
	print "Address data converted to numpy array..."

	# plot using pandas and seaborn
	g = sns.distplot(feature1);

	sns.plt.show();
	print "checkAddr passed..." + '\n'

def checkCPU():
	"""
	Customized function for checking the CPU feature.
	"""
	# first read in some features from sample
	data = Data()
	ft1 = extract('samples.csv', 15, start=1)
	print "Data loaded..."

	# do some optimization
	ft1 = toFloat(ft1)
	unq_addr = np.unique(ft1)
	print unq_addr.size
	print "Data optimized..."

	# then write out tesing csv
	my_list = zip(ft1)
	writeCSV('test_cpu.csv', my_list)
	print "Data written..."

	# load testing csv and plot
	data.load('test_cpu.csv',ify=False)
	X,y = data.getXy()
	feature1 = []
	print "CPU data loaded..."

	for i in range(235446):
	    feature1.append(float(X[i][0]))
	print "CPU data converted to numpy array..."

	# plot using pandas and seaborn
	g = sns.distplot(feature1);

	sns.plt.show();
	print "checkAddr passed..." + '\n'

def checkDataSrc_Latency():
	"""
	Customized function for checking the data_src with
	latency feature.
	"""
	data_lat = Data()
	data_src = Data()

	# load testing csv and plot
	data_lat.load('test_latency.csv',ify=False)
	data_src.load('test_data_src.csv',ify=False)
	X_1,y_1 = data_lat.getXy()
	X_2,y_2 = data_src.getXy()
	latency = []
	data_src = []
	print "Latency data loaded..."

	for i in xrange(235446):
	    latency.append(float(X_1[i][0]))
	    data_src.append(float(X_2[i][0]))
	print "Latency data converted to numpy array..."

	for x in xrange(20):
		print "data_src: ", data_src[x], " latency: ", latency[x]

	L_1 = []
	L_2 = []
	L_3 = []
	UNC = []

	for x in xrange(235446):
		if data_src[x] == 1:
			L_1.append(float(latency[x]))
		elif data_src[x] == 2:
			L_2.append(float(latency[x]))
		elif data_src[x] == 3:
			L_3.append(float(latency[x]))
		else:
			UNC.append(float(latency[x]))

	plt.hist(L_1, color='b', label='L_1 Cache', alpha=0.99, bins=range(0,500,1))
	plt.hist(L_2, color='g', label='L_2 Cache', alpha=0.90, bins=range(0,500,1))
	plt.hist(L_3, color='r', label='L_3 Cache', alpha=0.81, bins=range(0,500,1))
	plt.hist(UNC, color='y', label='Uncached', alpha=0.72, bins=range(0,500,1))
	plt.legend()
	axes = plt.gca()
	axes.set_xlim([-1,501])
	plt.show()

	# plot in seaborn
	a = sns.distplot(L_1)
	b = sns.distplot(L_2)
	c = sns.distplot(L_3)
	d = sns.distplot(UNC)
	sns.plt.show()

	print "checkLatency passed..." + '\n'

# checkTime()
# checkLatency()
# checkDataSrc()
# checkAddr()
# checkCPU()
checkDataSrc_Latency()