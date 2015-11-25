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
	plt.subplots_adjust(top=0.9)
	plt.xticks(range(-4,5), ['', 'Uncached', '', '', '', 'L1', 'L2', 'L3'], rotation=30)
	plt.yticks(range(0,6), ['', '', '', '', '', ''])
	sns.plt.title('Cache Frequencies')
	g.set_xlabel('cache catagories')
	g.set_ylabel('frequencies')

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

	plt.hist(L_1, color='b', label='L_1 Cache', alpha=0.99, bins=range(0,4000,1))
	plt.hist(L_2, color='g', label='L_2 Cache', alpha=0.90, bins=range(0,4000,1))
	plt.hist(L_3, color='r', label='L_3 Cache', alpha=0.81, bins=range(0,4000,1))
	plt.hist(UNC, color='y', label='Uncached', alpha=0.72, bins=range(0,4000,1))
	plt.legend()
	axes = plt.gca()
	axes.set_xlim([-1,4001])

	plt.title('Cache Catagories & Latencies')
	plt.xlabel('latencies (in cycles)')
	plt.ylabel('number of cache hits (misses)')
	plt.show()

	# plot in seaborn
	a = sns.distplot(L_1)
	b = sns.distplot(L_2)
	c = sns.distplot(L_3)
	d = sns.distplot(UNC)
	sns.plt.title('Cache Catagories & Latencies')
	a.set_xlabel('latencies (in cycles)')
	a.set_ylabel('number of cache hits (misses)')
	sns.plt.show()

	print "checkLatency passed..." + '\n'

def checkSharMetric():
	"""
	Customized function for checking the sharing metric.
	"""
	# first read in some features from sample
	data = Data()
	ft1 = extract('samples.csv', 15, start=1)
	ft2 = extract('samples.csv', 17, start=1)
	print "Data loaded..."

	# do some optimization
	ft1 = toInteger(ft1) # CPU
	ft2 = toInteger(ft2) # Cache
	ft2 = map(lambda x: map_data_src(x), ft2)
	print "Data optimized..."

	# build dictionary for the metric
	myDict = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 0,\
	          16: 0, 17: 0, 18: 0, 19: 0, 20: 0, 21: 0, 22: 0, 23: 0, 24: 0, 25: 0, 26: 0, 27: 0, 28: 0, 29: 0, 30: 0, 31: 0}
	for i in xrange(235446):
		if ft2[i] == 1:  # if using L1 cache
			value = myDict[ft1[i]]
			myDict[ft1[i]] = value+1

	for i in xrange(16):
		if myDict[ft1[i]] >= myDict[ft1[twin_proc(i)]]:
			myDict[ft1[i]] = myDict[ft1[twin_proc(i)]]
		else:
			myDict[ft1[twin_proc(i)]] = myDict[ft1[i]]

	sharmetric = []
	for i in xrange(235446):
		if ft2[i] == 1:
			sharmetric.append(myDict[ft1[i]])
		else:
			sharmetric.append(0)

	# then write out tesing csv
	my_list = zip(sharmetric)
	writeCSV('test_sharmetric.csv', my_list)
	print "Data written..."

	# load testing csv and plot
	data.load('test_sharmetric.csv',ify=False)
	X,y = data.getXy()
	feature1 = []
	print "sharing metric data loaded..."

	for i in range(235446):
	    feature1.append(float(X[i][0]))
	print "CPU data converted to numpy array..."

	# plot using pandas and seaborn
	g = sns.distplot(feature1);

	sns.plt.show();
	print "checkAddr passed..." + '\n'


def checkThreadMetric():
	"""
	Customized function for checking the thread variation 
	metric.
	"""
	# first read in some features from sample
	data = Data()
	ft1 = extract('samples.csv', 15, start=1)
	ft2 = extract('samples.csv', 12, start=1)
	print "Data loaded..."

	# do some optimization
	ft1 = toInteger(ft1) # CPU
	ft2 = toInteger(ft2) # Thread ID
	print "Data optimized..."

	# build dictionary for the metric
	myDict = {}
	tidcpumetric = []
	for i in xrange(235446):
		if ft2[i] not in myDict:
			myDict[ft2[i]] = ft1[i]
			tidcpumetric.append(0)
		else:
			if ft1[i] == myDict[ft2[i]]:
				tidcpumetric.append(0)
			else:
				# if for the same thread cpu changes
				myDict[ft2[i]] = ft1[i]
				tidcpumetric.append(1)

	# then write out tesing csv
	my_list = zip(tidcpumetric)
	writeCSV('test_tidupumetric.csv', my_list)
	print "Data written..."

	# load testing csv and plot
	data.load('test_tidupumetric.csv',ify=False)
	X,y = data.getXy()
	feature1 = []
	print "sharing metric data loaded..."

	for i in range(235446):
	    feature1.append(float(X[i][0]))
	print "CPU data converted to numpy array..."

	# plot using pandas and seaborn
	g = sns.distplot(feature1);
	sns.plt.title('Threads jump cross CPUs')
	g.set_xlabel('Threads Changes Metric (if change)')
	g.set_ylabel('number of data access')
	plt.subplots_adjust(top=0.9)
	plt.xticks(range(0,2), ['Not Change', 'Change', ''])

	sns.plt.show();
	print "checkAddr passed..." + '\n'


def checkFSharMetric():
	"""
	Customized function for checking the false sharing metric.
	"""
	# first read in some features from sample
	data = Data()
	ft1 = extract('samples.csv', 14, start=1)
	ft2 = extract('samples.csv', 17, start=1)
	ft3 = extract('samples.csv', 13, start=1)
	ft4 = extract('samples.csv', 15, start=1)
	print "type ft1: ", type(ft1[1])
	print "Data loaded..."

	# do some optimization
	ft1 = toLong(toFloat(ft1)) # data address
	ft2 = toInteger(ft2) # Cache
	ft3 = toInteger(ft3) # timestamp
	ft4 = toInteger(ft4) # CPU
	ft2 = map(lambda x: map_data_src(x), ft2)
	print "Data optimized..."

	# build dictionary for the metric
	myDict = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], \
	          8: [], 9: [], 10: [], 11: [], 12: [], 13: [], 14: [], 15: [],\
	          16: [], 17: [], 18: [], 19: [], 20: [], 21: [], 22: [], 23: [], \
	          24: [], 25: [], 26: [], 27: [], 28: [], 29: [], 30: [], 31: []}

	# gather smallest vlaue for addr of each CPU ID
	for i in xrange(235446):
		myDict[ft4[i]].append(ft1[i])

	means = [0.0]*32
	for i in xrange(32):
		addrs = myDict[i]
		means[i] = sum(addrs) / float(len(addrs))

	for i in xrange(32):
		for j in xrange(50):
			print "cpu: ", i, " addr: ", myDict[i][j]
			
		
		

	# for i in xrange(235446):
	# 	if ft1[i]<100:
	# 		print ft1[i]

	# for i in xrange(235446):
	# 	if ft2[i] == 1:  # if using L1 cache
	# 		value = myDict[ft1[i]]
	# 		myDict[ft1[i]] = value+1

	# for i in xrange(16):
	# 	if myDict[ft1[i]] >= myDict[ft1[twin_proc(i)]]:
	# 		myDict[ft1[i]] = myDict[ft1[twin_proc(i)]]
	# 	else:
	# 		myDict[ft1[twin_proc(i)]] = myDict[ft1[i]]

	# sharmetric = []
	# for i in xrange(235446):
	# 	if ft2[i] == 1:
	# 		sharmetric.append(myDict[ft1[i]])
	# 	else:
	# 		sharmetric.append(0)

	# # then write out tesing csv
	# my_list = zip(sharmetric)
	# writeCSV('test_sharmetric.csv', my_list)
	# print "Data written..."

	# # load testing csv and plot
	# data.load('test_sharmetric.csv',ify=False)
	# X,y = data.getXy()
	# feature1 = []
	# print "sharing metric data loaded..."

	# for i in range(235446):
	#     feature1.append(float(X[i][0]))
	# print "CPU data converted to numpy array..."

	# # plot using pandas and seaborn
	# g = sns.distplot(feature1);

	# sns.plt.show();
	print "checkFSharMetric passed..." + '\n'

# checkTime()
# checkLatency()
# checkDataSrc()
# checkAddr()
# checkCPU()
# checkDataSrc_Latency()
# checkSharMetric()
# checkThreadMetric()
checkFSharMetric()