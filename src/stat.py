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
from kmeans import *

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
	# g = sns.distplot(feature1);
	# plt.subplots_adjust(top=0.9)
	# plt.xticks(range(-4,5), ['', 'Uncached', '', '', '', 'L1', 'L2', 'L3'], rotation=30)
	# plt.yticks(range(0,6), ['', '', '', '', '', ''])
	# sns.plt.title('Cache Frequencies')
	# g.set_xlabel('cache catagories')
	# g.set_ylabel('frequencies')

	# sns.plt.show();
	L_1 = 0
	L_2 = 0
	L_3 = 0
	L_N = 0
	for i in xrange(235446):
		if feature1[i] == 1:
			L_1 += 1
		if feature1[i] == 2:
			L_2 += 1
		if feature1[i] == 3:
			L_3 += 1
		if feature1[i] == -3:
			L_N += 1

	# plot pie chart
	labels = 'L1', 'L2', 'L3', 'Non-cached'
	sizes = [L_1, L_2, L_3, L_N]
	colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue']
	explode = (0.08, 0.04, 0.02, 0)
	plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
	plt.axis('equal')
	plt.title('Cache Access Frequencies Comparison')
	plt.show()
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
	The sharing metric is calculated by weighted average of L1
	cache access.
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
	myDict = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, \
	          8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 0,\
	          16: 0, 17: 0, 18: 0, 19: 0, 20: 0, 21: 0, 22: 0, 23: 0, \
	          24: 0, 25: 0, 26: 0, 27: 0, 28: 0, 29: 0, 30: 0, 31: 0}

	sharmetric = [0.0]*235446
	for i in xrange(235446):
		if ft2[i] == 1:  # if using L1 cache
			value = myDict[ft1[i]]
			sharmetric[i] = value
			myDict[ft1[i]] = value+1

	L1Access = 0.0
	for i in xrange(32):
		L1Access += myDict[i]
	portAccess = [0.0]*32
	for i in xrange(16):
		portAccess[i] = (myDict[i]+myDict[twin_proc(i)]) / L1Access
		portAccess[twin_proc(i)] = portAccess[i]

	for i in xrange(235446):
		if ft2[i] == 1:  # if using L1 cache
			sharmetric[i] = sharmetric[i] * portAccess[ft1[i]]

	# for i in xrange(16):
	# 	if myDict[ft1[i]] >= myDict[ft1[twin_proc(i)]]:
	# 		myDict[ft1[i]] = myDict[ft1[twin_proc(i)]]
	# 	else:
	# 		myDict[ft1[twin_proc(i)]] = myDict[ft1[i]]

	# sharmetric = [0.0]*235446
	# for i in xrange(235446):
	# 	if ft2[i] == 1:
	# 		sharmetric.append(myDict[ft1[i]])
	# 	else:
	# 		sharmetric.append(0)

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

###################################################################
#                       Fuzzy False Sharing
###################################################################

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
	ft2 = toInteger(ft2) # Cache raw
	ft3 = toInteger(ft3) # timestamp
	ft4 = toInteger(ft4) # CPU
	ft2 = map(lambda x: map_data_src(x), ft2) # Cache decoded
	print "Data optimized..."

	my_list = zip(ft1,ft4,ft2,ft3)
	writeCSV('test_fsharing.csv', my_list)
	print "Data written..."

	# build dictionary for the metric
	myDict = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], \
	          8: [], 9: [], 10: [], 11: [], 12: [], 13: [], 14: [], 15: [],\
	          16: [], 17: [], 18: [], 19: [], 20: [], 21: [], 22: [], 23: [], \
	          24: [], 25: [], 26: [], 27: [], 28: [], 29: [], 30: [], 31: []}

	# gather smallest vlaue for addr of each CPU ID

	for i in xrange(235446):
		if ft2[i] == 1:  # L1 cache
			myDict[ft4[i]].append(ft1[i])

	means = [0.0]*32
	for i in xrange(32):
		addrs = myDict[i]
		means[i] = sum(addrs) / float(len(addrs))
		print "mean[", i, "]: ", means[i]

	for i in xrange(int(len(myDict[0]))):
		if myDict[0][i] < means[0]:
			print "round: ", i, " addr: ", myDict[0][i]
	print max(myDict[0])
	print min(myDict[0])

	md_max = max(myDict[0])
	md_min = min(myDict[0])

	# fuzzy false sharing metric
	# k is assigned to be 4 in this case
	# by plotting out addresses, we could
	# see there are roughly 4 clusters
	k = 4
	avrg_addr = [0.0]*32*k # mean for each cluster in 32 CPUs

	for l in xrange(32):
		CPU_d = [] # CPU data containing memory address
		for i in xrange(int(len(myDict[l]))):
			CPU_d.append([myDict[l][i]])

		# sns.distplot(CPU_d, bins=250)
		# plt.hist(CPU_d, 20)

		# kmeans clustering for address
		y = []
		y, centers, labels_unique = useKMeans(CPU_d, y, n_clusters=k)
		# print labels_unique
		# print centers

		# calculate mean addresses for each cluster
		sample_d = [0.0]*k # 2
		sig_d = [0.0]*k
		for i in xrange(k):
			sum_k = 0.0
			count_k = 0
			for j in xrange(int(len(myDict[l]))):
				if y[j] == i:
					sum_k += myDict[l][j]
					count_k += 1
			mean_k = sum_k / count_k
			sample_d[i] = mean_k

		# if sample_d[0] > sample_d[1]:
		# 	avrg_addr[l*2] = sample_d[0]
		# 	avrg_addr[l*2+1] = sample_d[1]
		# else:
		# 	avrg_addr[l*2] = sample_d[1]
		# 	avrg_addr[l*2+1] = sample_d[0]

		# assign mean address to the variable
		# k consecutive same mean addresses for each CPU
		for i in xrange(k):
			avrg_addr[l*k+i] = sample_d[i]

		for i in xrange(k):
			sig_d[i] = (sample_d[0]-sample_d[i]) / sample_d[i] * 100.0

		print "cpu: ", l

		print sample_d
		print sig_d

	print avrg_addr

	# calculate # of elements stored in the dic
	count_d = 0
	for i in xrange(32):
		count_d += int(len(myDict[i]))
	print count_d

	# create list for the metric
	ffsharing = [4.0]*235446
	for i in xrange(235446):
		if ft2[i] == 1:  # L1 cache
		    # calculate raw CPU index from the memory address
			region = closest_addr_region(avrg_addr, ft1[i])
			# convert raw CPU index to CPU index
			mregion = int(region)/k
			start_p = 0
			if i > 10:
				start_p = i-10
			if mregion != ft4[i]:
				fs = False
				for j in xrange(start_p,i):
					if ft4[j] == mregion:
						fs= True
				if mregion == twin_proc(ft4[i]):
					if fs == False:
						ffsharing[i] = 3.0 # same physical core, did not visit in a short time
					else:
						ffsharing[i] = 1.0 # same physical core, visited in 10 rounds
				else:
					if fs == False:
						ffsharing[i] = 2.0 # different physical core, did not visit in a short time
					else:
						ffsharing[i] = 0.0 # different physical core, visited in 10 rounds

	g_1 = sns.distplot(ffsharing)
	sns.plt.title('Fuzzy False Sharing')
	g_1.set_xlabel('sharing catagories (refer to code)')
	g_1.set_ylabel('frequencies')
	sns.plt.show()

	my_list = zip(ffsharing)
	writeCSV('test_ffsharing.csv', my_list)
	print "Data written..."
	print "checkFSharMetric passed..." + '\n'

def checkXYZ():
	"""
	Customized function for checking the data structure.
	"""
	# first read in some features from sample
	data = Data()
	ft1 = extract('samples.csv', 8, start=1)
	ft2 = extract('samples.csv', 9, start=1)
	ft3 = extract('samples.csv', 10, start=1)
	print "Data loaded..."

	# do some optimization
	ft1 = toInteger(ft1)
	ft2 = toInteger(ft2)
	ft3 = toInteger(ft3)
	print "Data optimized..."

	# then write out tesing csv
	my_list = zip(ft1,ft2,ft3)
	my_list_x = zip(ft1)
	my_list_y = zip(ft2)
	my_list_z = zip(ft3)
	writeCSV('test_xyz.csv', my_list)
	writeCSV('test_x.csv', my_list_x)
	writeCSV('test_y.csv', my_list_y)
	writeCSV('test_z.csv', my_list_z)
	print "Data written..."

	# load testing csv and plot
	data.load('test_xyz.csv',ify=False)
	X,y = data.getXy()
	feature1 = []
	feature2 = []
	feature3 = []
	print "CPU data loaded..."

	for i in range(235446):
	    feature1.append(float(X[i][0]))
	    feature2.append(float(X[i][1]))
	    feature3.append(float(X[i][2]))
	print "CPU data converted to numpy array..."

	# plot using pandas and seaborn
	g = sns.distplot(feature1);
	h = sns.distplot(feature2);
	i = sns.distplot(feature3);

	sns.plt.show();
	print "checkAddr passed..." + '\n'

# checkTime()
# checkLatency()
# checkDataSrc()
# checkAddr()
# checkCPU()
# checkDataSrc_Latency()
# checkSharMetric()
# checkThreadMetric()
checkFSharMetric()
# checkXYZ()