import os
import csv
import math
import numpy as np; np.random.seed(0)
import seaborn as sns; sns.set(color_codes=True)
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats
import statsmodels.api as sm
from sklearn.neighbors import KernelDensity

from mpl_toolkits.mplot3d import Axes3D

from processdata import *
from kmeans import *
from optimize import *
from kmeans import *

###################################################################
#                    Fuzzy False Sharing V1
###################################################################

def checkFSharMetric_1():
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

	# build dictionary for the metric (corresponding to 32 CPUs)
	myDict = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], \
	          8: [], 9: [], 10: [], 11: [], 12: [], 13: [], 14: [], 15: [],\
	          16: [], 17: [], 18: [], 19: [], 20: [], 21: [], 22: [], 23: [], \
	          24: [], 25: [], 26: [], 27: [], 28: [], 29: [], 30: [], 31: []}

	# gather value for addr of each CPU ID
	for i in xrange(235446):
		if ft2[i] == 1:  # L1 cache
			myDict[ft4[i]].append(ft1[i])

	# calculate mean address for each CPU
	means = [0.0]*32
	for i in xrange(32):
		addrs = myDict[i]
		means[i] = sum(addrs) / float(len(addrs))
		print "mean[", i, "]: ", means[i]

	# for test max and min addr in the first CPU
	print max(myDict[0])
	print min(myDict[0])

	# fuzzy false sharing metric, k is assigned to be 4 in this case by
	# plotting out addresses, we could see there are roughly 4 clusters
	k = 4
	avrg_addr = [0.0]*32*k # mean for each cluster in 32 CPUs, k copies
	std_addr = [0.0]*32*k # standard deviation for addresses of each CPU
	calc_distr = [0.0]*32 # calculated gaussian distribution probability

	for l in xrange(32):
		CPU_d = [] # copying addr data for each CPU
		for i in xrange(int(len(myDict[l]))):
			CPU_d.append([myDict[l][i]])

		# kmeans clustering for address
		y = [] # array with cluster assignments
		y, centers, labels_unique = useKMeans(CPU_d, y, n_clusters=k)

		# calculate mean addresses for each cluster
		sample_d = [0.0]*k # mean addr for the cluster
		sig_d = [0.0]*k # significance metric for each cluster
		for i in xrange(k):
			sum_k = 0.0
			count_k = 0
			for j in xrange(int(len(myDict[l]))):
				if y[j] == i:
					sum_k += myDict[l][j]
					count_k += 1
			mean_k = sum_k / count_k
			sample_d[i] = mean_k

		# assign mean address of each cluster to the variable
		for i in xrange(k):
			avrg_addr[l*k+i] = sample_d[i]

		# assign significance metric each cluster to the variable
		for i in xrange(k):
			sig_d[i] = (sample_d[0]-sample_d[i]) / sample_d[i] * 100.0

		# calculate standard deviation for each cluster of a CPU
		for i in xrange(k):
			dev = 0.0
			stddev = 0.0
			for j in xrange(int(len(myDict[l]))):
				if y[j] == i:
					dev += (myDict[l][j] - avrg_addr[l*k+i])**2
			stddev = dev**0.5
			std_addr[l*k+i] = stddev

		print "cpu: ", l
		print sample_d

	print " "
	print avrg_addr
	print " "
	print "standard deviation: "
	print std_addr

	# calculate number of elements stored in the dic
	count_d = 0
	for i in xrange(32):
		count_d += int(len(myDict[i]))
	print count_d

	# create list for the metric
	ffsharing = [1.0]*235446
	counter  = 0.0

	for i in xrange(235446):
		if ft2[i] == 1:  # L1 cache
		    # calculate raw CPU index by smallest distance
			region = closest_addr_region(avrg_addr, ft1[i])

			# convert raw CPU index to CPU cluster index
			mregion = int(region)/k # calculated cluster index
			ffs_metric = 0.0

			if i < 5:
				for x in xrange(i+5): # within 10 rounds (short period of time)
					temp_region = closest_addr_region(avrg_addr, ft1[x])
					temp_mregion = int(temp_region)/k
					if temp_mregion == mregion:
						if ft4[x] == twin_proc(ft4[i]): # same physical core; not same CPU id
							ffs_metric += 8
						elif ft4[x] != ft4[i]: # different physical core
							ffs_metric += 16
						else:
							ffs_metric += 2
			elif i >= 5 and i < 235441:
				for x in xrange(i-5, i+5):
					temp_region = closest_addr_region(avrg_addr, ft1[x])
					temp_mregion = int(temp_region)/k
					if temp_mregion == mregion:
						if ft4[x] == twin_proc(ft4[i]):
							ffs_metric += 8
						elif ft4[x] != ft4[i]:
							ffs_metric += 16
						else:
							ffs_metric += 2
			else:
				for x in xrange(i-5, 235446):
					temp_region = closest_addr_region(avrg_addr, ft1[x])
					temp_mregion = int(temp_region)/k
					if temp_mregion == mregion:
						if ft4[x] == twin_proc(ft4[i]):
							ffs_metric += 8
						elif ft4[x] != ft4[i]:
							ffs_metric += 16
						else:
							ffs_metric += 2

			ffsharing[i] = ffs_metric

	for x in xrange(1,100):
		print ffsharing[x]

	my_list = zip(ffsharing)
	writeCSV('test_ffsharing.csv', my_list)
	print "Data written..."
	print "checkFSharMetric passed..." + '\n'


###################################################################
#                    Fuzzy False Sharing V2
###################################################################

def checkFSharMetric_2():
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

	# build dictionary for the metric (corresponding to 32 CPUs)
	myDict = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], \
	          8: [], 9: [], 10: [], 11: [], 12: [], 13: [], 14: [], 15: [],\
	          16: [], 17: [], 18: [], 19: [], 20: [], 21: [], 22: [], 23: [], \
	          24: [], 25: [], 26: [], 27: [], 28: [], 29: [], 30: [], 31: []}

	# gather value for addr of each CPU ID
	for i in xrange(235446):
		if ft2[i] == 1:  # L1 cache
			myDict[ft4[i]].append(ft1[i])

	# calculate mean address for each CPU
	means = [0.0]*32
	for i in xrange(32):
		addrs = myDict[i]
		means[i] = sum(addrs) / float(len(addrs))
		print "mean[", i, "]: ", means[i]

	# for test max and min addr in the first CPU
	print max(myDict[0])
	print min(myDict[0])

	# fuzzy false sharing metric, k is assigned to be 4 in this case by
	# plotting out addresses, we could see there are roughly 4 clusters
	k = 4
	avrg_addr = [0.0]*32*k # mean for each cluster in 32 CPUs, k copies
	std_addr = [0.0]*32*k # standard deviation for addresses of each CPU
	calc_distr = [0.0]*32 # calculated gaussian distribution probability

	for l in xrange(32):
		CPU_d = [] # copying addr data for each CPU
		for i in xrange(int(len(myDict[l]))):
			CPU_d.append([myDict[l][i]])

		# kmeans clustering for address
		y = [] # array with cluster assignments
		y, centers, labels_unique = useKMeans(CPU_d, y, n_clusters=k)

		# calculate mean addresses for each cluster
		sample_d = [0.0]*k # mean addr for the cluster
		sig_d = [0.0]*k # significance metric for each cluster
		for i in xrange(k):
			sum_k = 0.0
			count_k = 0
			for j in xrange(int(len(myDict[l]))):
				if y[j] == i:
					sum_k += myDict[l][j]
					count_k += 1
			mean_k = sum_k / count_k
			sample_d[i] = mean_k

		# assign mean address of each cluster to the variable
		for i in xrange(k):
			avrg_addr[l*k+i] = sample_d[i]

		# assign significance metric each cluster to the variable
		for i in xrange(k):
			sig_d[i] = (sample_d[0]-sample_d[i]) / sample_d[i] * 100.0

		# calculate standard deviation for each cluster of a CPU
		for i in xrange(k):
			dev = 0.0
			stddev = 0.0
			for j in xrange(int(len(myDict[l]))):
				if y[j] == i:
					dev += (myDict[l][j] - avrg_addr[l*k+i])**2
			stddev = dev**0.5
			std_addr[l*k+i] = stddev

		print "cpu: ", l
		print sample_d

	print " "
	print avrg_addr
	print " "
	print "standard deviation: "
	print std_addr

	# calculate number of elements stored in the dic
	count_d = 0
	for i in xrange(32):
		count_d += int(len(myDict[i]))
	print count_d

	# create list for the metric
	ffsharing = [0.0]*235446
	counter  = 0.0

	for i in xrange(235446):
		if ft2[i] == 1:  # L1 cache
		    # calculate raw CPU index by smallest distance
			region = closest_addr_region(avrg_addr, ft1[i])

			CPU_id = ft4[i]
			act_reg = [] # means for clusters of address
			for j in xrange(k):
				act_reg.append(avrg_addr[CPU_id*k+j])
			act_reg_raw = closest_addr_region(act_reg, ft1[i]) # actual raw value for mean
			act_mean = act_reg[act_reg_raw] # actual mean value for the correponding cluster
			act_index = CPU_id*k + act_reg_raw # actual index for stddev value
			act_stddev = std_addr[act_index] # actual corresponding stddev

			# convert raw CPU index to CPU cluster index
			mregion = int(region)/k # calculated cluster index
			ffs_metric = 0.0

			if i < 5:
				for x in xrange(i+5): # within 10 rounds (short period of time)
					ffs_temp = scipy.stats.norm(act_mean, act_stddev).pdf(ft1[x])
					ffs_metric += ffs_temp
			elif i >= 5 and i < 235441:
				for x in xrange(i-5, i+5):
					ffs_temp = scipy.stats.norm(act_mean, act_stddev).pdf(ft1[x])
					ffs_metric += ffs_temp
			else:
				for x in xrange(i-5, 235446):
					ffs_temp = scipy.stats.norm(act_mean, act_stddev).pdf(ft1[x])
					ffs_metric += ffs_temp

			ffsharing[i] = ffs_metric

	for x in xrange(1,100):
		print ffsharing[x]

	my_list = zip(ffsharing)
	writeCSV('test_ffsharing.csv', my_list)
	print "Data written..."
	print "checkFSharMetric passed..." + '\n'


###################################################################
#                    Fuzzy False Sharing V3
###################################################################

def checkFSharMetric_3():
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

	# build dictionary for the metric (corresponding to 32 CPUs)
	myDict = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], \
	          8: [], 9: [], 10: [], 11: [], 12: [], 13: [], 14: [], 15: [],\
	          16: [], 17: [], 18: [], 19: [], 20: [], 21: [], 22: [], 23: [], \
	          24: [], 25: [], 26: [], 27: [], 28: [], 29: [], 30: [], 31: []}

	# gather value for addr of each CPU ID
	for i in xrange(235446):
		if ft2[i] == 1:  # L1 cache
			myDict[ft4[i]].append(ft1[i])

	# calculate mean address for each CPU
	means = [0.0]*32
	for i in xrange(32):
		addrs = myDict[i]
		means[i] = sum(addrs) / float(len(addrs))
		print "mean[", i, "]: ", means[i]

	# for test max and min addr in the first CPU
	print max(myDict[0])
	print min(myDict[0])

	# fuzzy false sharing metric, k is assigned to be 4 in this case by
	# plotting out addresses, we could see there are roughly 4 clusters
	k = 4
	avrg_addr = [0.0]*32*k # mean for each cluster in 32 CPUs, k copies
	std_addr = [0.0]*32*k # standard deviation for addresses of each CPU
	calc_distr = [0.0]*32 # calculated gaussian distribution probability

	for l in xrange(32):
		CPU_d = [] # copying addr data for each CPU
		for i in xrange(int(len(myDict[l]))):
			CPU_d.append([myDict[l][i]])

		# kmeans clustering for address
		y = [] # array with cluster assignments
		y, centers, labels_unique = useKMeans(CPU_d, y, n_clusters=k)

		# calculate mean addresses for each cluster
		sample_d = [0.0]*k # mean addr for the cluster
		sig_d = [0.0]*k # significance metric for each cluster
		for i in xrange(k):
			sum_k = 0.0
			count_k = 0
			for j in xrange(int(len(myDict[l]))):
				if y[j] == i:
					sum_k += myDict[l][j]
					count_k += 1
			mean_k = sum_k / count_k
			sample_d[i] = mean_k

		# assign mean address of each cluster to the variable
		for i in xrange(k):
			avrg_addr[l*k+i] = sample_d[i]

		# assign significance metric each cluster to the variable
		for i in xrange(k):
			sig_d[i] = (sample_d[0]-sample_d[i]) / sample_d[i] * 100.0

		# calculate standard deviation for each cluster of a CPU
		for i in xrange(k):
			dev = 0.0
			stddev = 0.0
			for j in xrange(int(len(myDict[l]))):
				if y[j] == i:
					dev += (myDict[l][j] - avrg_addr[l*k+i])**2
			stddev = dev**0.5
			std_addr[l*k+i] = stddev

		print "cpu: ", l
		print sample_d

	print " "
	print avrg_addr
	print " "
	print "standard deviation: "
	print std_addr

	# calculate number of elements stored in the dic
	count_d = 0
	for i in xrange(32):
		count_d += int(len(myDict[i]))
	print count_d

	# create list for the metric
	ffsharing = [4.0]*235446
	counter  = 0.0

	for i in xrange(235446):
		if ft2[i] == 1:  # L1 cache
		    # calculate raw CPU index by smallest distance
			region = closest_addr_region(avrg_addr, ft1[i])

			# convert raw CPU index to CPU cluster index
			mregion = int(region)/k # calculated cluster index (1:32)

			if i < 5:
				for x in xrange(i+5): # within 10 rounds (short period of time)
					if mregion != ft4[i]:
						fs = False
						for j in xrange(i+5):
							if ft4[j] == mregion:
								fs = True
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
			elif i >= 5 and i < 235441:
				for x in xrange(i-5, i+5):
					if mregion != ft4[i]:
						fs = False
						for j in xrange(i-5, i+5):
							if ft4[j] == mregion:
								fs = True
						if mregion == twin_proc(ft4[i]):
							if fs == False:
								ffsharing[i] = 3.0
							else:
								ffsharing[i] = 1.0
						else:
							if fs == False:
								ffsharing[i] = 2.0
							else:
								ffsharing[i] = 0.0
			else:
				for x in xrange(i-5, 235446):
					if mregion != ft4[i]:
						fs = False
						for j in xrange(i-5, 235446):
							if ft4[j] == mregion:
								fs = True
						if mregion == twin_proc(ft4[i]):
							if fs == False:
								ffsharing[i] = 3.0
							else:
								ffsharing[i] = 1.0
						else:
							if fs == False:
								ffsharing[i] = 2.0
							else:
								ffsharing[i] = 0.0

	for x in xrange(1,100):
		print ffsharing[x]

	my_list = zip(ffsharing)
	writeCSV('test_ffsharing.csv', my_list)
	print "Data written..."
	print "checkFSharMetric passed..." + '\n'


###################################################################
#                    Fuzzy False Sharing V4
###################################################################

def checkFSharMetric_4():
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

	# build dictionary for the metric (corresponding to 32 CPUs)
	myDict = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], \
	          8: [], 9: [], 10: [], 11: [], 12: [], 13: [], 14: [], 15: [],\
	          16: [], 17: [], 18: [], 19: [], 20: [], 21: [], 22: [], 23: [], \
	          24: [], 25: [], 26: [], 27: [], 28: [], 29: [], 30: [], 31: []}

	# gather value for addr of each CPU ID
	for i in xrange(235446):
		if ft2[i] == 1:  # L1 cache
			myDict[ft4[i]].append(ft1[i])

	# calculate mean address for each CPU
	means = [0.0]*32
	for i in xrange(32):
		addrs = myDict[i]
		means[i] = sum(addrs) / float(len(addrs))
		print "mean[", i, "]: ", means[i]

	# for test max and min addr in the first CPU
	print max(myDict[0])
	print min(myDict[0])

	# fuzzy false sharing metric, k is assigned to be 4 in this case by
	# plotting out addresses, we could see there are roughly 4 clusters
	k = 4
	avrg_addr = [0.0]*32*k # mean for each cluster in 32 CPUs, k copies
	std_addr = [0.0]*32*k # standard deviation for addresses of each CPU
	calc_distr = [0.0]*32 # calculated gaussian distribution probability

	for l in xrange(32):
		CPU_d = [] # copying addr data for each CPU
		for i in xrange(int(len(myDict[l]))):
			CPU_d.append([myDict[l][i]])

		# kmeans clustering for address
		y = [] # array with cluster assignments
		y, centers, labels_unique = useKMeans(CPU_d, y, n_clusters=k)

		# calculate mean addresses for each cluster
		sample_d = [0.0]*k # mean addr for the cluster
		sig_d = [0.0]*k # significance metric for each cluster
		for i in xrange(k):
			sum_k = 0.0
			count_k = 0
			for j in xrange(int(len(myDict[l]))):
				if y[j] == i:
					sum_k += myDict[l][j]
					count_k += 1
			mean_k = sum_k / count_k
			sample_d[i] = mean_k

		# assign mean address of each cluster to the variable
		for i in xrange(k):
			avrg_addr[l*k+i] = sample_d[i]

		# assign significance metric each cluster to the variable
		for i in xrange(k):
			sig_d[i] = (sample_d[0]-sample_d[i]) / sample_d[i] * 100.0

		# calculate standard deviation for each cluster of a CPU
		for i in xrange(k):
			dev = 0.0
			stddev = 0.0
			for j in xrange(int(len(myDict[l]))):
				if y[j] == i:
					dev += (myDict[l][j] - avrg_addr[l*k+i])**2
			stddev = dev**0.5
			std_addr[l*k+i] = stddev

		print "cpu: ", l
		print sample_d

	print " "
	print avrg_addr
	print " "
	print "standard deviation: "
	print std_addr

	# calculate number of elements stored in the dic
	count_d = 0
	for i in xrange(32):
		count_d += int(len(myDict[i]))
	print count_d

	# create list for the metric
	ffsharing = [0.0]*235446
	counter  = 0.0

	for i in xrange(235446):
		if ft2[i] == 1:  # L1 cache
		    # calculate raw CPU index by smallest distance
			region = closest_addr_region(avrg_addr, ft1[i])

			# convert raw CPU index to CPU cluster index
			mregion = int(region)/k # calculated cluster index (1:32)

			if i < 5:
				for x in xrange(i+5): # within 10 rounds (short period of time)
					if mregion != ft4[i]:
						fs = 0.0 # number of possible false sharing
						fs_t = 0.0 # number of possible false sharing on the same physical core
						for j in xrange(i+5):
							if ft4[j] == mregion and j != i:
								fs += 1
								if ft4[j] == twin_proc(ft4[i]):
									fs_t += 1
						if mregion == twin_proc(ft4[i]):
							if fs == 0.0:
								ffsharing[i] = 1.0 # same physical core, did not visit in a short time
							else:
								ffsharing[i] = 3.0 * fs_t # same physical core, visited in 10 rounds
						else:
							if fs == 0.0:
								ffsharing[i] = 2.0 # different physical core, did not visit in a short time
							else:
								ffsharing[i] = 4.0 * (fs - fs_t) # different physical core, visited in 10 rounds
			elif i >= 5 and i < 235441:
				for x in xrange(i-5, i+5):
					if mregion != ft4[i]:
						fs = 0.0
						fs_t = 0.0
						for j in xrange(i-5, i+5):
							if ft4[j] == mregion and j != i:
								fs += 1
								if ft4[j] == twin_proc(ft4[i]):
									fs_t += 1
						if mregion == twin_proc(ft4[i]):
							if fs == 0.0:
								ffsharing[i] = 1.0
							else:
								ffsharing[i] = 3.0 * fs_t
						else:
							if fs == 0.0:
								ffsharing[i] = 2.0
							else:
								ffsharing[i] = 4.0 * (fs - fs_t)
			else:
				for x in xrange(i-5, 235446):
					if mregion != ft4[i]:
						fs = 0.0
						fs_t = 0.0
						for j in xrange(i-5, 235446):
							if ft4[j] == mregion and j != i:
								fs += 1
								if ft4[j] == twin_proc(ft4[i]):
									fs_t += 1
						if mregion == twin_proc(ft4[i]):
							if fs == 0.0:
								ffsharing[i] = 1.0
							else:
								ffsharing[i] = 3.0 * fs_t
						else:
							if fs == 0.0:
								ffsharing[i] = 2.0
							else:
								ffsharing[i] = 4.0 * (fs - fs_t)

	for x in xrange(1,100):
		print ffsharing[x]

	my_list = zip(ffsharing)
	writeCSV('test_ffsharing.csv', my_list)
	print "Data written..."
	print "checkFSharMetric passed..." + '\n'


###################################################################
#                    Fuzzy False Sharing V5
###################################################################

def checkFSharMetric_5():
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

	# build dictionary for the metric (corresponding to 32 CPUs)
	myDict = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], \
	          8: [], 9: [], 10: [], 11: [], 12: [], 13: [], 14: [], 15: [],\
	          16: [], 17: [], 18: [], 19: [], 20: [], 21: [], 22: [], 23: [], \
	          24: [], 25: [], 26: [], 27: [], 28: [], 29: [], 30: [], 31: []}

	# gather value for addr of each CPU ID
	for i in xrange(235446):
		if ft2[i] == 1:  # L1 cache
			myDict[ft4[i]].append(ft1[i])

	# calculate mean address for each CPU
	means = [0.0]*32
	for i in xrange(32):
		addrs = myDict[i]
		means[i] = sum(addrs) / float(len(addrs))
		print "mean[", i, "]: ", means[i]

	# for test max and min addr in the first CPU
	print max(myDict[0])
	print min(myDict[0])

	# fuzzy false sharing metric, k is assigned to be 4 in this case by
	# plotting out addresses, we could see there are roughly 4 clusters
	k = 4
	avrg_addr = [0.0]*32*k # mean for each cluster in 32 CPUs, k copies
	std_addr = [0.0]*32*k # standard deviation for addresses of each CPU
	calc_distr = [0.0]*32 # calculated gaussian distribution probability

	for l in xrange(32):
		CPU_d = [] # copying addr data for each CPU
		for i in xrange(int(len(myDict[l]))):
			CPU_d.append([myDict[l][i]])

		# kmeans clustering for address
		y = [] # array with cluster assignments
		y, centers, labels_unique = useKMeans(CPU_d, y, n_clusters=k)

		# calculate mean addresses for each cluster
		sample_d = [0.0]*k # mean addr for the cluster
		sig_d = [0.0]*k # significance metric for each cluster
		for i in xrange(k):
			sum_k = 0.0
			count_k = 0
			for j in xrange(int(len(myDict[l]))):
				if y[j] == i:
					sum_k += myDict[l][j]
					count_k += 1
			mean_k = sum_k / count_k
			sample_d[i] = mean_k

		# assign mean address of each cluster to the variable
		for i in xrange(k):
			avrg_addr[l*k+i] = sample_d[i]

		# assign significance metric each cluster to the variable
		for i in xrange(k):
			sig_d[i] = (sample_d[0]-sample_d[i]) / sample_d[i] * 100.0

		# calculate standard deviation for each cluster of a CPU
		for i in xrange(k):
			dev = 0.0
			stddev = 0.0
			for j in xrange(int(len(myDict[l]))):
				if y[j] == i:
					dev += (myDict[l][j] - avrg_addr[l*k+i])**2
			stddev = dev**0.5
			std_addr[l*k+i] = stddev

		print "cpu: ", l
		print sample_d

	print " "
	print avrg_addr
	print " "
	print "standard deviation: "
	print std_addr

	# calculate number of elements stored in the dic
	count_d = 0
	for i in xrange(32):
		count_d += int(len(myDict[i]))
	print count_d

	# create list for the metric
	ffsharing = [4.0]*235446
	counter  = 0.0

	for i in xrange(235446):
		if ft2[i] == 1:  # L1 cache
		    # calculate raw CPU index by smallest distance
			region = closest_addr_region(avrg_addr, ft1[i])

			# convert raw CPU index to CPU cluster index
			mregion = int(region)/k # calculated cluster index (1:32)

			if i < 5:
				for x in xrange(i+5): # within 10 rounds (short period of time)
					if mregion != ft4[i]:
						fs = False
						for j in xrange(i+5):
							temp_reg = closest_addr_region(avrg_addr, ft1[j])
							if temp_reg == region:
								fs = True
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
			elif i >= 5 and i < 235441:
				for x in xrange(i-5, i+5):
					if mregion != ft4[i]:
						fs = False
						for j in xrange(i-5, i+5):
							temp_reg = closest_addr_region(avrg_addr, ft1[j])
							if temp_reg == region:
								fs = True
						if mregion == twin_proc(ft4[i]):
							if fs == False:
								ffsharing[i] = 3.0
							else:
								ffsharing[i] = 1.0
						else:
							if fs == False:
								ffsharing[i] = 2.0
							else:
								ffsharing[i] = 0.0
			else:
				for x in xrange(i-5, 235446):
					if mregion != ft4[i]:
						fs = False
						for j in xrange(i-5, 235446):
							temp_reg = closest_addr_region(avrg_addr, ft1[j])
							if temp_reg == region:
								fs = True
						if mregion == twin_proc(ft4[i]):
							if fs == False:
								ffsharing[i] = 3.0
							else:
								ffsharing[i] = 1.0
						else:
							if fs == False:
								ffsharing[i] = 2.0
							else:
								ffsharing[i] = 0.0

	for x in xrange(1,100):
		print ffsharing[x]

	my_list = zip(ffsharing)
	writeCSV('test_ffsharing.csv', my_list)
	print "Data written..."
	print "checkFSharMetric passed..." + '\n'


###################################################################
#                    Fuzzy False Sharing V6
###################################################################

def checkFSharMetric_6():
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

	# build dictionary for the metric (corresponding to 32 CPUs)
	myDict = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], \
	          8: [], 9: [], 10: [], 11: [], 12: [], 13: [], 14: [], 15: [],\
	          16: [], 17: [], 18: [], 19: [], 20: [], 21: [], 22: [], 23: [], \
	          24: [], 25: [], 26: [], 27: [], 28: [], 29: [], 30: [], 31: []}

	# gather value for addr of each CPU ID
	for i in xrange(235446):
		if ft2[i] == 1:  # L1 cache
			myDict[ft4[i]].append(ft1[i])

	# calculate mean address for each CPU
	means = [0.0]*32
	for i in xrange(32):
		addrs = myDict[i]
		means[i] = sum(addrs) / float(len(addrs))
		print "mean[", i, "]: ", means[i]

	# for test max and min addr in the first CPU
	print max(myDict[0])
	print min(myDict[0])

	# fuzzy false sharing metric, k is assigned to be 4 in this case by
	# plotting out addresses, we could see there are roughly 4 clusters
	k = 4
	avrg_addr = [0.0]*32*k # mean for each cluster in 32 CPUs, k copies
	std_addr = [0.0]*32*k # standard deviation for addresses of each CPU
	calc_distr = [0.0]*32 # calculated gaussian distribution probability

	for l in xrange(32):
		CPU_d = [] # copying addr data for each CPU
		for i in xrange(int(len(myDict[l]))):
			CPU_d.append([myDict[l][i]])

		# kmeans clustering for address
		y = [] # array with cluster assignments
		y, centers, labels_unique = useKMeans(CPU_d, y, n_clusters=k)

		# calculate mean addresses for each cluster
		sample_d = [0.0]*k # mean addr for the cluster
		sig_d = [0.0]*k # significance metric for each cluster
		for i in xrange(k):
			sum_k = 0.0
			count_k = 0
			for j in xrange(int(len(myDict[l]))):
				if y[j] == i:
					sum_k += myDict[l][j]
					count_k += 1
			mean_k = sum_k / count_k
			sample_d[i] = mean_k

		# assign mean address of each cluster to the variable
		for i in xrange(k):
			avrg_addr[l*k+i] = sample_d[i]

		# assign significance metric each cluster to the variable
		for i in xrange(k):
			sig_d[i] = (sample_d[0]-sample_d[i]) / sample_d[i] * 100.0

		# calculate standard deviation for each cluster of a CPU
		for i in xrange(k):
			dev = 0.0
			stddev = 0.0
			for j in xrange(int(len(myDict[l]))):
				if y[j] == i:
					dev += (myDict[l][j] - avrg_addr[l*k+i])**2
			stddev = dev**0.5
			std_addr[l*k+i] = stddev

		print "cpu: ", l
		print sample_d

	print " "
	print avrg_addr
	print " "
	print "standard deviation: "
	print std_addr

	# calculate number of elements stored in the dic
	count_d = 0
	for i in xrange(32):
		count_d += int(len(myDict[i]))
	print count_d

	# create list for the metric
	ffsharing = [5.0]*235446
	counter  = 0.0

	for i in xrange(235446):
		if ft2[i] == 1:  # L1 cache
		    # calculate raw CPU index by smallest distance
			region = closest_addr_region(avrg_addr, ft1[i])

			# convert raw CPU index to CPU cluster index
			mregion = int(region)/k # calculated cluster index (1:32)
			region_check = True if mregion == ft4[i] else False # if addr is close enough to the corresponding cluster mean

			if i < 5:
				fs = False
				fst = False
				for j in xrange(i+5):
					if ft4[j] == mregion:
						fs = True
					if ft4[j] == twin_proc(mregion):
						fst = True
				if region_check == True: # addr near the corresponding cluster mean
					ffsharing[i] = 3.0
				else: # addr near the other cluster mean
					if fs == False:
						if fst == True:
							ffsharing[i] = 2.0
						else:
							ffsharing[i] = 4.0
					else:
						ffsharing[i] = 1.0
			elif i >= 5 and i < 235441:
				fs = False
				for j in xrange(i-5, i+5):
					if ft4[j] == mregion:
						fs = True
				if region_check == True: # addr near the corresponding cluster mean
					ffsharing[i] = 3.0
				else: # addr near the other cluster mean
					if fs == False:
						if fst == True:
							ffsharing[i] = 2.0
						else:
							ffsharing[i] = 4.0
					else:
						ffsharing[i] = 1.0
			else:
				fs = False
				for j in xrange(i-5, 235446):
					if ft4[j] == mregion:
						fs = True
				if region_check == True: # addr near the corresponding cluster mean
					ffsharing[i] = 3.0
				else: # addr near the other cluster mean
					if fs == False:
						if fst == True:
							ffsharing[i] = 2.0
						else:
							ffsharing[i] = 4.0
					else:
						ffsharing[i] = 1.0

	for x in xrange(1,100):
		print ffsharing[x]

	my_list = zip(ffsharing)
	writeCSV('test_ffsharing.csv', my_list)
	print "Data written..."
	print "checkFSharMetric passed..." + '\n'


###################################################################
#                    Fuzzy False Sharing V7
###################################################################

def checkFSharMetric_7():
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

	# build dictionary for the metric (corresponding to 32 CPUs)
	myDict = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], \
	          8: [], 9: [], 10: [], 11: [], 12: [], 13: [], 14: [], 15: [],\
	          16: [], 17: [], 18: [], 19: [], 20: [], 21: [], 22: [], 23: [], \
	          24: [], 25: [], 26: [], 27: [], 28: [], 29: [], 30: [], 31: []}

	# gather value for addr of each CPU ID
	for i in xrange(235446):
		if ft2[i] == 1:  # L1 cache
			myDict[ft4[i]].append(ft1[i])

	# build kde list for estimating distribution of address
	kdeList = []
	kdeScores = [0]*235446

	for l in xrange(32):
		CPU_d = [] # copying addr data for each CPU
		for i in xrange(int(len(myDict[l]))):
			CPU_d.append([myDict[l][i]])

		# build kde for current cpu
		kde = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(CPU_d)
		kdeList.append(kde)


	# create list for the metric
	ffsharing = [0.0]*235446
	counter  = 0.0

	print "Begin to calculate probability for all data points..."
	print ""
	for i in xrange(235446):
		if ft2[i] == 1:  # L1 cache
			ffs_metric = 0.0

			kde = kdeList[ft4[i]]
			std_score = math.exp(kde.score_samples([[ft1[i]]])[0])

			if i < 3:
				for x in xrange(0,i-1): # within 10 rounds (short period of time)
					ffs_temp = math.exp(kde.score_samples([[ft1[x]]])[0])
					std_temp = math.exp((kdeList[ft4[x]]).score_samples([[ft1[i]]])[0])
					penalty = 1.0 - abs(ffs_temp - std_temp)
					print ft4[i], " - ", ft4[x], "  ", ft1[i], " ", ft1[x]
					ffs_temp = abs(ffs_temp - std_score) + penalty
					ffs_metric += ffs_temp
				for x in xrange(i+1,i+4): # within 10 rounds (short period of time)
					ffs_temp = math.exp(kde.score_samples([[ft1[x]]])[0])
					std_temp = math.exp((kdeList[ft4[x]]).score_samples([[ft1[i]]])[0])
					penalty = 1.0 - abs(ffs_temp - std_temp)
					print ft4[i], " - ", ft4[x], "  ", ft1[i], " ", ft1[x]
					ffs_temp = abs(ffs_temp - std_score) + penalty
					ffs_metric += ffs_temp
				ffs_metric = ffs_metric / (i+3)
				print ""
			elif i >= 3 and i < 235443:
				for x in xrange(i-3, i):
					ffs_temp = math.exp(kde.score_samples([[ft1[x]]])[0])
					std_temp = math.exp((kdeList[ft4[x]]).score_samples([[ft1[i]]])[0])
					penalty = 1.0 - abs(ffs_temp - std_temp)
					print ft4[i], " - ", ft4[x], "  ", ft1[i], " ", ft1[x]
					ffs_temp = abs(ffs_temp - std_score) + penalty
					ffs_metric += ffs_temp
				for x in xrange(i+1, i+4):
					ffs_temp = math.exp(kde.score_samples([[ft1[x]]])[0])
					std_temp = math.exp((kdeList[ft4[x]]).score_samples([[ft1[i]]])[0])
					penalty = 1.0 - abs(ffs_temp - std_temp)
					print ft4[i], " - ", ft4[x], "  ", ft1[i], " ", ft1[x]
					ffs_temp = abs(ffs_temp - std_score) + penalty
					ffs_metric += ffs_temp
				ffs_metric = ffs_metric / 6.0
				print ""
			else:
				for x in xrange(i-3, i):
					ffs_temp = math.exp(kde.score_samples([[ft1[x]]])[0])
					std_temp = math.exp((kdeList[ft4[x]]).score_samples([[ft1[i]]])[0])
					penalty = 1.0 - abs(ffs_temp - std_temp)
					ffs_temp = abs(ffs_temp - std_score) + penalty
					ffs_metric += ffs_temp
				for x in xrange(i+1, 235446):
					ffs_temp = math.exp(kde.score_samples([[ft1[x]]])[0])
					std_temp = math.exp((kdeList[ft4[x]]).score_samples([[ft1[i]]])[0])
					penalty = 1.0 - abs(ffs_temp - std_temp)
					ffs_temp = abs(ffs_temp - std_score) + penalty
					ffs_metric += ffs_temp
				ffs_metric = ffs_metric / (235448-i)

			ffsharing[i] = ffs_metric

	for x in xrange(1,100):
		print ffsharing[x]

	my_list = zip(ffsharing)
	writeCSV('test_ffsharing_6.csv', my_list)
	print "Data written..."
	print "checkFSharMetric passed..." + '\n'


###################################################################
#                    Fuzzy False Sharing V8
###################################################################

def checkFSharMetric_8():
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

	# build dictionary for the metric (corresponding to 32 CPUs)
	myDict = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], \
	          8: [], 9: [], 10: [], 11: [], 12: [], 13: [], 14: [], 15: [],\
	          16: [], 17: [], 18: [], 19: [], 20: [], 21: [], 22: [], 23: [], \
	          24: [], 25: [], 26: [], 27: [], 28: [], 29: [], 30: [], 31: []}

	# gather value for addr of each CPU ID
	for i in xrange(235446):
		if ft2[i] == 1:  # L1 cache
			myDict[ft4[i]].append(ft1[i])

	# build kde list for estimating distribution of address
	kdeList = []
	kdeScores = [0]*235446

	for l in xrange(32):
		CPU_d = [] # copying addr data for each CPU
		for i in xrange(int(len(myDict[l]))):
			CPU_d.append([myDict[l][i]])

		# build kde for current cpu
		kde = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(CPU_d)
		kdeList.append(kde)


	# create list for the metric
	ffsharing = [0.0]*235446
	counter  = 0.0

	print "Begin to calculate probability for all data points..."
	print ""
	for i in xrange(235446):
		if ft2[i] == 1:  # L1 cache
			ffs_metric = 0.0

			kde = kdeList[ft4[i]]
			std_score1 = math.exp(kde.score_samples([[ft1[i]]])[0]) # kde i; addr i

			if i < 3:
				for x in xrange(0,i-1): # within 10 rounds (short period of time)
					std_score2 = math.exp(kde.score_samples([[ft1[x]]])[0]) # kde i; addr x
					std_score3 = math.exp((kdeList[ft4[x]]).score_samples([[ft1[i]]])[0]) # kde x; addr i
					std_score4 = math.exp((kdeList[ft4[x]]).score_samples([[ft1[x]]])[0]) # kde x; addr x
					penalty = 1.0 - max(abs(std_score2 - std_score3), abs(std_score1 - std_score4))
					# print ft4[i], " - ", ft4[x], "  ", ft1[i], " ", ft1[x]
					ffs_temp = max(abs(std_score1 - std_score2), abs(std_score3 - std_score4)) + penalty
					ffs_metric += ffs_temp
				for x in xrange(i+1,i+4): # within 10 rounds (short period of time)
					std_score2 = math.exp(kde.score_samples([[ft1[x]]])[0])
					std_score3 = math.exp((kdeList[ft4[x]]).score_samples([[ft1[i]]])[0])
					std_score4 = math.exp((kdeList[ft4[x]]).score_samples([[ft1[x]]])[0])
					penalty = 1.0 - max(abs(std_score2 - std_score3), abs(std_score1 - std_score4))
					# print ft4[i], " - ", ft4[x], "  ", ft1[i], " ", ft1[x]
					ffs_temp = max(abs(std_score1 - std_score2), abs(std_score3 - std_score4)) + penalty
					ffs_metric += ffs_temp
				ffs_metric = ffs_metric / (i+3)
				# print ""
			elif i >= 3 and i < 235443:
				for x in xrange(i-3, i):
					std_score2 = math.exp(kde.score_samples([[ft1[x]]])[0])
					std_score3 = math.exp((kdeList[ft4[x]]).score_samples([[ft1[i]]])[0])
					std_score4 = math.exp((kdeList[ft4[x]]).score_samples([[ft1[x]]])[0])
					penalty = 1.0 - max(abs(std_score2 - std_score3), abs(std_score1 - std_score4))
					# print ft4[i], " - ", ft4[x], "  ", ft1[i], " ", ft1[x]
					ffs_temp = max(abs(std_score1 - std_score2), abs(std_score3 - std_score4)) + penalty
					ffs_metric += ffs_temp
				for x in xrange(i+1, i+4):
					std_score2 = math.exp(kde.score_samples([[ft1[x]]])[0])
					std_score3 = math.exp((kdeList[ft4[x]]).score_samples([[ft1[i]]])[0])
					std_score4 = math.exp((kdeList[ft4[x]]).score_samples([[ft1[x]]])[0])
					penalty = 1.0 - max(abs(std_score2 - std_score3), abs(std_score1 - std_score4))
					# print ft4[i], " - ", ft4[x], "  ", ft1[i], " ", ft1[x]
					ffs_temp = max(abs(std_score1 - std_score2), abs(std_score3 - std_score4)) + penalty
					ffs_metric += ffs_temp
				ffs_metric = ffs_metric / 6.0
				# print ""
			else:
				for x in xrange(i-3, i):
					std_score2 = math.exp(kde.score_samples([[ft1[x]]])[0])
					std_score3 = math.exp((kdeList[ft4[x]]).score_samples([[ft1[i]]])[0])
					std_score4 = math.exp((kdeList[ft4[x]]).score_samples([[ft1[x]]])[0])
					penalty = 1.0 - max(abs(std_score2 - std_score3), abs(std_score1 - std_score4))
					# print ft4[i], " - ", ft4[x], "  ", ft1[i], " ", ft1[x]
					ffs_temp = max(abs(std_score1 - std_score2), abs(std_score3 - std_score4)) + penalty
					ffs_metric += ffs_temp
				for x in xrange(i+1, 235446):
					std_score2 = math.exp(kde.score_samples([[ft1[x]]])[0])
					std_score3 = math.exp((kdeList[ft4[x]]).score_samples([[ft1[i]]])[0])
					std_score4 = math.exp((kdeList[ft4[x]]).score_samples([[ft1[x]]])[0])
					penalty = 1.0 - max(abs(std_score2 - std_score3), abs(std_score1 - std_score4))
					# print ft4[i], " - ", ft4[x], "  ", ft1[i], " ", ft1[x]
					ffs_temp = max(abs(std_score1 - std_score2), abs(std_score3 - std_score4)) + penalty
					ffs_metric += ffs_temp
				ffs_metric = ffs_metric / (235448-i)

			ffsharing[i] = ffs_metric

	for x in xrange(1,100):
		print ffsharing[x]

	my_list = zip(ffsharing)
	writeCSV('test_ffsharing_7.csv', my_list)
	print "Data written..."
	print "checkFSharMetric passed..." + '\n'


###################################################################
#                    Fuzzy False Sharing V9
###################################################################

def checkFSharMetric_9():
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

	# build dictionary for the metric (corresponding to 32 CPUs)
	myDict = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], \
	          8: [], 9: [], 10: [], 11: [], 12: [], 13: [], 14: [], 15: [],\
	          16: [], 17: [], 18: [], 19: [], 20: [], 21: [], 22: [], 23: [], \
	          24: [], 25: [], 26: [], 27: [], 28: [], 29: [], 30: [], 31: []}

	# gather value for addr of each CPU ID
	for i in xrange(235446):
		if ft2[i] == 1:  # L1 cache
			myDict[ft4[i]].append(ft1[i])

	# build kde list for estimating distribution of address
	kdeList = []
	kdeScores = [0]*235446

	for l in xrange(32):
		CPU_d = [] # copying addr data for each CPU
		for i in xrange(int(len(myDict[l]))):
			CPU_d.append([myDict[l][i]])

		# build kde for current cpu
		kde = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(CPU_d)
		kdeList.append(kde)


	# create list for the metric
	ffsharing = [0.0]*235446
	counter  = 0.0

	print "Begin to calculate probability for all data points..."
	print ""
	for i in xrange(235446):
		if ft2[i] == 1:  # L1 cache
			ffs_metric = 0.0

			kde = kdeList[ft4[i]]
			std_score1 = math.exp(kde.score_samples([[ft1[i]]])[0]) # kde i; addr i

			if i < 3:
				for x in xrange(0,i-1): # within 10 rounds (short period of time)
					std_score2 = math.exp(kde.score_samples([[ft1[x]]])[0]) # kde i; addr x
					std_score3 = math.exp((kdeList[ft4[x]]).score_samples([[ft1[i]]])[0]) # kde x; addr i
					std_score4 = math.exp((kdeList[ft4[x]]).score_samples([[ft1[x]]])[0]) # kde x; addr x
					std_score5 = math.exp((kdeList[ft4[x]]).score_samples([[(ft1[x]/2.0+ft1[i]/2.0)]])[0])
					penalty = 1.0 - abs(std_score5 - std_score1)
					# print ft4[i], " - ", ft4[x], "  ", ft1[i], " ", ft1[x]
					ffs_temp = max(abs(std_score1 - std_score2), abs(std_score3 - std_score4)) + penalty
					ffs_metric += ffs_temp
				for x in xrange(i+1,i+4): # within 10 rounds (short period of time)
					std_score2 = math.exp(kde.score_samples([[ft1[x]]])[0]) # kde i; addr x
					std_score3 = math.exp((kdeList[ft4[x]]).score_samples([[ft1[i]]])[0]) # kde x; addr i
					std_score4 = math.exp((kdeList[ft4[x]]).score_samples([[ft1[x]]])[0]) # kde x; addr x
					std_score5 = math.exp((kdeList[ft4[x]]).score_samples([[(ft1[x]/2.0+ft1[i]/2.0)]])[0])
					penalty = 1.0 - abs(std_score5 - std_score1)
					# print ft4[i], " - ", ft4[x], "  ", ft1[i], " ", ft1[x]
					ffs_temp = max(abs(std_score1 - std_score2), abs(std_score3 - std_score4)) + penalty
					ffs_metric += ffs_temp
				ffs_metric = ffs_metric / (i+3)
				# print ""
			elif i >= 3 and i < 235443:
				for x in xrange(i-3, i):
					std_score2 = math.exp(kde.score_samples([[ft1[x]]])[0]) # kde i; addr x
					std_score3 = math.exp((kdeList[ft4[x]]).score_samples([[ft1[i]]])[0]) # kde x; addr i
					std_score4 = math.exp((kdeList[ft4[x]]).score_samples([[ft1[x]]])[0]) # kde x; addr x
					std_score5 = math.exp((kdeList[ft4[x]]).score_samples([[(ft1[x]/2.0+ft1[i]/2.0)]])[0])
					penalty = 1.0 - abs(std_score5 - std_score1)
					# print ft4[i], " - ", ft4[x], "  ", ft1[i], " ", ft1[x]
					ffs_temp = max(abs(std_score1 - std_score2), abs(std_score3 - std_score4)) + penalty
					ffs_metric += ffs_temp
				for x in xrange(i+1, i+4):
					std_score2 = math.exp(kde.score_samples([[ft1[x]]])[0]) # kde i; addr x
					std_score3 = math.exp((kdeList[ft4[x]]).score_samples([[ft1[i]]])[0]) # kde x; addr i
					std_score4 = math.exp((kdeList[ft4[x]]).score_samples([[ft1[x]]])[0]) # kde x; addr x
					std_score5 = math.exp((kdeList[ft4[x]]).score_samples([[(ft1[x]/2.0+ft1[i]/2.0)]])[0])
					penalty = 1.0 - abs(std_score5 - std_score1)
					# print ft4[i], " - ", ft4[x], "  ", ft1[i], " ", ft1[x]
					ffs_temp = max(abs(std_score1 - std_score2), abs(std_score3 - std_score4)) + penalty
					ffs_metric += ffs_temp
				ffs_metric = ffs_metric / 6.0
				# print ""
			else:
				for x in xrange(i-3, i):
					std_score2 = math.exp(kde.score_samples([[ft1[x]]])[0]) # kde i; addr x
					std_score3 = math.exp((kdeList[ft4[x]]).score_samples([[ft1[i]]])[0]) # kde x; addr i
					std_score4 = math.exp((kdeList[ft4[x]]).score_samples([[ft1[x]]])[0]) # kde x; addr x
					std_score5 = math.exp((kdeList[ft4[x]]).score_samples([[(ft1[x]/2.0+ft1[i]/2.0)]])[0])
					penalty = 1.0 - abs(std_score5 - std_score1)
					# print ft4[i], " - ", ft4[x], "  ", ft1[i], " ", ft1[x]
					ffs_temp = max(abs(std_score1 - std_score2), abs(std_score3 - std_score4)) + penalty
					ffs_metric += ffs_temp
				for x in xrange(i+1, 235446):
					std_score2 = math.exp(kde.score_samples([[ft1[x]]])[0]) # kde i; addr x
					std_score3 = math.exp((kdeList[ft4[x]]).score_samples([[ft1[i]]])[0]) # kde x; addr i
					std_score4 = math.exp((kdeList[ft4[x]]).score_samples([[ft1[x]]])[0]) # kde x; addr x
					std_score5 = math.exp((kdeList[ft4[x]]).score_samples([[(ft1[x]/2.0+ft1[i]/2.0)]])[0])
					penalty = 1.0 - abs(std_score5 - std_score1)
					# print ft4[i], " - ", ft4[x], "  ", ft1[i], " ", ft1[x]
					ffs_temp = max(abs(std_score1 - std_score2), abs(std_score3 - std_score4)) + penalty
					ffs_metric += ffs_temp
				ffs_metric = ffs_metric / (235448-i)

			ffsharing[i] = ffs_metric

	for x in xrange(1,100):
		print ffsharing[x]

	my_list = zip(ffsharing)
	writeCSV('test_ffsharing_8.csv', my_list)
	print "Data written..."
	print "checkFSharMetric passed..." + '\n'


###################################################################
#                    Fuzzy False Sharing V10
###################################################################

def checkFSharMetric_10():
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

	# build dictionary for the metric (corresponding to 32 CPUs)
	myDict = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], \
	          8: [], 9: [], 10: [], 11: [], 12: [], 13: [], 14: [], 15: [],\
	          16: [], 17: [], 18: [], 19: [], 20: [], 21: [], 22: [], 23: [], \
	          24: [], 25: [], 26: [], 27: [], 28: [], 29: [], 30: [], 31: []}

	# gather value for addr of each CPU ID
	for i in xrange(235446):
		if ft2[i] == 1:  # L1 cache
			myDict[ft4[i]].append(ft1[i])

	# build kde list for estimating distribution of address
	kdeList = []
	kdeScores = [0]*235446

	for l in xrange(32):
		CPU_d = [] # copying addr data for each CPU
		for i in xrange(int(len(myDict[l]))):
			CPU_d.append([myDict[l][i]])

		# build kde for current cpu
		kde = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(CPU_d)
		kdeList.append(kde)


	# create list for the metric
	ffsharing = [0.0]*235446
	counter  = 0.0

	print "Begin to calculate probability for all data points..."
	print ""
	for i in xrange(235446):
		if ft2[i] == 1:  # L1 cache
			ffs_metric = 0.0

			kde = kdeList[ft4[i]]
			std_score1 = math.exp(kde.score_samples([[ft1[i]]])[0]) # kde i; addr i

			if i < 3:
				for x in xrange(0,i-1): # within 10 rounds (short period of time)
					std_score2 = math.exp(kde.score_samples([[ft1[x]]])[0]) # kde i; addr x
					std_score3 = math.exp((kdeList[ft4[x]]).score_samples([[ft1[i]]])[0]) # kde x; addr i
					std_score4 = math.exp((kdeList[ft4[x]]).score_samples([[ft1[x]]])[0]) # kde x; addr x
					std_score5 = math.exp((kdeList[ft4[x]]).score_samples([[(ft1[x]/2.0+ft1[i]/2.0)]])[0])
					penalty = 1.0 - abs(std_score5 - std_score1)
					# print ft4[i], " - ", ft4[x], "  ", ft1[i], " ", ft1[x]
					ffs_temp = max(abs(std_score1 - std_score2), abs(std_score3 - std_score4)) + penalty
					ffs_metric += ffs_temp
				for x in xrange(i+1,i+4): # within 10 rounds (short period of time)
					std_score2 = math.exp(kde.score_samples([[ft1[x]]])[0]) # kde i; addr x
					std_score3 = math.exp((kdeList[ft4[x]]).score_samples([[ft1[i]]])[0]) # kde x; addr i
					std_score4 = math.exp((kdeList[ft4[x]]).score_samples([[ft1[x]]])[0]) # kde x; addr x
					std_score5 = math.exp((kdeList[ft4[x]]).score_samples([[(ft1[x]/2.0+ft1[i]/2.0)]])[0])
					penalty = 1.0 - abs(std_score5 - std_score1)
					# print ft4[i], " - ", ft4[x], "  ", ft1[i], " ", ft1[x]
					ffs_temp = max(abs(std_score1 - std_score2), abs(std_score3 - std_score4)) + penalty
					ffs_metric += ffs_temp
				ffs_metric = ffs_metric / (i+3)
				# print ""
			elif i >= 3 and i < 235443:
				for x in xrange(i-3, i):
					std_score2 = math.exp(kde.score_samples([[ft1[x]]])[0]) # kde i; addr x
					std_score3 = math.exp((kdeList[ft4[x]]).score_samples([[ft1[i]]])[0]) # kde x; addr i
					std_score4 = math.exp((kdeList[ft4[x]]).score_samples([[ft1[x]]])[0]) # kde x; addr x
					std_score5 = math.exp((kdeList[ft4[x]]).score_samples([[(ft1[x]/2.0+ft1[i]/2.0)]])[0])
					penalty = 1.0 - abs(std_score5 - std_score1)
					# print ft4[i], " - ", ft4[x], "  ", ft1[i], " ", ft1[x]
					ffs_temp = max(abs(std_score1 - std_score2), abs(std_score3 - std_score4)) + penalty
					ffs_metric += ffs_temp
				for x in xrange(i+1, i+4):
					std_score2 = math.exp(kde.score_samples([[ft1[x]]])[0]) # kde i; addr x
					std_score3 = math.exp((kdeList[ft4[x]]).score_samples([[ft1[i]]])[0]) # kde x; addr i
					std_score4 = math.exp((kdeList[ft4[x]]).score_samples([[ft1[x]]])[0]) # kde x; addr x
					std_score5 = math.exp((kdeList[ft4[x]]).score_samples([[(ft1[x]/2.0+ft1[i]/2.0)]])[0])
					penalty = 1.0 - abs(std_score5 - std_score1)
					# print ft4[i], " - ", ft4[x], "  ", ft1[i], " ", ft1[x]
					ffs_temp = max(abs(std_score1 - std_score2), abs(std_score3 - std_score4)) + penalty
					ffs_metric += ffs_temp
				ffs_metric = ffs_metric / 6.0
				# print ""
			else:
				for x in xrange(i-3, i):
					std_score2 = math.exp(kde.score_samples([[ft1[x]]])[0]) # kde i; addr x
					std_score3 = math.exp((kdeList[ft4[x]]).score_samples([[ft1[i]]])[0]) # kde x; addr i
					std_score4 = math.exp((kdeList[ft4[x]]).score_samples([[ft1[x]]])[0]) # kde x; addr x
					std_score5 = math.exp((kdeList[ft4[x]]).score_samples([[(ft1[x]/2.0+ft1[i]/2.0)]])[0])
					penalty = 1.0 - abs(std_score5 - std_score1)
					# print ft4[i], " - ", ft4[x], "  ", ft1[i], " ", ft1[x]
					ffs_temp = max(abs(std_score1 - std_score2), abs(std_score3 - std_score4)) + penalty
					ffs_metric += ffs_temp
				for x in xrange(i+1, 235446):
					std_score2 = math.exp(kde.score_samples([[ft1[x]]])[0]) # kde i; addr x
					std_score3 = math.exp((kdeList[ft4[x]]).score_samples([[ft1[i]]])[0]) # kde x; addr i
					std_score4 = math.exp((kdeList[ft4[x]]).score_samples([[ft1[x]]])[0]) # kde x; addr x
					std_score5 = math.exp((kdeList[ft4[x]]).score_samples([[(ft1[x]/2.0+ft1[i]/2.0)]])[0])
					penalty = 1.0 - abs(std_score5 - std_score1)
					# print ft4[i], " - ", ft4[x], "  ", ft1[i], " ", ft1[x]
					ffs_temp = max(abs(std_score1 - std_score2), abs(std_score3 - std_score4)) + penalty
					ffs_metric += ffs_temp
				ffs_metric = ffs_metric / (235448-i)

			ffsharing[i] = ffs_metric

	for x in xrange(1,100):
		print ffsharing[x]

	my_list = zip(ffsharing)
	writeCSV('test_ffsharing_8.csv', my_list)
	print "Data written..."
	print "checkFSharMetric passed..." + '\n'








###################################################################
#                          metric scratch
###################################################################

def metric_plot():
	"""
	Visualizing the false sharing metric.
	"""
	# first read in some features from sample
	data = Data()
	ft1 = extract('samples.csv', 14, start=1)
	ft2 = extract('samples.csv', 17, start=1)
	ft3 = extract('samples.csv', 13, start=1)
	ft4 = extract('samples.csv', 15, start=1)
	print "Data loaded..."

	# do some optimization
	ft1 = toLong(toFloat(ft1)) # data address
	ft2 = toInteger(ft2) # Cache raw
	ft3 = toInteger(ft3) # timestamp
	ft4 = toInteger(ft4) # CPU
	ft2 = map(lambda x: map_data_src(x), ft2) # Cache decoded
	ft3 = subTimeBase(ft3) # timestamp subtracted from the base
	print "Data optimized..."

	my_list = zip(ft1,ft4,ft2,ft3)
	writeCSV('test_fsharing.csv', my_list)
	print "Data written..."

	# build dictionary for the metric (corresponding to 32 CPUs)
	myDict = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], \
	          8: [], 9: [], 10: [], 11: [], 12: [], 13: [], 14: [], 15: [],\
	          16: [], 17: [], 18: [], 19: [], 20: [], 21: [], 22: [], 23: [], \
	          24: [], 25: [], 26: [], 27: [], 28: [], 29: [], 30: [], 31: []}

	# gather value for addr of each CPU ID
	# form: {0:[(addr0,time0), (addr1,time1), ...], ...}
	for i in xrange(235446):
		myDict[ft4[i]].append((ft1[i],ft3[i]))

	# form: {0:[(addr0,addr1, ...), (time0,time1, ...)], ...}
	for i in xrange(32):
		myDict[i] = zip(*myDict[i])

	d11 = np.array(list(myDict[0][0])) # CPU 0 (addr, time)
	max0 = max(list(myDict[0][0]))
	min0 = min(list(myDict[0][0]))
	d12 = np.array(list(myDict[0][1]))
	max1 = max(list(myDict[0][1]))
	min1 = min(list(myDict[0][1]))
	d21 = np.array(list(myDict[1][0])) # CPU 1 (addr, time)
	max2 = max(list(myDict[1][0]))
	min2 = min(list(myDict[1][0]))
	d22 = np.array(list(myDict[1][1]))
	max3 = max(list(myDict[1][1]))
	min3 = min(list(myDict[1][1]))
	d31 = np.array(list(myDict[2][0])) # CPU 2 (addr, time)
	max4 = max(list(myDict[2][0]))
	min4 = min(list(myDict[2][0]))
	d32 = np.array(list(myDict[2][1]))
	max5 = max(list(myDict[2][1]))
	min5 = min(list(myDict[2][1]))

	# should be replaced be the overall max and min in the real calculation process

	# print max0,max1,max2,max3,min0,min1,min2,min3

	maxx = max(max0,max2,max4)
	maxy = max(max1,max3,max5)
	minx = min(min0,min2,min4)
	miny = min(min1,min3,min5)

	dens_u1 = sm.nonparametric.KDEMultivariate(data=[d11,d12],var_type='cc', bw='normal_reference')
	dens_u2 = sm.nonparametric.KDEMultivariate(data=[d21,d22],var_type='cc', bw='normal_reference')
	dens_u3 = sm.nonparametric.KDEMultivariate(data=[d31,d32],var_type='cc', bw='normal_reference')

	# plot 3d kde
	fig = plt.figure()
	ax1 = fig.add_subplot(131, projection='3d')
	ax2 = fig.add_subplot(132, projection='3d')
	ax3 = fig.add_subplot(133, projection='3d')

	x = np.arange(minx, maxx, (maxx-minx)/100.0)
	y = np.arange(miny, maxy, (maxy-miny)/100.0)
	x,y = np.meshgrid(x, y)

	print len(x)
	print len(x[0])
	print len(y)
	print len(y[0])
	
	z0 = []
	z1 = []
	z2 = []
	for i in xrange(len(x)):
		z_0 = []
		z_1 = []
		z_2 = []
		for j in xrange(len(y)):
			tempa = float(dens_u1.pdf([x[0][i],y[j][0]]))
			tempb = float(dens_u2.pdf([x[0][i],y[j][0]]))
			tempc = float(dens_u3.pdf([x[0][i],y[j][0]]))
			z_0.append(tempa*tempb)
			z_1.append(tempa*tempc)
			z_2.append(z_0[j]+z_1[j])
		z0.append(z_0)
		z1.append(z_1)
		z2.append(z_2)

	wire1 = ax1.plot_wireframe(x,y,z0,rstride=1,cstride=1)
	wire2 = ax2.plot_wireframe(x,y,z1,rstride=1,cstride=1)
	wire3 = ax3.plot_wireframe(x,y,z2,rstride=1,cstride=1)
	fig.set_size_inches(20, 5.7, forward=True)

	ax1.set_xlabel('addr')
	ax1.set_ylabel('time')
	ax1.set_zlabel('prob')
	ax2.set_xlabel('addr')
	ax2.set_ylabel('time')
	ax2.set_zlabel('prob')
	ax3.set_xlabel('addr')
	ax3.set_ylabel('time')
	ax3.set_zlabel('prob')

	# ax.set_ylim3d(2.5e10, 3.8e10)
	plt.show()


def process_kde_data():
	"""
	Calculate and store kde data for each CPU.
	"""
	# first read in some features from sample
	data = Data()
	ft1 = extract('samples.csv', 14, start=1)
	ft2 = extract('samples.csv', 17, start=1)
	ft3 = extract('samples.csv', 13, start=1)
	ft4 = extract('samples.csv', 15, start=1)
	print "Data loaded..."

	# do some optimization
	ft1 = toLong(toFloat(ft1)) # data address
	ft2 = toInteger(ft2) # Cache raw
	ft3 = toInteger(ft3) # timestamp
	ft4 = toInteger(ft4) # CPU
	ft2 = map(lambda x: map_data_src(x), ft2) # Cache decoded
	ft3 = subTimeBase(ft3) # timestamp subtracted from the base
	print "Data optimized..."

	my_list = zip(ft1,ft4,ft2,ft3)
	writeCSV('test_fsharing.csv', my_list)
	print "Data written..."

	# build dictionary for the metric (corresponding to 32 CPUs)
	myDict = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], \
	          8: [], 9: [], 10: [], 11: [], 12: [], 13: [], 14: [], 15: [],\
	          16: [], 17: [], 18: [], 19: [], 20: [], 21: [], 22: [], 23: [], \
	          24: [], 25: [], 26: [], 27: [], 28: [], 29: [], 30: [], 31: []}

	# gather value for addr of each CPU ID
	# form: {0:[(addr0,time0), (addr1,time1), ...], ...}
	for i in xrange(235446):
		myDict[ft4[i]].append((ft1[i],ft3[i]))

	# find max and min
	addr_max = max(ft2)
	addr_min = min(ft2)
	time_max = max(ft3)
	time_min = min(ft3)

	# form: {0:[(addr0,addr1, ...), (time0,time1, ...)], ...}
	for i in xrange(32):
		myDict[i] = zip(*myDict[i])

	time_addr = []
	for i in xrange(32):
		for j in xrange(2):
			time_addr.append(np.array(list(myDict[i][j])))

	dens_us = []
	for i in xrange(32):
		d1 = time_addr[i*2]
		d2 = time_addr[i*2+1]
		tempdens = sm.nonparametric.KDEMultivariate(data=[d1,d2],var_type='cc', bw='normal_reference')
		dens_us.append(tempdens)

	x = np.arange(addr_min, addr_max, (addr_max-addr_min)/100.1)
	y = np.arange(time_min, time_max, (time_max-time_min)/100.1)
	x,y = np.meshgrid(x, y)

	print len(x)
	print len(y)
	
	for i in xrange(32):
		kdeout = []
		for j in xrange(len(x)):
			kdein = []
			for k in xrange(len(y)):
				tempkde = float(dens_us[i].pdf([x[0][j],y[k][0]]))
				kdein.append(tempkde)
			kdeout.append(kdein)
		flatkde = sum(kdeout,[])
		zipkde = zip(flatkde)
		file_name = 'cpu_kde_'
		if i < 10:
			file_name = file_name + '0' + str(i) + '.csv'
		else:
			file_name = file_name + str(i) + '.csv'
		writeCSV(file_name, zipkde)
		print "Finished writing data for CPU " + str(i)

def process_xyrange():
	"""
	Calculate x and y ranges.
	"""
	# first read in some features from sample
	data = Data()
	ft1 = extract('samples.csv', 14, start=1)
	ft2 = extract('samples.csv', 17, start=1)
	ft3 = extract('samples.csv', 13, start=1)
	ft4 = extract('samples.csv', 15, start=1)
	print "Data loaded..."

	# do some optimization
	ft1 = toLong(toFloat(ft1)) # data address
	ft2 = toInteger(ft2) # Cache raw
	ft3 = toInteger(ft3) # timestamp
	ft4 = toInteger(ft4) # CPU
	ft2 = map(lambda x: map_data_src(x), ft2) # Cache decoded
	ft3 = subTimeBase(ft3) # timestamp subtracted from the base
	print "Data optimized..."

	# build dictionary for the metric (corresponding to 32 CPUs)
	myDict = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], \
	          8: [], 9: [], 10: [], 11: [], 12: [], 13: [], 14: [], 15: [],\
	          16: [], 17: [], 18: [], 19: [], 20: [], 21: [], 22: [], 23: [], \
	          24: [], 25: [], 26: [], 27: [], 28: [], 29: [], 30: [], 31: []}

	# gather value for addr of each CPU ID
	# form: {0:[(addr0,time0), (addr1,time1), ...], ...}
	for i in xrange(235446):
		myDict[ft4[i]].append((ft1[i],ft3[i]))

	# find max and min
	addr_max = max(ft2)
	addr_min = min(ft2)
	time_max = max(ft3)
	time_min = min(ft3)

	return addr_max,addr_min,time_max,time_min

def product_sum_kde(lenx, leny, addr_max, addr_min, time_max, time_min):
	# extract all 32 matrix
	matrix_list = []
	for i in xrange(32):
		data = Data()
		file_name = 'cpu_kde_'
		if i < 10:
			file_name = file_name + '0' + str(i) + '.csv'
		else:
			file_name = file_name + str(i) + '.csv'
		matrixr = extract(file_name, 0, start=0)
		matrixr = toFloat(matrixr)
		matrix = []
		for x in xrange(lenx):
			tempm = []
			for y in xrange(leny):
				tempm.append(matrixr[x*leny+y])
			matrix.append(tempm)
	print "kde data extracted..."

	# do pairwise calculation
	z_res = [0.0]*lenx*leny
	for i in xrange(32):
		for j in xrange(i+1,32):
			z = []
			for x in xrange(lenx):
				temp_z = []
				for y in xrange(leny):
					temp_z.append(matrix_list[i][x][y] * matrix_list[j][x][y])
				z.append(temp_z)
			zflat = sum(z,[])
			z_res = map(lambda x,y: x+y, z_res, zflat)
	print "Finished pairwise calculation..."

	# calculate final matrix z
	z = []
	for x in xrange(lenx):
		temp_z = []
		for y in xrange(leny):
			temp_z.append(z_res[x*leny+y])
		z.append(temp_z)
	print "z calculated..."

	# plot in 3D
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	x = np.arange(addr_min, addr_max, (addr_max-addr_min)/100.1)
	y = np.arange(time_min, time_max, (time_max-time_min)/100.1)
	x,y = np.meshgrid(x, y)
	wire = ax.plot_wireframe(x,y,z,rstride=1,cstride=1)

	ax.set_xlabel('addr')
	ax.set_ylabel('time')
	ax.set_zlabel('prob')

	plt.show()


###################################################################
#                              testing
###################################################################

# checkFSharMetric_1()
# checkFSharMetric_2()
# checkFSharMetric_3()
# checkFSharMetric_4()
# checkFSharMetric_5()
# checkFSharMetric_6()

# Note: could be very slow
# checkFSharMetric_7()
# checkFSharMetric_8()
# checkFSharMetric_9()
# checkFSharMetric_10()

# step by step process data for 32 CPUs
# metric_plot()
# process_kde_data()
xmax,xmin,ymax,ymin = process_xyrange()
product_sum_kde(101,101,xmax,xmin,ymax,ymin)