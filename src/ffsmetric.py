import os
import csv
import math
import numpy as np; np.random.seed(0)
# import seaborn as sns; sns.set(color_codes=True)
# import matplotlib.pyplot as plt
# import pandas as pd
import scipy.stats
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

	# build kde list for estimating distribution of address
	kdeList = []
	kdeScores = [0]*235446

	for l in xrange(32):
		CPU_d = [] # copying addr data for each CPU
		for i in xrange(int(len(myDict[l]))):
			CPU_d.append([myDict[l][i]])

		# build kde for current cpu
		kde = KernelDensity(kernel='gaussian', bandwidth=0.3).fit(CPU_d)
		kdeList.append(kde)

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

	print "Begin to calculate probability for all data points..."
	print ""
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
					kde = kdeList[ft4[i]]
					ffs_temp = math.exp(kde.score_samples([[ft1[x]]])[0])
					ffs_metric += ffs_temp
			elif i >= 5 and i < 235441:
				for x in xrange(i-5, i+5):
					kde = kdeList[ft4[i]]
					ffs_temp = math.exp(kde.score_samples([[ft1[x]]])[0])
					ffs_metric += ffs_temp
			else:
				for x in xrange(i-5, 235446):
					kde = kdeList[ft4[i]]
					ffs_temp = math.exp(kde.score_samples([[ft1[x]]])[0])
					ffs_metric += ffs_temp

			ffsharing[i] = ffs_metric

	for x in xrange(1,100):
		print ffsharing[x]

	my_list = zip(ffsharing)
	writeCSV('test_ffsharing_2.csv', my_list)
	print "Data written..."
	print "checkFSharMetric passed..." + '\n'


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
checkFSharMetric_7()