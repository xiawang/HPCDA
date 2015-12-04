import os
import csv
import numpy as np; np.random.seed(0)
import seaborn as sns; sns.set(style="white", color_codes=True)
import matplotlib.pyplot as plt
import pandas as pd

from string import punctuation

from processdata import *
from kmeans import *
from optimize import *

"""
Class for optimizing input data (especially csv). 
"""

######################################################################
# basic optimization
######################################################################

# basic standardization
def toFloat(arr):
	"""
	Converting elements in an array to floats.
	"""
	fltarr = map(lambda x: float(x), arr)
	return fltarr

def toInteger(arr):
	"""
	Converting elements in an array to integers.
	"""
	intarr = map(lambda x: int(x), arr)
	return intarr

def toStr(arr):
	"""
	Converting elements in an array to strings.
	"""
	strarr = map(lambda x: str(x), arr)
	return strarr

def toLong(arr):
	"""
	Converting elements in an array to longs.
	"""
	longarr = map(lambda x: long(x), arr)
	return longarr

def bsc_min_standardization(arr):
	"""
	Preprocess the data by taking min and change 
	other data accordingly.
	"""
	arr = toFloat(arr) # convert to floats
	nparr = np.array(arr) # use numpy array
	nparrmin = np.amin(nparr)
	stdarr = map(lambda x: x-nparrmin+1, nparr)
	return stdarr

def bsc_mean_standardization(arr):
	"""
	Preprocess the data by taking mean and change 
	other data accordingly.
	"""
	arr = toFloat(arr) # convert to floats
	nparr = np.array(arr) #use numpy array
	nparrmean = np.mean(nparr)
	stdarr = map(lambda x: x-nparrmean, nparr)
	return stdarr

def bsc_mean_norm(arr):
	"""
	Preprocess the data by taking mean and change 
	other data accordingly by dviding the mean.
	"""
	arr = toFloat(arr) # convert to floats
	nparr = np.array(arr) #use numpy array
	nparrmean = np.mean(nparr)
	stdarr = map(lambda x: x*1.0/nparrmean, nparr)
	return stdarr

def toHour(secArray):
	"""
	Convert array of seconds to hours.
	"""
	arr = toFloat(secArray) # convert to floats
	nparr = np.array(arr) # use numpy array
	hArray = map(lambda x: x/3600, nparr)
	return hArray


def extract_map(infile, infy):
	"""
	Convert array of hasing numbers to different catagories.
	"""
	data = Data()
	data.load(infile,ify=infy)
	X,y = data.getXy()
	feature = []

	for i in range(data.length()):
	    feature.append(float(X[i][0]))

	frec_map = {}
	index = 1
	for elmt in feature:
		if elmt in frec_map:
			frec_map[elmt][1] += 1
		else:
			frec_map[elmt] = [index,1]
			index += 1
	return frec_map

def map_data_src(src):
	"""
	Specially designed for data_src feature. Decoding the 
	data-src to different memory catagories.
	"""
	src >>= 5
	if src & 0x08: # PERF_MEM_LVL_L1 - L1
		return 1
	if src & 0x10: # PERF_MEM_LVL_LFB - Line Fill Buffer
		return 1
	if src & 0x20: # PERF_MEM_LVL_L2 - L2
		return 2
	if src & 0x40: # PERF_MEM_LVL_L3 - L3
		return 3
	if src & 0x80: # PERF_MEM_LVL_LOC_RAM - Local DRAM
		return 4
	if src & 0x100: # PERF_MEM_LVL_REM_RAM1 - Remote DRAM (1 hop)
		return 5
	if src & 0x200: # PERF_MEM_LVL_REM_RAM2 - Remote DRAM (2 hops)
		return 6
	if src & 0x400: # PERF_MEM_LVL_REM_CCE1 - Remote Cache (1 hop)
		return 5
	if src & 0x800: # PERF_MEM_LVL_REM_CCE2 - Remote Cache (2 hops)
		return 6
	if src & 0x01: # PERF_MEM_LVL_NA - not available
		return -1
	if src & 0x1000: # PERF_MEM_LVL_IO - I/O memory
		return -2
	if src & 0x2000: # PERF_MEM_LVL_UNC - Uncached memory
		return -3
	return -4

def map_data_src_f(src):
	if src == -4:
		return 0
	if src == -3:
		return 1
	if src == 1:
		return 2
	if src == 2:
		return 3
	if src == 3:
		return 4
	return 5


def twin_proc(cpu):
	"""
	Specially designed for calculating the cpu sharing
	same L1 and L2 cache.
	"""
	if cpu < 16:
		return cpu+16
	else:
		return cpu-16


def closest_addr_region(avrg_addr, addr):
	"""
	Specially designed for calculating the closest
	memory address region for each cpu.
	"""
	diff = map(lambda x: abs(x-addr), avrg_addr)
	min_idx = diff.index(min(diff))
	return min_idx