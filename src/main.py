import random
import sys
import time

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import pandas as pd

from processdata import *
from kmeans import *
from optimize import *
from ffsmetric import *

def test1():
	# first read in some features from sample
	data = Data()
	ft1 = extract('samples.csv', 12, start=1)
	ft2 = extract('samples.csv', 13, start=1)
	ft3 = extract('samples.csv', 16, start=1)

	# do some optimization
	ft1 = bsc_min_standardization(ft1)
	ft2 = bsc_min_standardization(ft2)
	ft3 = toFloat(ft3)

	# then write out tesing csv
	my_list = zip(ft1,ft2,ft3)
	writeCSV('test.csv', my_list)

	# load testing csv and plot
	data.load('test.csv')
	X,y = data.getXy()
	feature1 = []
	feature2 = []

	for i in range(235446):
	    feature1.append(float(X[i][0]))
	    feature2.append(float(X[i][1]))

	# plot using pandas and seaborn
	df = pd.DataFrame()
	df['x1'] = feature1
	df['x2'] = feature2
	g = sns.jointplot(x="x1", y="x2", data=df, kind="hex")

	sns.plt.show();
	print "Test1 passed..." + '\n'

def test2():
	# first read in some features from sample
	data = Data()
	ft1 = extract('test.csv', 0)
	ft2 = extract('test.csv', 1)
	ft1 = bsc_mean_norm(ft1)
	ft2 = bsc_mean_norm(ft2)
	feature1 = []
	feature2 = []

	for i in range(235446):
	    feature1.append(float(ft1[i]))
	    feature2.append(float(ft2[i]))
	my_list = zip(feature1,feature2)

	# training with k-means algorithm
	y = []
	y, centers, labels_unique = useKMeans(my_list, y, n_clusters=2)
	print y
	plotKMeans(my_list,y,centers,2)
	print "Test2 passed..." + '\n'

def test3():
	# first read in some features from sample
	data = Data()
	ft1 = extract('test_data_src.csv', 0)
	ft2 = extract('test_latency.csv', 0)
	print "Data loaded..."

	# do some optimization
	ft1 = toInteger(ft1)
	ft1 = map(lambda x: map_data_src(x), ft1)
	ft1 = map(lambda x: map_data_src_f(x), ft1)
	ft2 = toFloat(ft2)
	print "Data optimized..."

	plt.scatter(ft1, ft2, alpha=0.5)
	plt.show()

	feature1 = []
	feature2 = []

	for i in range(235446):
	    feature1.append(float(ft1[i]))
	    feature2.append(float(ft2[i]))
	my_list = zip(feature1,feature2)

	# training with k-means algorithm
	y = []
	y, centers, labels_unique = useKMeans(my_list, y, n_clusters=8)
	print y
	plotKMeans(my_list,y,centers,8,-0.5,5.5,-100,4100)
	print "Test3 passed..." + '\n'

def test4():
	# check the false sharing metric and save the plot
	xmax,xmin,ymax,ymin = process_xyrange()
	product_sum_kde(111,111,xmax,xmin,ymax,ymin)

def main():
	# run tests
	test4()

# automation
if __name__ == "__main__": main()