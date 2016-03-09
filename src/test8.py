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

k = []
x = 20
y = 10
for i in xrange(x):
	ki = []
	for j in xrange(y):
		ki.append(j)
	k.append(ki)

print k

flatk = sum(k,[])
zipk = zip(flatk)

writeCSV('test8.csv', zipk)

data = Data()
matrixr = extract('test8.csv', 0, start=0)
matrixr = toFloat(matrixr)
matrix = []
for i in xrange(x):
	tempm = []
	for j in xrange(y):
		tempm.append(matrixr[i*y+j])
	matrix.append(tempm)

print matrix