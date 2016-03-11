import math
import random
import numpy as np; np.random.seed(12345)
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(color_codes=True)
from statsmodels.distributions.mixture_rvs import mixture_rvs
from scipy.stats.distributions import norm
from sklearn.neighbors import KernelDensity

for i in xrange(32):
	for j in xrange(32):
		if i != j and abs(i-j) != 16:
			print i, " - ", j
	print " "

for i in xrange(31):
	for j in xrange(i+1,32):
		if abs(i-j) != 16:
			print i, " - ", j
	print " "