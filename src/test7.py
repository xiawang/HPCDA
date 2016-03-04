from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(color_codes=True)
import math
import random
import numpy as np; np.random.seed(12345)
from scipy import stats
import statsmodels.api as sm
from sklearn.neighbors import KernelDensity
import pandas as pd


def test1():
	X = [[1], [2], [3], [3], [2], [8], [8], [9], [10], [12], [11], [9]]
	kde = KernelDensity(kernel='gaussian', bandwidth=0.4).fit(X)
	scores = kde.score_samples(X)
	for x in xrange(len(scores)):
		scores[x] = math.exp(scores[x])
	print scores
	Y = [[1], [2], [2], [1], [5], [6], [6], [7], [9], [10], [8], [7]]
	density = kde.score_samples(Y)
	for x in xrange(len(density)):
		density[x] = math.exp(density[x])
	print density


def test2():
	X = [[1], [2], [3], [3], [2], [8], [8], [9], [10], [12], [11], [9]]
	kde1 = KernelDensity(kernel='gaussian', bandwidth=0.4).fit(X)
	scores1 = kde1.score_samples(X)
	for x in xrange(len(scores1)):
		scores1[x] = math.exp(scores1[x])
	Y = [[1], [4], [8], [9], [11], [13], [10], [7], [5], [4], [2], [1]]
	kde2 = KernelDensity(kernel='gaussian', bandwidth=0.4).fit(Y)
	scores2 = kde2.score_samples(Y)
	for x in xrange(len(scores2)):
		scores2[x] = math.exp(scores2[x])
	
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	X1 = np.arange(0, 14, 0.25)
	tx = np.arange(0, 14, 0.25)
	Y1 = np.arange(0, 14, 0.25)
	ty = np.arange(0, 14, 0.25)
	tx, ty = np.meshgrid(tx, ty)
	for x in xrange(len(X1)):
		score1 = math.exp(kde1.score_samples(x)[0])
		print score1
		X1[x] = score1
	print X1
	for x in xrange(len(Y1)):
		score2 = math.exp(kde2.score_samples(x)[0])
		Y1[x] = score2
		# print Y1[x]
	Z = map(lambda x,y: (x*y), X1,Y1)
	print Z
	surf = ax.plot_surface(tx, ty, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
	        linewidth=0, antialiased=False)
	# ax.set_zlim(-1.01, 1.01)

	# ax.zaxis.set_major_locator(LinearLocator(10))
	# ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

	fig.colorbar(surf, shrink=0.5, aspect=5)

	plt.show()


def test3():
	nobs = 300
	c1 = np.random.normal(size=(nobs,1))
	print c1
	c2 = np.random.normal(2, 1, size=(nobs,1))
	dens_u = sm.nonparametric.KDEMultivariate(data=[c1,c2],var_type='cc', bw='normal_reference')
	print dens_u.cdf([1,3])


def test4():
	X = [[1], [2], [3], [3], [2], [8], [8], [9], [10], [12], [11], [9]]
	Y = [[1], [4], [8], [9], [11], [13], [10], [7], [5], [4], [2], [1]]
	dens_u = sm.nonparametric.KDEMultivariate(data=[X,Y],var_type='cc', bw='normal_reference')
	print dens_u.cdf([1,3])

	fig = plt.figure()
	ax = fig.gca(projection='3d')
	X1 = np.arange(0, 14, 0.25)
	tx = np.arange(0, 14, 0.25)
	Y1 = np.arange(0, 14, 0.25)
	ty = np.arange(0, 14, 0.25)
	Z = np.arange(0, 14, 0.25)
	tx, ty = np.meshgrid(tx, ty)
	for x in xrange(len(Z)):
		score = dens_u.cdf([X1[x],Y1[x]])
		Z[x] = score
	print Z
	surf = ax.plot_surface(tx, ty, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
	        linewidth=0, antialiased=False)
	print dens_u.pdf([10,10])

	fig.colorbar(surf, shrink=0.5, aspect=5)

	plt.show()


def test5():
	# X = [[1], [2], [3], [3], [2], [8], [8], [9], [10], [12], [11], [9]]
	X = np.array([1,2,3,3,2,8,8,9,10,12,11,9])
	# Y = [[1], [4], [8], [9], [11], [13], [10], [7], [5], [4], [2], [1]]
	Y = np.array([1,4,8,9,11,13,10,7,5,4,2,1])
	dens_u1 = sm.nonparametric.KDEUnivariate(X.astype(np.double))
	dens_u2 = sm.nonparametric.KDEUnivariate(Y.astype(np.double))
	dens_u1.fit()
	dens_u2.fit()

	# plt.plot(dens_u1.cdf)
	# plt.plot(dens_u2.cdf)
	# print dens_u1.evaluate(1)
	# print dens_u1.evaluate(2)
	# print dens_u1.evaluate(3)

	# print dens_u2.evaluate(1)
	# print dens_u2.evaluate(2)
	# print dens_u2.evaluate(3)

	fig = plt.figure()
	ax = fig.gca(projection='3d')
	X1 = np.arange(0, 14, 0.25)
	tx = np.arange(0, 14, 0.25)
	Y1 = np.arange(0, 14, 0.25)
	ty = np.arange(0, 14, 0.25)
	Z = np.arange(0, 14, 0.25)
	tx, ty = np.meshgrid(tx, ty)
	for x in xrange(len(Z)):
		score = dens_u1.evaluate([X1[x]]) * dens_u2.evaluate([Y1[x]])
		Z[x] = score
	print Z
	surf = ax.plot_surface(tx, ty, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
	        linewidth=0, antialiased=False)

	# fig.colorbar(surf, shrink=0.5, aspect=5)

	plt.show()


def test6():
	# X = [[1], [2], [3], [3], [2], [8], [8], [9], [10], [12], [11], [9]]
	X = np.array([1,2,3,3,2,8,8,9,10,12,11,9])
	# Y = [[1], [4], [8], [9], [11], [13], [10], [7], [5], [4], [2], [1]]
	Y = np.array([1,4,8,9,11,13,10,7,5,4,2,1])
	dens_u1 = sm.nonparametric.KDEUnivariate(X.astype(np.double))
	dens_u2 = sm.nonparametric.KDEUnivariate(Y.astype(np.double))
	dens_u1.fit()
	dens_u2.fit()

	# plt.plot(dens_u1.cdf)
	# plt.plot(dens_u2.cdf)
	# print dens_u1.evaluate(1)
	# print dens_u1.evaluate(2)
	# print dens_u1.evaluate(3)

	# print dens_u2.evaluate(1)
	# print dens_u2.evaluate(2)
	# print dens_u2.evaluate(3)

	# fig = plt.figure()
	# ax = fig.gca(projection='3d')
	X1 = np.arange(0, 14, 0.25)
	tx = np.arange(0, 14, 0.25)
	Y1 = np.arange(0, 14, 0.25)
	ty = np.arange(0, 14, 0.25)
	Z = np.arange(0, 14, 0.25)
	tx, ty = np.meshgrid(tx, ty)
	for x in xrange(len(Z)):
		score = dens_u1.evaluate([X1[x]]) * dens_u2.evaluate([Y1[x]])
		Z[x] = score
	
	df1 = pd.DataFrame({'x':X, 'y':Y})
	# df2 = pd.DataFrame(Y, columns=['y'])

	graph = sns.jointplot(x='x', y='y', data=df1, kind="kde")

	plt.show()


def test7():
	# X = [[1], [2], [3], [3], [2], [8], [8], [9], [10], [12], [11], [9]]
	X = np.array([1,2,3,3,2,8,8,9,10,12,11,9])
	# Y = [[1], [4], [8], [9], [11], [13], [10], [7], [5], [4], [2], [1]]
	Y = np.array([1,4,8,9,11,13,10,7,5,4,2,1])
	dens_u1 = sm.nonparametric.KDEUnivariate(X.astype(np.double))
	dens_u2 = sm.nonparametric.KDEUnivariate(Y.astype(np.double))
	dens_u1.fit()
	dens_u2.fit()

	# plt.plot(dens_u1.cdf)
	# plt.plot(dens_u2.cdf)
	# print dens_u1.evaluate(1)
	# print dens_u1.evaluate(2)
	# print dens_u1.evaluate(3)

	# print dens_u2.evaluate(1)
	# print dens_u2.evaluate(2)
	# print dens_u2.evaluate(3)

	fig = plt.figure()
	ax = fig.gca(projection='3d')
	X1 = np.arange(0, 14, 0.25)
	tx = np.arange(0, 14, 0.25)
	Y1 = np.arange(0, 14, 0.25)
	ty = np.arange(0, 14, 0.25)
	Z = np.arange(0, 14, 0.25)
	tx, ty = np.meshgrid(tx, ty)
	for x in xrange(len(Z)):
		score = dens_u1.evaluate([X1[x]]) * dens_u2.evaluate([Y1[x]])
		Z[x] = score
	print Z
	surf = ax.plot_surface(tx, ty, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
	        linewidth=0, antialiased=False)

	# fig.colorbar(surf, shrink=0.5, aspect=5)

	plt.show()

###################################################################
#                              testing
###################################################################

# test1()
# test2()
# test3()
# test4()
# test5()
test6()