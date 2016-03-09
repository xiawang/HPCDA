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
from mpl_toolkits.mplot3d import axes3d


def test1():
	X = [[1], [2], [4], [3], [2], [8], [8], [9], [10], [12], [11], [9]]
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
	X = [[1], [2], [4], [3], [2], [8], [8], [9], [10], [12], [11], [9]]
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
	X = [[1], [2], [4], [3], [2], [8], [8], [9], [10], [12], [11], [9]]
	Y = [[1], [2], [3], [5], [6], [7], [8], [13], [14], [15], [18], [20]]
	dens_u = sm.nonparametric.KDEMultivariate(data=[X,Y],var_type='cc', bw='normal_reference')
	print dens_u.cdf([1,3])

	fig = plt.figure()
	ax = fig.gca(projection='3d')
	X1 = np.arange(0, 14, 0.25)
	tx = np.arange(0, 14, 0.25)
	Y1 = np.arange(0, 14, 0.25)
	ty = np.arange(0, 14, 0.25)
	# Z = np.arange(0, 14, 0.25)
	tx, ty = np.meshgrid(tx, ty)
	# for x in xrange(len(Z)):
	# 	score = dens_u.cdf([X1[x],Y1[x]])
	# 	Z[x] = score

	# print Z
	Z = dens_u.pdf([X1,Y1])
	surf = ax.plot_surface(tx, ty, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
	        linewidth=0, antialiased=False)
	print dens_u.pdf([10,10])

	fig.colorbar(surf, shrink=0.5, aspect=5)

	plt.show()


def test5():
	# X = [[1], [2], [3], [3], [2], [8], [8], [9], [10], [12], [11], [9]]
	X = np.array([1,2,4,3,2,8,8,9,10,12,11,9])
	# Y = [[1], [4], [8], [9], [11], [13], [10], [7], [5], [4], [2], [1]]
	Y = np.array([1,2,3,5,6,7,8,13,14,15,18,20])
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
	print dens_u1.evaluate(2) * dens_u2.evaluate(2)
	print dens_u1.evaluate(7) * dens_u2.evaluate(7)
	print dens_u1.evaluate(11) * dens_u2.evaluate(15)
	print dens_u1.evaluate(5) * dens_u2.evaluate(15)

	fig = plt.figure()
	ax = fig.gca(projection='3d')
	X1 = np.arange(0, 14, 0.2)
	tx = np.arange(0, 14, 0.2)
	Y1 = np.arange(0, 14, 0.2)
	ty = np.arange(0, 14, 0.2)
	tx, ty = np.meshgrid(tx, ty)
	zs = np.array([dens_u1.evaluate(x) * dens_u2.evaluate(y) for x,y in zip(np.ravel(X1), np.ravel(Y1))])
	Z = zs


	# for x in xrange(len(Z)):
	# 	for y in xrange(len(Z)):
	# 		score = dens_u1.evaluate([X1[x]]) * dens_u2.evaluate([Y1[y]])
	# 		Z[x*56+y] = score

	surf = ax.plot_surface(tx, ty, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
	        linewidth=0, antialiased=False)

	plt.show()


def test6():
	# X = [[1], [2], [3], [3], [2], [8], [8], [9], [10], [12], [11], [9]]
	X = np.array([1,2,4,3,2,8,8,9,10,12,11,9])
	# Y = [[1], [4], [8], [9], [11], [13], [10], [7], [5], [4], [2], [1]]
	Y = np.array([1,2,3,5,6,7,8,13,14,15,18,20])
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
	X = np.array([1,2,4,3,2,8,8,9,10,12,11,9])
	Y = np.array([1,2,3,5,6,7,8,13,14,15,18,20])
	dens_u = sm.nonparametric.KDEMultivariate(data=[X,Y],var_type='cc', bw='normal_reference')

	fig = plt.figure()
	ax = fig.gca(projection='3d')
	X1 = np.arange(0, 14, 0.2)
	tx = np.arange(0, 14, 0.2)
	Y1 = np.arange(0, 14, 0.2)
	ty = np.arange(0, 14, 0.2)
	tx, ty = np.meshgrid(tx, ty)
	# Z = np.array([dens_u1.evaluate(x) * dens_u2.evaluate(y) for x,y in zip(np.ravel(X1), np.ravel(Y1))])

	# for x in xrange(len(Z)):
	# 	for y in xrange(len(Z)):
	# 		score = dens_u1.evaluate([X1[x]]) * dens_u2.evaluate([Y1[y]])
	# 		Z[x*56+y] = score

	# surf = ax.plot_surface(tx, ty, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
	#         linewidth=0, antialiased=False)

	print dens_u.pdf([2,2])
	print dens_u.pdf([7,7])
	print dens_u.pdf([11,15])
	print dens_u.pdf([5,15])

	# plt.show()

def test8():
	X = np.array([1,2,4,3,2,8,8,9,10,12,11,9])
	Y = np.array([1,2,3,5,6,7,8,13,14,15,18,20])

	X1 = np.array([1,3,4,3,5,5,10,11,10,10,11,12])
	Y1 = np.array([1,2,6,5,6,7,8,17,16,17,18,20])
	dens_u1 = sm.nonparametric.KDEMultivariate(data=[X,Y],var_type='cc', bw='normal_reference')
	dens_u2 = sm.nonparametric.KDEMultivariate(data=[X1,Y1],var_type='cc', bw='normal_reference')

	df1 = pd.DataFrame({'x':X, 'y':Y})
	df2 = pd.DataFrame({'x':X1, 'y':Y1})
	graph1 = sns.jointplot(x='x', y='y', data=df1, kind="kde")
	graph2 = sns.jointplot(x='x', y='y', data=df2, kind="kde")

	plt.show()

def test9():
	X = np.array([1,2,4,3,2,8,8,9,10,12,11,9])
	Y = np.array([1,2,3,5,6,7,8,13,14,15,18,20])

	dens_u1 = sm.nonparametric.KDEMultivariate(data=[X,Y],var_type='cc', bw='normal_reference')
	
	print dens_u1.pdf([4,7])
	print dens_u1.pdf([5,14])

	df1 = pd.DataFrame({'x':X, 'y':Y})
	graph1 = sns.jointplot(x='x', y='y', data=df1, kind="kde")

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')

	x = np.arange(0, 20, 0.25)
	y = np.arange(0, 20, 0.25)
	x,y = np.meshgrid(x, y)
	z1 = []
	for i in xrange(len(x)):
		z = []
		for j in xrange(len(x)):
			z.append(float(dens_u1.pdf([x[0][i],y[j][0]])))
		z1.append(z)

	# surf = ax.plot_surface(x, y, z1, rstride=1, cstride=1, cmap=cm.coolwarm,
	#         linewidth=0, antialiased=False)
	wire = ax.plot_wireframe(x,y,z1,rstride=1,cstride=1)

	plt.show()

def test10():
	X = np.array([1,2,4,3,2,8,8,9,10,12,11,9])
	Y = np.array([1,2,3,5,6,7,8,13,14,15,18,20])
	X1 = np.array([1,3,4,3,5,5,10,11,10,10,11,12])
	Y1 = np.array([1,2,6,5,6,7,8,17,16,17,18,20])

	dens_u1 = sm.nonparametric.KDEMultivariate(data=[X,Y],var_type='cc', bw='normal_reference')
	dens_u2 = sm.nonparametric.KDEMultivariate(data=[X1,Y1],var_type='cc', bw='normal_reference')

	df1 = pd.DataFrame({'x':X, 'y':Y})
	graph1 = sns.jointplot(x='x', y='y', data=df1, kind="kde")
	df2 = pd.DataFrame({'x1':X1, 'y1':Y1})
	graph2 = sns.jointplot(x='x1', y='y1', data=df2, kind="kde")

	fig = plt.figure()
	ax1 = fig.add_subplot(121, projection='3d')
	ax2 = fig.add_subplot(122, projection='3d')

	x = np.arange(0, 20, 0.25)
	y = np.arange(0, 20, 0.25)
	x,y = np.meshgrid(x, y)
	z1 = []
	z2 = []
	for i in xrange(len(x)):
		z_1 = []
		z_2 = []
		for j in xrange(len(x)):
			z_1.append(float(dens_u1.pdf([x[0][i],y[j][0]])))
			z_2.append(float(dens_u2.pdf([x[0][i],y[j][0]])))
		z1.append(z_1)
		z2.append(z_2)

	# surf = ax.plot_surface(x, y, z1, rstride=1, cstride=1, cmap=cm.coolwarm,
	#         linewidth=0, antialiased=False)
	wire1 = ax1.plot_wireframe(x,y,z1,rstride=1,cstride=1)
	wire2 = ax2.plot_wireframe(x,y,z2,rstride=1,cstride=1)

	plt.show()

def test11():
	X = np.array([1,2,4,3,2,8,8,9,10,12,11,9])
	Y = np.array([1,2,3,5,6,7,8,13,14,15,18,20])
	X1 = np.array([1,3,4,3,5,5,10,11,10,10,11,12])
	Y1 = np.array([1,2,6,5,6,7,8,17,16,17,18,20])

	dens_u1 = sm.nonparametric.KDEMultivariate(data=[X,Y],var_type='cc', bw='normal_reference')
	dens_u2 = sm.nonparametric.KDEMultivariate(data=[X1,Y1],var_type='cc', bw='normal_reference')

	# df1 = pd.DataFrame({'x':X, 'y':Y})
	# graph1 = sns.jointplot(x='x', y='y', data=df1, kind="kde")
	# df2 = pd.DataFrame({'x1':X1, 'y1':Y1})
	# graph2 = sns.jointplot(x='x1', y='y1', data=df2, kind="kde")

	fig = plt.figure()
	ax1 = fig.add_subplot(131, projection='3d')
	ax2 = fig.add_subplot(132, projection='3d')
	ax3 = fig.add_subplot(133, projection='3d')

	x = np.arange(0, 20, 0.25)
	y = np.arange(0, 20, 0.25)
	x,y = np.meshgrid(x, y)
	z1 = []
	z2 = []
	z3 = []
	for i in xrange(len(x)):
		z_1 = []
		z_2 = []
		z_3 = []
		for j in xrange(len(x)):
			z_1.append(float(dens_u1.pdf([x[0][i],y[j][0]])))
			z_2.append(float(dens_u2.pdf([x[0][i],y[j][0]])))
			z_3.append(z_1[j]*z_2[j])
		z1.append(z_1)
		z2.append(z_2)
		z3.append(z_3)
	print z1

	# surf = ax.plot_surface(x, y, z1, rstride=1, cstride=1, cmap=cm.coolwarm,
	#         linewidth=0, antialiased=False)
	wire1 = ax1.plot_wireframe(x,y,z1,rstride=1,cstride=1)
	wire2 = ax2.plot_wireframe(x,y,z2,rstride=1,cstride=1)
	wire3 = ax3.plot_wireframe(x,y,z3,rstride=1,cstride=1)
	fig.set_size_inches(20, 5, forward=True)

	plt.show()


###################################################################
#                              testing
###################################################################

# test1()
# test2()
# test3()
# test4()
# test5()
# test6()
# test7()
# test8()
# test9()
# test10()
test11()