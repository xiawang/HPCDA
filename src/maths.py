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

def normpdf(x, mean, sd):
	var = float(sd)**2
	pi = 3.14159265358979323846
	denom = (2*pi*var)**0.5
	num = math.exp(-(float(x) - float(mean))**2 / (2*var))
	return num / denom



###################################################################
#                              testing
###################################################################

# prob = normpdf(50, 70, 5)
# print prob

def test1():
	obs_dist1 = mixture_rvs([.25,.75], size=10000, dist=[stats.norm, stats.norm],
				kwargs = (dict(loc=-1,scale=.5),dict(loc=1,scale=.5)))

	kde = sm.nonparametric.KDEUnivariate(obs_dist1)
	kde.fit()

	fig = plt.figure(figsize=(12,8))
	ax = fig.add_subplot(111)
	ax.hist(obs_dist1, bins=50, normed=True, color='red')
	ax.plot(kde.support, kde.density, lw=2, color='black')
	plt.show()


def test2():
	nobs = 300
	np.random.seed(1234)  # Seed random generator
	dens = sm.nonparametric.KDEUnivariate(np.random.normal(size=nobs))
	dens.fit()
	plt.plot(dens.cdf)
	plt.show()


def test3():
	np.random.seed(1234)
	hours = np.linspace(0,23,50)
	freq = np.concatenate([ norm(8,2.).rvs(100), norm(18,1.).rvs(100) ])
	 
	# Plot the kernel density estimates
	fig, ax = plt.subplots(1, 5, sharey=True, figsize=(18, 3))
	fig.subplots_adjust(wspace=0)
	 
	for (i,bw) in enumerate([0.2,0.5,1.0,2.0,5.0]):
		# sklearn
		kde_skl = KernelDensity(bandwidth=bw)
		kde_skl.fit(freq[:,np.newaxis])
		density = np.exp(kde_skl.score_samples(hours[:,np.newaxis]))
		ax[i].plot(hours, density, color='red', alpha=0.5, lw=3)
		ax[i].set_title('sklearn, bw={0}'.format(bw))
	 
		ax[i].set_xlim(0,23)
	plt.show()


def test4():
	X = [[1], [2], [3], [3], [2], [8], [8], [9], [10], [12], [11], [9]]
	kde = KernelDensity(kernel='gaussian', bandwidth=0.4).fit(X)
	scores = kde.score_samples(X)
	for x in xrange(len(scores)):
		scores[x] = math.exp(scores[x])
	print scores
	X_test = [[1], [2], [2], [1], [5], [6], [6], [7], [9], [10], [8], [7]]
	density = kde.score_samples(X_test)
	for x in xrange(len(density)):
		density[x] = math.exp(density[x])
	print density


def test5():
	X = [[1], [2], [3], [3], [2], [8], [8], [9], [10], [12], [11], [9]]
	kde = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(X)
	scores = kde.score_samples(X)
	print scores
	X_test = [[1], [2], [2], [1], [5], [6], [6], [7], [9], [10], [8], [7]]
	density = kde.score_samples(X_test)
	print density
	# print
	g = sns.distplot(X);
	h = sns.distplot(X_test);
	sns.plt.show();


def test6():
	X = [[1], [2], [3], [3], [2], [8], [8], [9], [10], [12], [11], [9]]
	X_1 = sum(X, [])
	kde1 = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(X)

	Y = [[1], [2], [2], [1], [5], [6], [6], [7], [9], [10], [8], [7]]
	Y_1 = sum(Y, [])
	kde2 = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(Y)

	# scores with own kde
	scores1 = kde1.score_samples(X)
	for x in xrange(len(scores1)):
		scores1[x] = math.exp(scores1[x])
	print scores1
	print ""
	scores2 = kde2.score_samples(Y)
	for x in xrange(len(scores2)):
		scores2[x] = math.exp(scores2[x])
	print scores2
	print ""

	# scores with others kde
	density1 = kde1.score_samples(Y)
	for x in xrange(len(density1)):
		density1[x] = math.exp(density1[x])
	print density1
	print ""
	density2 = kde2.score_samples(X)
	for x in xrange(len(density2)):
		density2[x] = math.exp(density2[x])
	print density2
	print ""

	# general test
	test_set = []
	for x in xrange(0,14):
		test_set.append([x])
	test_res1 = kde1.score_samples(test_set)
	for x in xrange(len(test_res1)):
		test_res1[x] = math.exp(test_res1[x])
	print test_res1
	test_res2 = kde2.score_samples(test_set)
	for x in xrange(len(test_res2)):
		test_res2[x] = math.exp(test_res2[x])
	print test_res2
	test_res3 = map(lambda x,y:abs(x-y), test_res1,test_res2)
	belonging = []
	for x in xrange(len(test_res1)):
		if test_res1[x] - test_res2[x] > 0:
			belonging.append(1)
		elif test_res1[x] - test_res2[x] < 0:
			belonging.append(2)
		else:
			belonging.append(3)
	for x in xrange(len(test_res3)):
		print x," ",test_res3[x]," - ",belonging[x]
	print ""

	# print
	g = sns.distplot(X);
	h = sns.distplot(Y);
	sns.plt.show();



# test1()
# test2()
# test3()
# test4()
# test5()
# test6()

# x = np.concatenate([norm(-1, 1.).rvs(400),
#                     norm(1, 0.3).rvs(100)])
# print x


#----------------------------------------------------------------------
# Plot the progression of histograms to kernels
# np.random.seed(1)
# N = 20
# X = np.concatenate((np.random.normal(0, 1, 0.3 * N),
#                     np.random.normal(5, 1, 0.7 * N)))[:, np.newaxis]
# X_plot = np.linspace(-5, 10, 1000)[:, np.newaxis]
# bins = np.linspace(-5, 10, 10)

# fig, ax = plt.subplots(2, 2, sharex=True, sharey=True)
# fig.subplots_adjust(hspace=0.05, wspace=0.05)

# # histogram 1
# ax[0, 0].hist(X[:, 0], bins=bins, fc='#AAAAFF', normed=True)
# ax[0, 0].text(-3.5, 0.31, "Histogram")

# # histogram 2
# ax[0, 1].hist(X[:, 0], bins=bins + 0.75, fc='#AAAAFF', normed=True)
# ax[0, 1].text(-3.5, 0.31, "Histogram, bins shifted")

# # tophat KDE
# kde = KernelDensity(kernel='tophat', bandwidth=0.75).fit(X)
# log_dens = kde.score_samples(X_plot)
# ax[1, 0].fill(X_plot[:, 0], np.exp(log_dens), fc='#AAAAFF')
# ax[1, 0].text(-3.5, 0.31, "Tophat Kernel Density")

# # Gaussian KDE
# kde = KernelDensity(kernel='gaussian', bandwidth=0.75).fit(X)
# log_dens = kde.score_samples(X_plot)
# ax[1, 1].fill(X_plot[:, 0], np.exp(log_dens), fc='#AAAAFF')
# ax[1, 1].text(-3.5, 0.31, "Gaussian Kernel Density")

# for axi in ax.ravel():
#     axi.plot(X[:, 0], np.zeros(X.shape[0]) - 0.01, '+k')
#     axi.set_xlim(-4, 9)
#     axi.set_ylim(-0.02, 0.34)

# for axi in ax[:, 0]:
#     axi.set_ylabel('Normalized Density')

# for axi in ax[1, :]:
#     axi.set_xlabel('x')

#----------------------------------------------------------------------
# Plot all available kernels
# X_plot = np.linspace(-6, 6, 1000)[:, None]
# X_src = np.zeros((1, 1))

# fig, ax = plt.subplots(2, 3, sharex=True, sharey=True)
# fig.subplots_adjust(left=0.05, right=0.95, hspace=0.05, wspace=0.05)


# def format_func(x, loc):
#     if x == 0:
#         return '0'
#     elif x == 1:
#         return 'h'
#     elif x == -1:
#         return '-h'
#     else:
#         return '%ih' % x

# for i, kernel in enumerate(['gaussian', 'tophat', 'epanechnikov',
#                             'exponential', 'linear', 'cosine']):
#     axi = ax.ravel()[i]
#     log_dens = KernelDensity(kernel=kernel).fit(X_src).score_samples(X_plot)
#     axi.fill(X_plot[:, 0], np.exp(log_dens), '-k', fc='#AAAAFF')
#     axi.text(-2.6, 0.95, kernel)

#     axi.xaxis.set_major_formatter(plt.FuncFormatter(format_func))
#     axi.xaxis.set_major_locator(plt.MultipleLocator(1))
#     axi.yaxis.set_major_locator(plt.NullLocator())

#     axi.set_ylim(0, 1.05)
#     axi.set_xlim(-2.9, 2.9)

# ax[0, 1].set_title('Available Kernels')

#----------------------------------------------------------------------
# Plot a 1D density example
# N = 100
# np.random.seed(1)
# X = np.concatenate((np.random.normal(0, 1, 0.3 * N),
# 					np.random.normal(5, 1, 0.7 * N)))[:, np.newaxis]

# X_plot = np.linspace(-5, 10, 1000)[:, np.newaxis]

# true_dens = (0.3 * norm(0, 1).pdf(X_plot[:, 0])
# 			 + 0.7 * norm(5, 1).pdf(X_plot[:, 0]))

# fig, ax = plt.subplots()
# ax.fill(X_plot[:, 0], true_dens, fc='black', alpha=0.2,
# 		label='input distribution')

# for kernel in ['gaussian', 'tophat', 'epanechnikov']:
# 	kde = KernelDensity(kernel=kernel, bandwidth=0.5).fit(X)
# 	log_dens = kde.score_samples(X_plot)
# 	ax.plot(X_plot[:, 0], np.exp(log_dens), '-',
# 			label="kernel = '{0}'".format(kernel))
# 	scores = kde.score_samples(X)
# 	for x in xrange(len(scores)):
# 		scores[x] = math.exp(scores[x])
# 	print scores
# 	print " "

# ax.text(6, 0.38, "N={0} points".format(N))

# ax.legend(loc='upper left')
# ax.plot(X[:, 0], -0.005 - 0.01 * np.random.random(X.shape[0]), '+k')

# ax.set_xlim(-4, 9)
# ax.set_ylim(-0.02, 0.4)
# plt.show()