import os
import csv
import numpy as np; np.random.seed(0)
import seaborn as sns; sns.set(color_codes=True)
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.grid_search import GridSearchCV
from sklearn.grid_search import RandomizedSearchCV
from sklearn import metrics

from sklearn.tree import DecisionTreeClassifier

"""
Class for tuning parameters. 
"""

######################################################################
# decision tree
######################################################################

# tune decision tree randomly or exhaustively
def tune_dt(X, y, mfeature_range, mdepth, cvi, metric, randm=True):
	"""
    Tune parameters for knn algorithm.
    
    Parameters
    --------------------
        X              -- 2D array, features
        y              -- array, labels (expected to be an empty array)
        mfeature_range -- range corresponding to number of features
        mdepth         -- range corresponding to max depth
        cvi            -- integer, number of folds
        metric         -- string, indicating metric for evaluation
        randm          -- boolean, indicating whether to do random 
                          grid search CV or not
    Returns
    --------------------
        best_params_   -- dictionary, containing optimal parameters
    """
	mfeature_range_grid = mfeature_range
	mdepth_grid = mdepth
	# create a parameter grid
	param_grid = dict(max_features=mfeature_range_grid, max_depth=mdepth_grid)
	dt = DecisionTreeClassifier(criterion='entropy')
	# instantiate and fit the grid
	grid = GridSearchCV(dt, param_grid, cv=cvi, scoring=metric)
	rand = RandomizedSearchCV(dt, param_grid, cv=cvi, scoring=metric, n_iter=10, random_state=5)
	if randm:
		rand.fit(X, y)
		# preview best stats
		print rand.best_score_
		print rand.best_params_
		return rand.best_params_
	else:
		grid.fit(X, y)
		# preview best stats
		print grid.best_score_
		print grid.best_params_
		return grid.best_params_


######################################################################
# logistic regression
######################################################################



######################################################################
# other metrics
######################################################################

# # generating confusion matrix
# y_true = [0,0,1,1]
# y_pred = [1,0,1,1]
# cmatrix = metrics.confusion_matrix(y_true, y_pred)
# # getting TP, TN, FP, FN
# tp = cmatrix[1][1]
# tn = cmatrix[0][0]
# fp = cmatrix[0][1]
# fn = cmatrix[1][0]
# # specificity and sensitivity
# sensitivity = tp/(tp+fn)
# specificity = tn/(tn+fp)

# X = np.array([[1, 5], [2, 6], [3, 7], [4, 8], [5, 9], [7, 2], [8, 3],
# 		          [2, 7], [3, 8], [5, 1], [6, 2], [7, 3], [8, 4], [9, 5]])
# y = np.array([1,1,1,1,1,1,1,
# 		          2,2,2,2,2,2,2])
# mfeature_range = range(1, 3)
# mdepth = range(1, 3)
# cvi = 3
# metric = 'accuracy'

# tune_dt(X, y, mfeature_range, mdepth, cvi, metric, randm=False)