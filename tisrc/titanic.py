"""
Class for applying ML algorithms and training data
for titanic data set.
"""

import math
import csv
from titanicutil import *

from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
from sklearn import metrics

######################################################################
# classes
######################################################################

class Classifier(object) :
    """
    Classifier interface.
    """
    
    def fit(self, X, y):
        raise NotImplementedError()
        
    def predict(self, X):
        raise NotImplementedError()


class MajorityVoteClassifier(Classifier) :
    
    def __init__(self) :
        """
        A classifier that always predicts the majority class.
        
        Attributes
        --------------------
            prediction_ -- majority class
        """
        self.prediction_ = None
    
    def fit(self, X, y) :
        """
        Build a majority vote classifier from the training set (X, y).
        
        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
            y    -- numpy array of shape (n,), target classes
        
        Returns
        --------------------
            self -- an instance of self
        """
        vals, counts = np.unique(y, return_counts=True)
        majority_val, majority_count = max(zip(vals, counts), key=lambda (val, count): count)
        self.prediction_ = majority_val
        return self
    
    def predict(self, X) :
        """
        Predict class values.
        
        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
        
        Returns
        --------------------
            y    -- numpy array of shape (n,), predicted classes
        """
        if self.prediction_ is None :
            raise Exception("Classifier not initialized. Perform a fit first.")
        
        n,d = X.shape
        y = [self.prediction_] * n 
        return y


class RandomClassifier(Classifier) :
    
    def __init__(self) :
        """
        A classifier that predicts according to the distribution of the classes.
        
        Attributes
        --------------------
            probabilities_ -- class distribution dict (key = class, val = probability of class)
        """
        self.probabilities_ = None
    
    def fit(self, X, y) :
        """
        Build a random classifier from the training set (X, y).
        
        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
            y    -- numpy array of shape (n,), target classes
        
        Returns
        --------------------
            self -- an instance of self
        """
        
        # set self.probabilities_ according to the training set
        # note that in this case there are only 2 classes
        vals, counts = np.unique(y, return_counts=True)
        numClasses = 2
        probs = [0]*numClasses
        totalCount = sum(counts)

        for index in range(numClasses):
            probs[index] = counts[index]*1.0 / totalCount

        self.probabilities_ = probs
        
        return self
    
    def predict(self, X, seed=1234) :
        """
        Predict class values.
        
        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
            seed -- integer, random seed
        
        Returns
        --------------------
            y    -- numpy array of shape (n,), predicted classes
        """
        if self.probabilities_ is None :
            raise Exception("Classifier not initialized. Perform a fit first.")
        np.random.seed(seed)
        
        # predict the class for each test example
        n,d = X.shape
        y = np.random.choice([0,1], n, p=self.probabilities_)
        
        return y


######################################################################
# functions
######################################################################

def plot_histogram(X, y, Xname, yname) :
    """
    Plots histogram of values in X grouped by y.
    
    Parameters
    --------------------
        X     -- numpy array of shape (n,d), feature values
        y     -- numpy array of shape (n,), target classes
        Xname -- string, name of feature
        yname -- string, name of target
    """
    
    # set up data for plotting
    targets = sorted(set(y))
    data = []; labels = []
    for target in targets :
        features = [X[i] for i in xrange(len(y)) if y[i] == target]
        data.append(features)
        labels.append('%s = %s' % (yname, target))
    
    # set up histogram bins
    features = set(X)
    nfeatures = len(features)
    test_range = range(int(math.floor(min(features))), int(math.ceil(max(features)))+1)
    if nfeatures < 10 and sorted(features) == test_range:
        bins = test_range + [test_range[-1] + 1] # add last bin
        align = 'left'
    else :
        bins = 10
        align = 'mid'
    
    plot
    plt.figure()
    n, bins, patches = plt.hist(data, bins=bins, align=align, alpha=0.5, label=labels)
    plt.xlabel(Xname)
    plt.ylabel('Frequency')
    plt.legend() #plt.legend(loc='upper left')
    plt.show()


def error(clf, X, y, ntrials=100, test_size=0.2) :
    """
    Computes the classifier error over a random split of the data,
    averaged over ntrials runs.
    
    Parameters
    --------------------
        clf         -- classifier
        X           -- numpy array of shape (n,d), features values
        y           -- numpy array of shape (n,), target classes
        ntrials     -- integer, number of trials
    
    Returns
    --------------------
        train_error -- float, training error
        test_error  -- float, test error
    """
    
    fullTrainError = 0.0
    fullTestError = 0.0
    sizeFrac = test_size

    for trial in range(0,ntrials):
        xTrainingSet, xTestSet, yTrainingSet, yTestSet = train_test_split(X, y, \
            test_size=sizeFrac, random_state=trial)

        # calculate training error
        clf.fit(xTrainingSet, yTrainingSet)
        y_pred_train = clf.predict(xTrainingSet)
        train_error = 1 - metrics.accuracy_score(yTrainingSet, y_pred_train, normalize=True)
        fullTrainError += train_error

        # calculate testing error
        y_pred_test = clf.predict(xTestSet)
        test_error = 1 - metrics.accuracy_score(yTestSet, y_pred_test, normalize=True)
        fullTestError += test_error

    train_error = fullTrainError/ntrials
    test_error = fullTestError/ntrials
    
    return train_error, test_error


def write_predictions(y_pred, filename, yname=None) :
    """Write out predictions to csv file."""
    out = open(filename, 'wb')
    f = csv.writer(out)
    if yname :
        f.writerow([yname])
    f.writerows(zip(y_pred))
    out.close()


######################################################################
# main
######################################################################

def main():
    # load Titanic dataset
    titanic = load_data("titanic_train.csv", header=1, predict_col=0)
    X = titanic.X; Xnames = titanic.Xnames
    y = titanic.y; yname = titanic.yname
    n,d = X.shape  # n = number of examples, d =  number of features
    
    
    # plot histograms of each feature
    print 'Plotting...'
    #for i in xrange(d) :
        #plot_histogram(X[:,i], y, Xname=Xnames[i], yname=yname)


    # evaluate training error of Decision Tree classifier
    clf = DecisionTreeClassifier(criterion="entropy") # create MajorityVote classifier, which includes all model parameters
    clf.fit(X, y)                  # fit training data using the classifier
    y_pred = clf.predict(X)        # take the classifier and run it on the training data
    train_error_random = 1 - metrics.accuracy_score(y, y_pred, normalize=True)
    print 'training error for Decision Tree classifier: %.3f' % train_error_random

    
    # note: uncomment out the following lines to output the Decision Tree graph
    """
    # save the classifier -- requires GraphViz and pydot
    import StringIO, pydot
    from sklearn import tree
    dot_data = StringIO.StringIO()
    tree.export_graphviz(clf, out_file=dot_data,
                         feature_names=Xnames)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf("dtree.pdf") 
    """
    
    # evaluate training error of Decision Tree classifier
    clf = DecisionTreeClassifier(criterion="entropy") # create MajorityVote classifier, which includes all model parameters
    train_error, test_error = error(clf, X, y, ntrials = 100, test_size=0.2)
    print 'DecisionTree -- training error: '+str(train_error)+', test error: '+str(test_error)
 
    print 'Done'


if __name__ == "__main__":
    main()