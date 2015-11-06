import os
import csv
import numpy as np; np.random.seed(0)
import seaborn as sns; sns.set(color_codes=True)
import matplotlib.pyplot as plt
import pandas as pd

"""
Class for reading and writing data to csv files.
This class also provide some useful functions to
get detailed information of the data before using.
"""

######################################################################
# basic class
######################################################################

class Data:
    
    def __init__(self, X=None, y=None, row_count=0, csv_list=None):
        """
        Data class.
        
        Attributes
        --------------------
            X       -- numpy array of shape (n,d), features
            y       -- numpy array of shape (n,), targets
            row_count  -- integer, indicating # of data
        """
        
        # n = number of examples, d = dimensionality
        self.X = X
        self.y = y
        self.csv_list = csv_list
        self.row_count = row_count
    
    def load(self, filename, ify=True):
        """
        Load csv file into X array of features and y array of labels.
        
        Parameters
        --------------------
            filename -- string, filename
        """
        
        # determine filename
        dir = os.path.dirname(__file__)
        
        f = open(os.path.join(dir, '..', 'data', filename))
        self.csv_list = list(csv.reader(f))
        self.row_count = len(self.csv_list)
        lX = []
        ly = []
        for x in xrange(self.row_count):
        	if ify:
        		lX.append(self.csv_list[x][0:-1])
        	else:
        		lX.append(self.csv_list[x][:])
        	ly.append(self.csv_list[x][-1])
        self.X = np.array(lX)
        if ify:
        	self.y = np.array(ly)

    def getXy(self):
    	"""
    	Count for data in the csv file.
    	"""
    	return self.X, self.y

    def length(self):
    	"""
    	Count for data in the csv file.
    	"""
    	return self.row_count

    def getlist(self):
    	"""
    	Get original csv file.
    	"""
    	return self.csv_list
    
    def plot(self, **kwargs):
        """
        Plot data.
        """
        
        if 'color' not in kwargs:
            kwargs['color'] = 'b'
        
        plt.scatter(self.X, self.y, **kwargs)
        plt.xlabel('x', fontsize = 16)
        plt.ylabel('y', fontsize = 16)
        plt.show()


######################################################################
# wrapper and complex functions
######################################################################

# wrapper functions around Data class
def load_data(filename):
	"""
    Wrapper function for loading the csv file.
    """
	data = Data()
	data.load(filename)
	return data

def plot_data(X, y, **kwargs):
	"""
    Wrapper function for quickly ploting data, may not 
    work sometimes.
    """
	data = Data(X, y)
	data.plot(**kwargs)

def preview(filename):
	"""
    Preview the csv file reading in by printing the first
    few lines.
    """
	data = Data()
	data.load(filename)
	row_count = data.length()
	if row_count <= 10:
		for x in xrange(row_count):
			print data.getlist()[x]
	if row_count > 10:
		for x in xrange(10):
			print data.getlist()[x]
    	print ' '
    	print "Not all data displayed, the file is too large."
    	print ' '

def writeCSV(filename, datalist):
	"""
    Write out csv file from a list.
    """
	dir = os.path.dirname(__file__)
	f = os.path.join(dir, '..', 'data', filename)
	with open(f, 'w') as fp:
		a = csv.writer(fp, delimiter=',')
		a.writerows(datalist)

def extract(filename, col, start=0):
	"""
    Extract features from the reading in csv files.
    """
	feature = []
	data = Data()
	data.load(filename)
	row_count = data.length()
	thelist = data.getlist()
	for x in xrange(start,row_count):
		feature.append(thelist[x][col])
	return feature

