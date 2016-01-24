import math
import random
import numpy as np; np.random.seed(0)
import matplotlib.pyplot as plt

def normpdf(x, mean, sd):
	var = float(sd)**2
	pi = 3.14159265358979323846
	denom = (2*pi*var)**0.5
	num = math.exp(-(float(x) - float(mean))**2 / (2*var))
	return num / denom

prob = normpdf(50, 70, 5)
print prob