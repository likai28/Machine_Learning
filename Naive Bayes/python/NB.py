import math
import numpy as np

# The logProd function takes a vector of numbers in logspace 
# (i.e., x[i] = log p[i]) and returns the product of those numbers in logspace
# (i.e., logProd(x) = log(product_i p[i]))
def logProd(x):
	## Inputs ## 
	# x - 1D numpy ndarray
	
	## Outputs ##
	# log_product - float

	log_product = 0
	log_product = np.sum(x)

	return log_product

# The NB_XGivenY function takes a training set XTrain and yTrain and
# Beta parameters beta_0 and beta_1, then returns a matrix containing
# MAP estimates of theta_yw for all words w and class labels y
def NB_XGivenY(XTrain, yTrain, beta_0, beta_1):
	## Inputs ## 
	# XTrain - (n by V) numpy ndarray
	# yTrain - 1D numpy ndarray of length n
	# alpha - float
	# beta - float
	
	## Outputs ##
	# D - (2 by V) numpy ndarray

	D = np.zeros([2, XTrain.shape[1]])
	alpha_0 = np.dot(XTrain.transpose(),1-yTrain)
	alpha_1 = float(np.sum(1-yTrain))-alpha_0
	D[0,:] = (alpha_0+beta_0-1)/((alpha_0+beta_0-1)+(alpha_1+beta_1-1))

	alpha_a = np.dot(XTrain.transpose(),yTrain)
	alpha_b = float(np.sum(yTrain))-alpha_a
	D[1,:] = (alpha_a+beta_0-1)/((alpha_a+beta_0-1)+(alpha_b+beta_1-1))

	return D
	
# The NB_YPrior function takes a set of training labels yTrain and
# returns the prior probability for class label 0
def NB_YPrior(yTrain):
	## Inputs ## 
	# yTrain - 1D numpy ndarray of length n

	## Outputs ##
	# p - float

	p = 0
	n = yTrain.size
	alpha = n-np.count_nonzero(yTrain)
	p = float(alpha)/n

	return p

# The NB_Classify function takes a matrix of MAP estimates for theta_yw,
# the prior probability for class 0, and uses these estimates to classify
# a test set.
def NB_Classify(D, p, XTest):
	## Inputs ## 
	# D - (2 by V) numpy ndarray
	# p - float
	# XTest - (m by V) numpy ndarray
	
	## Outputs ##
	# yHat - 1D numpy ndarray of length m

	yHat = np.ones(XTest.shape[0])
	m = XTest.shape[0]
	for i in range(m):
		x1 = XTest[i,:]*np.log(D[0,:])
		x0 =(1-XTest[i,:])*(np.log(1-D[0,:]))

		y0 = logProd(x0)+ logProd(x1)+math.log(p)

		x1 = XTest[i,:]*np.log(D[1,:])
		x0 = (1-XTest[i,:])*np.log((1-D[1,:]))
		y1 = logProd(x0)+ logProd(x1)+math.log(1-p)

		if y0>y1:
			yHat[i]=0
		else:
			yHat[i]=1
	
	return yHat

# The classificationError function takes two 1D arrays of class labels
# and returns the proportion of entries that disagree
def classificationError(yHat, yTruth):
	## Inputs ## 
	# yHat - 1D numpy ndarray of length m
	# yTruth - 1D numpy ndarray of length m
	
	## Outputs ##
	# error - float

	error = 0

	for i in range(yHat.size):
		if yHat[i]!=yTruth[i]:
			error=error+1
	error = float(error)/yHat.size

	return error
