import numpy as np
import math
import random

in_dim = 785 # input dimension
out_dim = 10 # number of classes (0-9)
eta = 1 # Learning rate. You might try different rates (between 0.001 and 1) to maximize the accuracy

def Weight_update(feature, label, weight_i2o):
	##
	#Update the weights for a train feature.
		# Inputs:
			# feature: feature vector (ndarray) of a data point with 785 dimensions. Here, the feature represents a handwritten digit 
			         # of a 28x28 pixel grayscale image, which is flattened into a 785-dimensional vector (include bias)
			# label: Actual label of the train feature 
			# weight_i2o: current weights with shape (in_dim x out_dim) from input (feature vector) to output (digit number 0-9)
		# Return: updated weight
	##
	#"*** YOUR CODE HERE ***"
	result = np.dot(feature, weight_i2o)

	index = 0 # index of max y*
	for i in range(1, out_dim):
		if result[i] > result[index]:
			index = i

	if index == int(label): # if match -> no change
		return weight_i2o
		
	diff = np.zeros(out_dim) # t(x) - y(x)
	diff[int(label)] = 1.0
	diff[index] = -1.0

	
	feature_t = np.transpose([feature])
	diff_arr = np.array([diff])

	dot = np.dot(feature_t, diff_arr) # xT * (t(x) - y(x))
	weight_i2o = np.array(weight_i2o) + np.array(np.multiply(dot, eta)) # new weights
	return weight_i2o

def get_predictions(dataset, weight_i2o):
	#"""
	#Calculates the predicted label for each feature in dataset.
		# Inputs:
			# dataset: a set of feature vectors with shape  
			# weight_i2o: current weights with shape (in_dim x out_dim)
		# Return: list (or ndarray) of predicted labels from given dataset
	#"""
	#"*** YOUR CODE HERE ***"
	result = []
	for x in dataset:
		temp = np.dot(x, weight_i2o)
		index = 0
		for i in range(1, out_dim):
			if temp[i] > temp[index]:
				index = i
		result.append(index)
	return result
	

def train(train_set, labels, weight_i2o):
	#"""
	#Train the perceptron until convergence.
	# Inputs:
		# train_set: training set (ndarray) with shape (number of data points x in_dim)
		# labels: list (or ndarray) of actual labels from training set
		# weight_i2o:
	# Return: the weights for the entire training set
	#"""
    for i in range(0, train_set.shape[0]):        
        weight_i2o = Weight_update(train_set[i, :], labels[i], weight_i2o)        
    return weight_i2o