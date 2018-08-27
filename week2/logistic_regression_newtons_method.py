# matrix math
import numpy as np 
# data manipulattion
import pandas as pd 
# matrix data structure
from pasty import dmatrices
# for error logging
import warnings

# Outputs probility between 0 and 1, used to help define our logistic regression curve
def sigmoid(x):
	'''Sigmoid function of x.'''
	return 1/(1+np.exp(-x))

# makes the random numbers predictable
# (pseudo-)random numbers work by starting with a number (seed),
# multiplying it by a large number, then taking modulo of that product.
# The resulting number is then used as the seed to generate the next "random" number.
# When you set the seed (every time), it does the same thing every time,
# giving you the same numbers.
# good for reproducting results for debugging

np.random.seed(0) # set the seed

## Step 1 - Define model parameters (hyperparameters)

## algorithm settings
# the minimum threshold for the difference between the predicted output
# and the actual output, this tells us our model when to stop learning
# when our prediction capability is good enough
tol = 1e-8 # convergence tolerance

lam = None # l2-regularization
# how long to train for?
max_iter m= 20 # maximum allowed iterations

## data creation settings
# Covariance measures how two variables move together.
# It measures whether the two move in the same direction (a positive covariance)
# or in opposite directions (a negative covaiance).
r = 0.95 # covariance between x and z
n = 1000 # number of observations (size of dataset to generate)
sigma = 1 # variance of noise - how spread out is the data?

## model settings
beta_x, beta_z, beta_v = -4, .9, 1 # true beta coefficients
var_x, var_z, var_v = 1, 1, 4 # variances of inputs

## the model specification you want to fit
formula = 'y ~ x + z + v + np.epp(x) + I(v**2 + z)'